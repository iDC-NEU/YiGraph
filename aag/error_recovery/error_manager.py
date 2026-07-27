from __future__ import annotations

import logging
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aag.error_recovery.trace import PromptTraceBuffer
from aag.error_recovery.enhancer import enhance_prompt
from aag.error_recovery.policies import get_policy

logger = logging.getLogger(__name__)


def prepare_error_info(
    error: Exception, *, location: Optional[str] = None, hint: Optional[str] = None
) -> Dict[str, Any]:
    info = {
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    if location:
        info["location"] = location
    if hint:
        info["hint"] = hint
    return info


class ErrorRecovery:
    def __init__(self, *, trace_maxlen: int = 200):
        self.trace = PromptTraceBuffer(maxlen=trace_maxlen)

    # ---------- Trace ----------
    def record_prompt(
        self, fn_name: str, base_prompt: str, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        self.trace.record(fn_name, base_prompt, meta=meta)

    def get_last_base_prompt(self, fn_name: str) -> Optional[str]:
        return self.trace.last_prompt(fn_name)

    # ---------- Prompt Enhancing ----------
    def build_enhanced_prompt(
        self,
        *,
        fn_name: str,
        error_history: List[Dict[str, Any]],
        operation_type: str = "generic",
    ) -> Optional[str]:
        base_prompt = self.get_last_base_prompt(fn_name)
        if not base_prompt:
            return None

        policy = get_policy(operation_type)
        return enhance_prompt(
            base_prompt,
            error_history,
            operation_type=operation_type,
            keep_last_k=policy.keep_last_k_in_prompt,
            extra_constraints=policy.extra_constraints,
        )

    # ---------- Retry (block-level) ----------
    async def run(
        self,
        operation: Callable[[List[Dict[str, Any]]], Awaitable[Any]],
        *,
        name: str,
        operation_type: str = "generic",
        location: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Any:
        """
        带重试的操作执行器。

        参数：
            operation: 接收 error_history 并返回结果的异步操作。
            name: 操作名称，用于日志和 prompt 跟踪。
            operation_type: 操作类型，用于策略选择。
            location: 错误发生位置标识。
            prompt: 操作的原始 prompt，用于记录和增强重试。
        """
        policy = get_policy(operation_type)
        max_attempts = policy.max_attempts
        max_error_history = policy.max_error_history

        error_history: List[Dict[str, Any]] = []
        last_exc: Optional[Exception] = None

        for attempt in range(max_attempts + 1):
            # 首次尝试前记录原始 prompt；重试时构建增强 prompt
            if attempt == 0 and prompt:
                self.record_prompt(
                    name,
                    prompt,
                    meta={"operation_type": operation_type, "location": location},
                )
            elif attempt > 0:
                enhanced = self.build_enhanced_prompt(
                    fn_name=name,
                    error_history=error_history,
                    operation_type=operation_type,
                )
                if enhanced:
                    error_history.append(
                        {
                            "type": "enhanced_prompt",
                            "content": enhanced,
                            "attempt": attempt + 1,
                        }
                    )

            try:
                result = await operation(error_history)
                if attempt > 1:
                    logger.info(
                        "✅ %s succeeded on attempt %d/%d", name, attempt, max_attempts
                    )
                return result
            except Exception as e:
                last_exc = e
                err = prepare_error_info(e, location=location)
                error_history.append(err)
                error_history[:] = error_history[-max_error_history:]

                if attempt < max_attempts:
                    logger.warning(
                        "⚠️ %s failed on attempt %d/%d: %s",
                        name,
                        attempt,
                        max_attempts,
                        err["error"],
                    )
                else:
                    logger.error("❌ %s failed after %d attempts", name, max_attempts)

        raise last_exc

    # ---------- Cross-step recovery (DAG-level) ----------
    def _collect_descendants(self, dag: Any, step_id: int) -> List[int]:
        """Collect all descendant step ids via dag.children_of()."""
        descendants: set[int] = set()
        queue: List[int] = list(dag.children_of(step_id))
        while queue:
            cur = queue.pop()
            if cur in descendants:
                continue
            descendants.add(cur)
            queue.extend(dag.children_of(cur))
        return list(descendants)

    def decide_cross_step_recovery(
        self,
        dag: Any,
        step_id: int,
        error_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        Decide which DAG steps should be reset to "pending" for a new round.

        This is intentionally rule-based (minimal & safe) so the recovery module
        can be evolved without changing scheduler control flow.
        """
        context = context or {}
        error_type = (error_info or {}).get("error_type") or ""
        stage = (error_info or {}).get("stage") or ""

        # Always include current failing step.
        target: set[int] = {step_id}

        if error_type == "DEPENDENCY_EMPTY" or stage == "dependency_resolve":
            # Dependency resolution is empty: rerun its upstream providers and
            # anything that depends on this step output.
            for pid in dag.parents_of(step_id):
                target.add(pid)
                target.update(dag.ancestors_of(pid))
            target.update(self._collect_descendants(dag, step_id))
            return list(target)

        if error_type in {"NUMERIC_EXEC_FAIL", "GRAPH_EXEC_FAIL", "UPSTREAM_FAILED"}:
            # Rerun from upstream context to ensure generated parameters/tools see
            # consistent inputs.
            target.update(dag.ancestors_of(step_id))
            target.update(self._collect_descendants(dag, step_id))
            return list(target)

        # Fallback: only rerun current node.
        return [step_id]

    def apply_cross_step_recovery(
        self,
        dag: Any,
        target_step_ids: List[int],
        *,
        reason: str,
    ) -> None:
        """Reset target steps in DAG for another global round."""
        for tid in target_step_ids:
            try:
                dag.set_pending(tid)
                logger.info(
                    f"🔄 Cross-step recovery: reset step {tid} -> pending ({reason})"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to reset step {tid} to pending: {e}")

    def global_recover(
        self,
        dag: Any,
        step_id: int,
        error_info: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> List[int]:
        """
        Decide + apply cross-step recovery in one call.
        Returns the target step ids that were reset.
        """
        targets = self.decide_cross_step_recovery(
            dag, step_id, error_info, context=context
        )
        if not targets:
            return []
        self.apply_cross_step_recovery(
            dag,
            targets,
            reason=reason
            or f"cross_step_recovery(error_type={error_info.get('error_type')})",
        )
        return targets
