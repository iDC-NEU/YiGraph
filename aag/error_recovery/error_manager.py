from __future__ import annotations

import logging
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aag.error_recovery.trace import PromptTraceBuffer
from aag.error_recovery.enhancer import enhance_prompt
from aag.error_recovery.policies import get_policy

logger = logging.getLogger(__name__)


def prepare_error_info(error: Exception, *, location: Optional[str] = None, hint: Optional[str] = None) -> Dict[str, Any]:
    """
    最小结构化错误信息：够用、可扩展
    """
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
    """
    统一入口（现在支持：块级重试 + prompt 增强 + base_prompt trace）
    未来你可以在这里加：多步回滚（execute_step / rollback_to / checkpoint 等）
    """
    def __init__(self, *, trace_maxlen: int = 200):
        self.trace = PromptTraceBuffer(maxlen=trace_maxlen)

    # ---------- Trace ----------
    def record_prompt(self, fn_name: str, base_prompt: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.trace.record(fn_name, base_prompt, meta=meta)

    def get_last_base_prompt(self, fn_name: str) -> Optional[str]:
        return self.trace.last_prompt(fn_name)

    # ---------- Prompt Enhancing ----------
    def build_enhanced_prompt(
        self,
        *,
        fn_name: str,
        operation_type: str,
        error_history: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        从 trace 获取 base_prompt，再注入错误信息。
        如果 trace 没记录到 prompt，则返回 None（调用方可退回到“自己重建 base_prompt”）
        """
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
    ) -> Any:
        """
        块级重试：operation 必须包含完整流程（构建prompt->调用reasoner->处理/校验）
        重试时会从 operation 开头重新执行整段逻辑（符合你的要求）。
        """
        policy = get_policy(operation_type)
        max_attempts = policy.max_attempts
        max_error_history = policy.max_error_history

        error_history: List[Dict[str, Any]] = []
        last_exc: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                result = await operation(error_history)
                if attempt > 1:
                    logger.info("✅ %s succeeded on attempt %d/%d", name, attempt, max_attempts)
                return result
            except Exception as e:
                last_exc = e
                err = prepare_error_info(e, location=location)
                error_history.append(err)
                error_history[:] = error_history[-max_error_history:]

                if attempt < max_attempts:
                    logger.warning("⚠️ %s failed on attempt %d/%d: %s", name, attempt, max_attempts, err["error"])
                else:
                    logger.error("❌ %s failed after %d attempts", name, max_attempts)

        raise last_exc