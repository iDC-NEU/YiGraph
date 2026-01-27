from __future__ import annotations

from typing import Any, Dict, List, Optional
import json


def _format_errors(error_history: List[Dict[str, Any]], keep_last_k: int = 2) -> str:
    if not error_history:
        return ""
    recent = error_history[-keep_last_k:]

    # 只取关键信息，避免 prompt 膨胀
    simplified = []
    for e in recent:
        simplified.append({
            "error_type": e.get("error_type"),
            "error": e.get("error"),
            "location": e.get("location"),
            "hint": e.get("hint"),
        })
    return json.dumps(simplified, ensure_ascii=False, indent=2)


def enhance_prompt(
    base_prompt: str,
    error_history: List[Dict[str, Any]],
    *,
    operation_type: str,
    keep_last_k: int = 2,
    extra_constraints: Optional[str] = None,
) -> str:
    """
    最小化增强：在 base_prompt 末尾附加“修复模式”段落。

    operation_type 用于区分不同任务，可做轻量差异化约束（目前只做标签）。
    """
    if not error_history:
        return base_prompt

    error_blob = _format_errors(error_history, keep_last_k=keep_last_k)

    # 可以按 operation_type 给不同的“输出约束”提示
    # 最小实现：只提示“严格输出 JSON / 不要额外文本”等
    common_constraints = (
        "You are in REPAIR MODE.\n"
        "1) Use the error(s) below to fix the issue.\n"
        "2) Output MUST follow the required schema/format of the original task.\n"
        "3) Do NOT add any extra explanations.\n"
    )

    if extra_constraints:
        common_constraints += f"{extra_constraints.strip()}\n"

    repair_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ REPAIR MODE ({operation_type})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Previous attempt errors (most recent last):
{error_blob}

Repair requirements:
{common_constraints}
Now regenerate the output strictly following the original requirements.
"""

    return base_prompt + repair_section