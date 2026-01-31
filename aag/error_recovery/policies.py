from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class RetryPolicy:
    max_attempts: int = 3          # 总尝试次数（包含第一次）
    max_error_history: int = 2     # error_history 最多保留几条
    keep_last_k_in_prompt: int = 2 # 注入 prompt 的错误条数
    extra_constraints: Optional[str] = None


DEFAULT_POLICY = RetryPolicy()

# 可选：按 operation_type 定制（现在给个最小示例）
OPERATION_POLICIES: Dict[str, RetryPolicy] = {
    # "parameter_extraction": RetryPolicy(extra_constraints="Return JSON only."),
    # "dependency_resolution": RetryPolicy(extra_constraints="Return JSON only."),
    # "code_generation": RetryPolicy(extra_constraints="Return python code only."),
}


def get_policy(operation_type: str) -> RetryPolicy:
    return OPERATION_POLICIES.get(operation_type, DEFAULT_POLICY)