from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time


@dataclass
class PromptTraceItem:
    fn: str
    base_prompt: str
    meta: Dict[str, Any]
    ts: float


class PromptTraceBuffer:
    """
    运行时记录“实际渲染后的 base_prompt”，用于：
    - debug / 可观测性
    - retry 时按 fn_name 找到最近一次 base_prompt，再注入错误信息构造 enhanced_prompt
    """
    def __init__(self, maxlen: int = 200):
        self.maxlen = maxlen
        self.items: List[PromptTraceItem] = []

    def record(self, fn: str, base_prompt: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not base_prompt:
            return
        self.items.append(
            PromptTraceItem(fn=fn, base_prompt=base_prompt, meta=meta or {}, ts=time.time())
        )
        if len(self.items) > self.maxlen:
            self.items = self.items[-self.maxlen:]

    def last_prompt(self, fn: str) -> Optional[str]:
        for it in reversed(self.items):
            if it.fn == fn:
                return it.base_prompt
        return None

    def last_item(self, fn: str) -> Optional[PromptTraceItem]:
        for it in reversed(self.items):
            if it.fn == fn:
                return it
        return None

    def clear(self) -> None:
        self.items.clear()