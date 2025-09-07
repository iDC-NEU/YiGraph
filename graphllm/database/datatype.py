from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class VertexData:
    vid: int
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeData:
    src: int
    dst: int
    rank: Optional[int] = None  # 可选的 rank 值
    properties: Dict[str, Any] = field(default_factory=dict)
