from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class VertexData:
    vid: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为标准字典结构，便于 JSON 传输或算法输入"""
        return {
            "vid": self.vid,
            "properties": self.properties
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VertexData":
        """从字典反序列化为 VertexData"""
        return VertexData(
            vid=data.get("vid"),
            properties=data.get("properties", {})
        )

    def __repr__(self):
        return f"VertexData(vid={self.vid}, props={list(self.properties.keys())})"


@dataclass
class EdgeData:
    src: str
    dst: str
    rank: Optional[int] = None  # 可选的 rank 值
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为标准字典结构，便于 JSON 传输或算法输入"""
        return {
            "src": self.src,
            "dst": self.dst,
            "rank": self.rank,
            "properties": self.properties
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EdgeData":
        """从字典反序列化为 EdgeData"""
        return EdgeData(
            src=data.get("src"),
            dst=data.get("dst"),
            rank=data.get("rank"),
            properties=data.get("properties", {})
        )

    def __repr__(self):
        return f"EdgeData({self.src}->{self.dst}, props={list(self.properties.keys())})"