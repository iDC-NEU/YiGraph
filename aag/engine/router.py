from enum import Enum
from dataclasses import dataclass
from typing import Optional
import json
from aag.reasoner.model_deployment import Reasoner
from aag.reasoner.prompt_template.llm_prompt_en import query_router_prompt

class QueryType(str, Enum):
    GRAPH = "graph"          
    RAG = "rag"              
    GENERAL = "general"      


@dataclass
class RouteDecision:
    query_type: QueryType
    reason: str              # LLM 给出的简要理由（可用于 debug/log）    


class QueryRouter:
    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner

    def route(
        self,
        query: str,
    ) -> RouteDecision:

        messages = [
            {"role": "system", "content": query_router_prompt},
            {
                "role": "user",
                "content": query
            },
        ]

        raw = self.reasoner.chat(messages)

        try:
            data = json.loads(raw)
            qtype = QueryType(data.get("type", "general"))
            reason = data.get("reason", "")
        except Exception:
            qtype = QueryType.GENERAL
            reason = f"Failed to parse LLM output: {raw}"

        return RouteDecision(query_type=qtype, reason=reason)    