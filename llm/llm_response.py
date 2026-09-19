from typing import Dict, List
from pydantic import BaseModel, Field
from agent.core.message import ToolCall

class LLMResponse(BaseModel):
    """调用LLM响应对象（由 LLMClient 从 SDK 响应内部构造，Agent 层只读使用）"""
    model: str
    finish_reason: str|None
    content: str|None
    reasoning_content: str|None
    tool_calls: List[ToolCall]
    latency_s: float
    usage: Dict[str, int] = Field(default_factory=dict)

    def __str__(self) -> str:
        return self.content or self.reasoning_content or ""
    
    def __repr__(self) -> str:
        """详细信息展示"""
        parts = [
            f"LLMResponse(model={self.model}",
            f"latency={self.latency_s}s",
            f"tokens={self.usage.get('total_tokens', 0)}",
            f"finish_reason={self.finish_reason}"
        ]
        parts.append(f"has_tool_calls={len(self.tool_calls)}")
        parts.append(f"content_length={len(self.content or '')}")
        parts.append(f"reasoning_content_length={len(self.reasoning_content or '')}")
        return ", ".join(parts)
    
    def to_dict(self) -> Dict:
        result = {
            "model": self.model,
            "finish_reason": self.finish_reason,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": [tc.name for tc in self.tool_calls],
            "usage": self.usage,
            "latency_s": self.latency_s,
        }
        return result
