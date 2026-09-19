"""消息系统"""
import json
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

# 定义消息角色的类型
# Literal 表示"变量的值只能是指定的几个固定值之一"。
# 注意：在 pydantic 模型（如 Message.role）中 Literal 会被运行时强制校验；
# 而在普通函数签名里它只服务于 Pyright 等静态类型检查，不是运行时限制。
MessageRole = Literal["user", "assistant", "system", "tool"]

class ToolCall(BaseModel):
    """工具调用：对应assistant消息中tool_calls数组的元素"""
    id: str
    name: str
    arguments: dict[str, Any] | str  # API返回的是JSON字符串，本地构造可传dict

    def get_arguments_dict(self) -> dict[str, Any]:
        """解析出参数字典，用于实际执行工具"""
        if isinstance(self.arguments, str):
            return json.loads(self.arguments) if self.arguments else {}
        return self.arguments

    def to_api_dict(self) -> dict[str, Any]:
        """转换为OpenAI API要求的格式（arguments必须是JSON字符串）"""
        args = self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments, ensure_ascii=False) # "JSON 里保留中文原样，不转成 \uXXXX"，主要为了可读性和省 token。
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": args},
        }

    def __str__(self) -> str:
        """友好展示：工具名(参数)"""
        return f"{self.name}[{self.arguments}]"


class Message(BaseModel):
    """通用消息类

    覆盖OpenAI四种消息角色：
    - system / user: Message(role, content)
    - assistant带工具调用: Message(role, content, tool_calls=[ToolCall(...)])
    - 工具结果: Message(role, content, tool_call_id="call_xxx")
    """

    role: MessageRole
    content: str|None = None
    reasoning_content: str|None = None  # 推理模型的思维链（DeepSeek 思考模式），仅 assistant 消息使用；带工具调用的轮次在后续请求中必须回传，否则 API 400
    tool_calls: list[ToolCall]|None = None  # 仅assistant消息使用
    tool_call_id: str|None = None          # 仅tool消息使用
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")) # 默认生成

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（OpenAI API格式），只输出API认可的字段"""
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_api_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def __str__(self) -> str:
        return f"({self.timestamp}) [{self.role}] {self.content}"
