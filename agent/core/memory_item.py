from typing import Literal
from pydantic import BaseModel


class MemoryItem(BaseModel):
    """一条待入库的长期记忆（由记忆抽取 LLM 返回，属外部数据，入口处用 pydantic 强校验）。

    source_type 的合法值与 memory_add 工具的 schema 枚举保持一致
    （见 tools/builtin/memory_add_tool.py）；Literal 在 pydantic 中是运行时强校验。
    """

    content: str
    source_type: Literal["user_preference", "user_fact", "tool_result", "conversation"]  # 强制校验
    evidence: str = ""
