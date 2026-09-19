from tools.tool import Tool
from memory.memory_store import memory_store


class MemoryAddTool(Tool):
    """
    跨会话记忆写入工具（仅供框架内部调用，不暴露给 LLM）：
    把一条信息存入长期记忆。schema 只需 content + source_type，非常干净。

    为什么不让对话 LLM 调用："该不该记"应由框架侧的专门抽取 LLM 决定（见 Agent._remember），
    对话 LLM 看不到向量库已有内容、也无法判断信息的未来价值。
    """

    name = "memory_add"
    expose_to_llm = False  # 内部工具：不在 get_available_tools* 中暴露给 LLM

    description = (
        "内部工具：把一条信息存入跨会话长期记忆（仅由框架内部调用，不暴露给对话 LLM）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要存入记忆的内容",
            },
            "source_type": {
                "type": "string",
                "enum": ["user_preference", "user_fact", "tool_result", "conversation"],
                "description": "记忆类型：user_preference=用户偏好；user_fact=用户事实/个人信息；tool_result=工具/检索得到的知识快照；conversation=跨会话对话纪要",
            },
        },
        "required": ["content"],
    }

    def run(self, **kwargs) -> list[str]:
        content = kwargs.get("content")
        if not content:
            return ["错误：未提供记忆内容"]
        return memory_store.add(content, source_type=kwargs.get("source_type"))


# --- 工具使用示例 ---
if __name__ == "__main__":
    tool = MemoryAddTool()
    print(tool.to_openai_schema())
    print("\n".join(tool.run(content="用户喜欢喝美式咖啡，不加糖", source_type="user_preference")))
