from tools.tool import Tool
from memory.memory_store import memory_store


class MemoryQueryTool(Tool):
    """
    跨会话长期记忆检索工具（暴露给对话 LLM）：
    它感知当前上下文的信息缺口，按 description 的触发条件自行决定是否检索。
    """

    name = "memory_query"
    expose_to_llm = True

    description = (
        "跨会话长期记忆检索工具：读取框架自动保存的用户长期记忆（偏好/事实/之前调用工具的结果/对话纪要）。"
        "必须 query 的情形：在当前上下文找不到答案，且问题涉及用户个人偏好、个人信息、过去会话内容，"
        "禁止 query 的情形：所需信息已在当前上下文出现时直接回答；需要实时信息请用 search工具，需要本地知识库信息请用 rag_query。"
        "提示：可用 source_type 参数按记忆类型过滤搜索（user_preference/user_fact/tool_result/conversation）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要检索的关键词描述",
            },
            "source_type": {
                "type": "string",
                "enum": ["user_preference", "user_fact", "tool_result", "conversation"],
                "description": "记忆类型（可选过滤条件）：user_preference=用户偏好；user_fact=用户事实/个人信息；tool_result=工具/检索得到的知识快照；conversation=跨会话对话纪要",
            },
        },
        "required": ["query"],
    }

    def run(self, **kwargs) -> list[str]:
        query = kwargs.get("query")
        if not query:
            return ["错误：未提供检索关键词"]
        return memory_store.query(query, source_type=kwargs.get("source_type"))


# --- 工具使用示例 ---
if __name__ == "__main__":
    tool = MemoryQueryTool()
    print(tool.to_openai_schema())
    print("\n".join(tool.run(query="用户喝咖啡的习惯")))
