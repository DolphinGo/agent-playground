from tools.tool import Tool
from rag.rag_store import rag_store


class RagQueryTool(Tool):
    """
    知识库（RAG）检索工具（暴露给对话 LLM）：
    当用户问题涉及"本地知识库中的既定知识/资料"且当前上下文不足以回答时，
    按 query 检索知识库，返回最相关的若干 chunk，作为上下文供 LLM 作答（RAG 经典流程）。
    内部经历了「向量召回 top_k → rerank 重排 top_n」的两阶段检索。
    """

    name = "rag_query"
    expose_to_llm = True

    description = (
        "本地知识库（RAG）检索工具："
        "返回最相关的若干片段（每条带来源与相关度标注）。"
        "必须 query 的情形：问题依赖本地知识库中的既定知识（公司文档、内部资料）且当前上下文不足，也不属于memory_query时。"
        "禁止 query 的情形：所需信息已在当前上下文、或应由 search（实时）/ memory_query（用户记忆）回答时。"
        "检索无结果时基于已有信息继续回答，不要反复调用。"
        "提示：query 应是与知识库内容匹配的关键词或完整问题，不应是口语化的闲聊。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要检索的知识库问题或关键词描述",
            },
        },
        "required": ["query"],
    }

    def run(self, **kwargs) -> list[str]:
        query = kwargs.get("query")
        if not query:
            return ["错误：未提供检索关键词"]
        results = rag_store.query(query)
        if not results:
            return ["知识库中未找到与问题相关的内容。"]
        # 每条返回一个元素，标注来源，便于 LLM 引用
        formatted = [
            f"[来源: {r.source}, 相关度 {r.score:.3f}]\n{r.text}"
            for r in results
        ]
        return formatted


# --- 工具使用示例 ---
if __name__ == "__main__":
    tool = RagQueryTool()
    print(tool.to_openai_schema())
    print("\n".join(tool.run(query="NBA 的常规赛规则是什么")))