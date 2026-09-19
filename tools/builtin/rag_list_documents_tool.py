from tools.tool import Tool
from rag.rag_store import rag_store


class RagListDocumentsTool(Tool):
    """
    知识库文档清单查询工具（仅供框架内部调用，不暴露给 LLM）：
    统计当前 RAG 知识库收录了哪些文档（source）及各自 chunk 数量，
    用于把「知识库覆盖范围」动态注入系统提示词，让 LLM 知道 rag_query 能查到哪些资料。

    为什么做成内部工具而非普通函数：它属于"访问向量库"的能力，
    与 memory_add 一样放在 Tool 体系内可复用、可被 ToolManager 统一执行，
    但不应暴露给对话 LLM（LLM 没有理由主动查询知识库目录）。
    """

    name = "rag_list_documents"
    expose_to_llm = False  # 内部工具：仅在框架构建系统提示词时由 ToolManager 调用

    description = (
        "内部工具：查询当前本地知识库收录了哪些文档及各自片段数（仅由框架内部调用，不暴露给对话 LLM）。"
    )
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs) -> list[str]:
        counts = rag_store.list_sources()
        if not counts:
            return ["（当前为空）"]
        return ["、".join(f"《{src}》（{n} 个片段）" for src, n in counts.items())]


# --- 工具使用示例 ---
if __name__ == "__main__":
    tool = RagListDocumentsTool()
    print("\n".join(tool.run()))