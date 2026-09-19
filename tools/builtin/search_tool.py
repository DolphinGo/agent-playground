import logging
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from tools.tool import Tool

# 加载 .env 文件中的环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """
    基于 Tavily 的网页搜索工具。
    一个 Tool 实例 = LLM 可见的一个函数，只实现 run() 这一个入口。
    """

    name = "search"
    description = (
        "网页实时搜索工具：用于获取模型训练数据之外的最新信息。"
        "必须 search 的情形：用户询问时事新闻、最新事件、实时行情/比分/天气，或需要验证某个最新事实时。"
        "禁止 search 的情形：常识问题、模型已知的知识、当前上下文已覆盖的信息、涉及用户个人/历史信息用memory_query，依赖本地知识库用 rag_query。"
        "提示：query 用简洁关键词（可中英混合），一次搜索一个主题。"
    )
    # openai schema 格式，用于 LLM 调用
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询词，用于在网页中检索相关信息"}
        },
        "required": ["query"],
    }

    def run(self, **kwargs) -> list[str]:
        query = kwargs["query"]
        if not query:
            return ["错误：未提供搜索查询词"]

        logger.info(f"🔍 正在执行 [Tavily] 网页搜索: {query}")

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return ["错误：未配置TAVILY_API_KEY"]

        tavily = TavilyClient(api_key=api_key)

        try:
            response = tavily.search(
                query=query, search_depth="basic", include_answer=True, max_results=3
            )

            # response['answer'] 是一个基于所有搜索结果的总结性回答
            if response.get("answer"):
                return [response["answer"]]

            # 如果没有综合性回答，则格式化原始结果（每条结果 = list 中的一个元素）
            formatted_results = [
                f"- {result['title']}: {result['content']}"
                for result in response.get("results", [])
            ]
            if not formatted_results:
                return ["抱歉，没有找到相关信息。"]

            return formatted_results

        except Exception as e:
            return [f"错误：执行Tavily搜索时出现问题 - {e}"]


# --- 工具使用示例 ---
if __name__ == "__main__":
    tool = SearchTool()
    print(tool.to_openai_schema())
    print("\n--- 执行 search ---")
    print("\n".join(tool.run(query="2026世界杯冠军是谁")))
