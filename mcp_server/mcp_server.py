"""MCP 通用 Server（streamable-http 传输，端口 8010）。

一个可扩展的 MCP Server：通过 @mcp.tool() 装饰器可任意注册新的工具函数，
供 MCP Client（tools/builtin/mcp_tool.py 适配器 + tools/tool_manager.py 的 sync_mcp_tools() 发现层）远程调用。
新增能力时，只需在本文件再加一个 @mcp.tool() 函数，客户端会自动发现。

运行方式（在 env_xp 环境中，先启动本 Server）：
    python mcp_server/mcp_server.py
"""

import os
from typing import Annotated
from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from mcp.server import MCPServer
from tavily import TavilyClient

# 从 mcp/ 目录向上找到项目根目录的 .env，读取 TAVILY_API_KEY
load_dotenv(find_dotenv())

# 创建一个 MCP Server 实例（第一个参数是服务器名称）
mcp = MCPServer(
    "McpServer",
    instructions="通用 MCP 服务，供 MCP Client 远程调用。",
)

@mcp.tool()
def search(query: Annotated[str, Field(description="搜索查询词，用于在网页中检索相关信息")]) -> str:
    """网页实时搜索工具：用于获取模型训练数据之外的最新信息。query 用简洁关键词（可中英混合），一次搜索一个主题。"""
    if not query:
        return "错误：未提供搜索查询词"

    print(f"🔍 [MCP Server] 正在执行 Tavily 网页搜索: {query}")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY"

    try:
        response = TavilyClient(api_key=api_key).search(
            query=query, search_depth="basic", include_answer=True, max_results=3
        )

        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果（每条结果占一行）
        formatted_results = [
            f"- {result['title']}: {result['content']}"
            for result in response.get("results", [])
        ]
        if not formatted_results:
            return "抱歉，没有找到相关信息。"
        return "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题 - {e}"


if __name__ == "__main__":
    # 手动在终端启动：streamable-http 传输，监听 127.0.0.1:8010/mcp
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8010,
        streamable_http_path="/mcp",
    )