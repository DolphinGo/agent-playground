"""MCPTool：把远程 MCP Server 暴露的某个工具翻译成本地 Tool 契约的通用适配器。

与 tools/tool_manager.py 的分工：
    本 MCPTool           ：一个实例对应远端一个函数，只管 run() 单函数执行；
    tool_manager.py      ：sync_mcp_tools() 连接 server、list_tools() 发现全部函数，
                           并批量生成 MCPTool 实例后注册。
"""

import asyncio
from mcp import Client
from mcp.types import TextContent
from tools.tool import Tool


class MCPTool(Tool):
    """把一个远程 MCP 工具包装成本地 Tool。

    args:
        server_url:  MCP Server 的 streamable-http 地址，如 "http://127.0.0.1:8010/mcp"
        name:        远端工具名（LLM 调用时使用的函数名，需全局唯一）
        description: 远端工具描述（告诉 LLM 何时该用）
        parameters:  远端工具的 input_schema（JSON Schema 格式）
    """

    expose_to_llm = True  # 所有 MCP 工具都暴露给 LLM 选择

    def __init__(self, server_url: str, name: str, description: str, parameters: dict):
        self.server_url = server_url
        self.name = name
        self.description = description
        self.parameters = parameters

    def run(self, **kwargs) -> list[str]:
        """统一执行入口：每次调用新建一次 MCP 连接，执行本实例对应的远端工具。"""
        try:
            return asyncio.run(self._call_tool(self.name, kwargs))
        except Exception as e:
            # 连接失败（server 未启动 / 网络不通）等异常统一在这里兜底
            return [f"错误：无法连接 MCP Server {self.server_url} - {e}"]

    async def _call_tool(self, func_name: str, func_args: dict) -> list[str]:
        # async with：进入连接、退出断开（每次调用新建一次连接）
        async with Client(self.server_url) as client:
            result = await client.call_tool(func_name, func_args)

            # MCP 返回 content 块列表：提取纯文本内容，一条文本 = list 中的一个元素
            return [block.text for block in result.content if isinstance(block, TextContent)]


# --- 工具使用示例 ---
if __name__ == "__main__":
    # 批量生成与执行请走 tools/tool_manager.py 的 sync_mcp_tools()，
    # 或直接运行 python mcp_server/mcp_test.py（先启动 python mcp_server/mcp_server.py）。
    print("通用 MCP 适配器：请通过 tools/tool_manager.py 的 sync_mcp_tools() 批量使用。")