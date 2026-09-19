import json
import asyncio
import logging
from typing import Dict, Any
from mcp import Client
from tools.tool import Tool
from tools.builtin.mcp_tool import MCPTool
from tools.builtin.memory_add_tool import MemoryAddTool
from tools.builtin.memory_query_tool import MemoryQueryTool
from tools.builtin.rag_query_tool import RagQueryTool
from tools.builtin.rag_list_documents_tool import RagListDocumentsTool

logger = logging.getLogger(__name__)

# MCP server 默认地址（与 mcp_server/mcp_server.py 的监听端口一致）
DEFAULT_SERVER_URL = "http://127.0.0.1:8010/mcp"


class ToolManager:
    """
    工具管理器：负责注册、描述和执行所有工具。

    只依赖 Tool 基类的契约（name / description / parameters / run()），
    不关心任何具体工具的内部实现——新增工具无需修改本类。
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """
        向工具箱中注册一个工具实例（Tool 子类）。
        """
        if tool.name in self.tools:
            logger.warning(f"⚠️ 警告：工具 '{tool.name}' 已存在，将被覆盖。")

        self.tools[tool.name] = tool
        logger.debug(f"工具 '{tool.name}' 已注册。")

    async def _list_mcp_tools(self, server_url: str = DEFAULT_SERVER_URL) -> list[MCPTool]:
        """连接一次 MCP server，把 list_tools() 返回的每个函数包装成一个 MCPTool 实例。"""
        async with Client(server_url) as client:
            server_tools = await client.list_tools()
            return [
                MCPTool(
                    server_url=server_url,
                    name=tool.name,
                    description=tool.description or "",  # MCP 的 description 是 str | None，兜底为空串
                    parameters=tool.input_schema,
                )
                for tool in server_tools.tools
            ]

    def sync_mcp_tools(self, server_url: str = DEFAULT_SERVER_URL) -> list[MCPTool]:
        """同步桥接入口：内部 asyncio.run 跑异步的 _list_mcp_tools，供 agent 启动时一次性调用。

        返回：MCPTool 实例列表（每个对应 server 上暴露的一个工具）；
             若连接失败或 server 未启动，异常会向上抛出，由模块级注册逻辑兜底。
        """
        return asyncio.run(self._list_mcp_tools(server_url))

    def _get_tool(self, name: str) -> Tool | None:
        """
        根据名称获取一个工具实例。
        """
        if name not in self.tools:
            logger.error(f"错误：未找到名为 '{name}' 的工具。")
            return None

        return self.tools[name]

    def get_available_tools(self) -> str:
        """
        获取所有 LLM 可见工具的格式化描述字符串。
        内部工具（expose_to_llm=False，如 memory_add）不参与——LLM 不应看到它们。
        """
        return "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools.values()
            if tool.expose_to_llm
        ])

    def get_available_tools_openai_schema(self) -> list[dict[str, Any]]:
        """
        获取所有 LLM 可见工具的 OpenAI Function Calling Schema 列表。
        只返回 expose_to_llm=True 的工具：内部工具（如 memory_add）不暴露给 LLM，
        它们是框架侧（Agent._remember）通过 execute_tool 调用的。
        Schema 生成逻辑收敛在 Tool 基类的 to_openai_schema() 中。
        """
        return [tool.to_openai_schema() for tool in self.tools.values() if tool.expose_to_llm]

    def execute_tool(self, func_name: str, func_args: dict[str, Any] | str) -> str:
        """
        执行一个工具，返回其输出。
        工具统一返回 list[str]（每条结果一行），这里拼接成一条观察文本回传给 LLM。
        args:
            func_name: 工具的名称(openai返回值中的function.name)
            func_args: 工具的输入参数，dict或JSON字符串均可(openai返回值中的function.arguments是JSON字符串格式。实例：'{"query": "2026世界杯冠军是谁"}')
        """
        tool = self._get_tool(func_name)
        if not tool:
            return f"错误：未找到名为 '{func_name}' 的工具。"

        try:
            # API返回的arguments是JSON字符串，先解析成dict再解包
            if isinstance(func_args, str):
                func_args = json.loads(func_args)

            # 校验必须是JSON对象（dict）。模型可能返回'[1,2]'这类合法JSON但不是对象
            if not isinstance(func_args, dict):
                return f"错误：工具 '{func_name}' 的参数必须是JSON对象（如{{\"query\": \"...\"}}），实际为: {func_args}"

            # 唯一执行入口：所有工具都是 run(**kwargs)，返回 list[str]
            result = tool.run(**func_args)
            if isinstance(result, list):
                return "\n".join(result)
            return str(result)
        except json.JSONDecodeError:
            return f"错误：工具 '{func_name}' 的参数JSON语法错误，无法解析: {func_args}"
        except Exception as e:
            return f"错误：执行工具 '{func_name}' 时出现问题 - {e}"


# 创建一个全局单例实例，并注册所有内置工具
tool_manager = ToolManager()

# ---- MCP 工具注册（网络相关，失败时容错降级）----
try:
    # sync_mcp_tools() 会连一次 server 并 list_tools()，把发现的每个远程函数包装成 MCPTool 后批量注册
    for tool in tool_manager.sync_mcp_tools():
        tool_manager.register_tool(tool)
except Exception as e:
    # server 未启动 / 网络不通时打印提示，跳过 MCP 工具，本地工具仍可正常注册使用。
    # mcp 2.0 会把底层连接错误包在 ExceptionGroup/TaskGroup 里，直接打 {e} 只能看到
    # "unhandled errors in a TaskGroup"，定位不到真实原因；这里下探到最底层异常再打印
    root = e
    while True:
        # 异常组（兼容 ExceptionGroup / BaseExceptionGroup）的成员在 .exceptions 里
        if isinstance(root, BaseExceptionGroup) and root.exceptions:
            root = root.exceptions[0]
        elif root.__cause__ is not None:
            root = root.__cause__
        else:
            break
    logger.warning(f"⚠️ MCP 工具同步失败（请先运行 python mcp_server/mcp_server.py）：{type(root).__name__}: {root}")

# ---- 内置工具注册（无网络，安全）----
tool_manager.register_tool(MemoryAddTool())
tool_manager.register_tool(MemoryQueryTool())
tool_manager.register_tool(RagQueryTool())
tool_manager.register_tool(RagListDocumentsTool())


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1、打印可用的工具
    print("\n--- 可用的工具 ---")
    print(tool_manager.get_available_tools())

    # 2、打印可用的工具的Schema
    print("\n--- 可用的工具的Schema ---")
    print(tool_manager.get_available_tools_openai_schema())

    # 3、遍历执行所有已注册工具，覆盖每个工具的 run() 入口。
    #    统一模拟「模型返回的 function.name + function.arguments(JSON字符串)」接入方式。
    print("\n--- 遍历执行所有工具 ---")
    test_cases = {
        # 网页搜索工具
        "search": '{"query": "2026世界杯冠军是谁"}',
        # 内部记忆写入工具（不暴露给 LLM）
        "memory_add": '{"content": "用户喜欢喝美式咖啡，不加糖", "source_type": "user_preference"}',
        # 跨会话记忆检索工具
        "memory_query": '{"query": "用户喝咖啡的习惯"}',
        # 本地知识库（RAG）检索工具
        "rag_query": '{"query": "2024NBA杯冠军是谁"}',
        # 内部知识库目录工具（不暴露给 LLM）
        "rag_list_documents": '{}',
    }

    for func_name, func_arguments in test_cases.items():
        print(f"\n=== 执行 Action: {func_name}[{func_arguments}] ===")
        observation = tool_manager.execute_tool(func_name, func_arguments)
        print("\n--- 观察 (Observation) ---")
        print(observation)
