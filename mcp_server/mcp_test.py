"""MCP 三件套（mcp_server / mcp_client / mcp_tool）联通性测试脚本。

场景1【Server 未启动】: sync_mcp_tools() 抛异常(注册报错)；手动构造 MCPTool.run() 返回"无法连接"错误(调用报错)。
场景2【Server 已启动】: 正常注册工具、打印 OpenAI Schema，并真实调用 search 工具输出结果。
"""

import sys
from pathlib import Path

# 把项目根目录加入 sys.path，确保能导入 tools
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool_manager import tool_manager, DEFAULT_SERVER_URL
from tools.builtin.mcp_tool import MCPTool

def t1_register():
    """测试1: 通过 tool_manager.sync_mcp_tools() 同步注册(发现) server 上的全部工具。"""
    print("=" * 60)
    print("测试1｜同步注册工具 tool_manager.sync_mcp_tools()")
    print("=" * 60)
    try:
        tools = tool_manager.sync_mcp_tools()
    except Exception as e:
        print(f"✘ 注册失败，sync_mcp_tools() 抛出异常：{type(e).__name__}: {e}")
        print("  → 符合『Server 未启动时报错』的预期")
        return []
    print(f"✔ 注册成功，发现 {len(tools)} 个 MCP 工具：")
    for t in tools:
        print(f"  - [{t.name}] {t.description}（expose_to_llm={t.expose_to_llm}）")
        print(f"    OpenAI Schema: {t.to_openai_schema()}")
    return tools

def t2_call(tools):
    """测试2: 调用工具。有注册结果则真实调用；否则手动构造 MCPTool 验证失败兜底。"""
    print("=" * 60)
    print("测试2｜调用工具 MCPTool.run()")
    print("=" * 60)
    if tools:
        tool = tools[0]
    else:
        print("（Server 未启动，手动构造 MCPTool 以验证调用失败的兜底行为）")
        tool = MCPTool(
            server_url=DEFAULT_SERVER_URL,
            name="search",
            description="网页实时搜索工具",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
    print(f"调用工具 [{tool.name}]，参数 query='HelloAgent 是什么'")
    for line in tool.run(query="2026世界杯冠军是哪支球队？"):
        print(f"  ▸ {line}")

if __name__ == "__main__":
    t2_call(t1_register())