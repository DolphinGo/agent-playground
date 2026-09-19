"""
HelloAgents 交互式入口：统一调度项目中的四种经典Agent范式
运行方式（在根目录下）: python main.py
流程：选择Agent -> 调用大模型回答 -> 展示结果 -> 回到菜单继续下一轮。
其中 ChatAgent 是多轮对话范式：选定后进入对话循环，可连续提问，输入 exit / quit / 退出 才返回主菜单；
另外三个是单次任务范式，执行完一条任务后即返回菜单。
"""
import logging

# 入口处只配一次：过程日志仅写文件 log_file.txt（不进屏幕）；排错时把 level 改成 DEBUG 可看工具返回等细节。
# 必须放在其它项目 import 之前——tool_manager 等模块在 import 时就会触发日志（如 MCP 同步失败），
# 若 basicConfig 在其后运行，那些日志会走 logging 的 lastResort 兜底打印到屏幕，破坏"屏幕仅保留问答"。
# 格式与 chat_agent.py 的 __main__ 入口保持一致。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("log_file.txt", encoding="utf-8"),  # encoding 保证中文不乱码
        # 不挂 StreamHandler → 屏幕干净
    ]
)

from agent.core.config import config
from agent.core.agent_response import AgentResponse
from agent.chat_agent import ChatAgent
from agent.react_agent import ReActAgent
from agent.plan_and_solve_agent import PlanAndSolveAgent
from agent.reflection_agent import ReflectionAgent

# 1、Agent菜单：编号 -> (展示名称, 一句话简介, 类)
AGENT_MENU = [
    ("1", "ReAct Agent         ", "🤖 推理+行动范式：边思考边调用工具获取信息"),
    ("2", "Plan-and-Solve Agent", "🧩 先规划后执行范式：分解任务为步骤再逐步执行"),
    ("3", "Reflection Agent    ", "🔁 自我反思范式：生成草稿由审阅者批评并修订迭代"),
    ("4", "Chat Agent          ", "💬 多轮聊天+记忆助手：结合长期记忆与工具连续对话"),
]

# 2、构建四个Agent实例（config 为全局单例）
agents = {
    "1": ReActAgent(name="ReactAgent", config=config),
    "2": PlanAndSolveAgent(name="PlanAndSolveAgent", config=config),
    "3": ReflectionAgent(name="ReflectionAgent", config=config),
    "4": ChatAgent(name="ChatAgent", config=config),
}

# ChatAgent 是唯一的多轮对话范式（内部维护会话历史），其余三个是单次任务范式：
# 前者选定后进入对话循环，后者执行一条任务后立即回到菜单，故交互流程分开处理
CHAT_AGENT_KEY = "4"
EXIT_WORDS = ("exit", "quit", "退出")  # 退出对话循环的关键词


def show_menu():
    """打印选择菜单"""
    print("\n" + "=" * 60)
    print("👋 欢迎使用 HelloAgents 交互式控制台！")
    print("=" * 60)
    for num, name, desc in AGENT_MENU:
        print(f"  [{num}] {name} {desc}")
    print("  [0] 退出程序")
    print("-" * 60)


def show_answer(answer: AgentResponse) -> None:
    """统一展示AgentResponse：按状态码区分成功/失败，再打印执行摘要"""
    print("\n" + "=" * 60)
    # Agent 内部已对 LLM 失败兜底为 status_code != 200 的响应，
    # 这里据此区分「成功」与「失败」，避免把错误提示当成最终答案展示
    if answer.status_code == 200:
        print(f"\n 🤖 Agent 给出的答案: {answer.content}")
    else:
        print(f"❌ {answer.status_desc}")
    print("=" * 60)


def run_single_task(choice: str) -> None:
    """单次任务范式（ReAct / Plan-and-Solve / Reflection）：执行一条任务后即返回菜单"""
    task = input("📝 请输入你的任务描述: ").strip()
    if not task:
        print("⚠️ 任务不能为空，请重新输入！")
        return

    selected_name = next(name for num, name, _ in AGENT_MENU if num == choice).strip()
    print(f"\n🚀 已选定 {selected_name}，开始处理任务...")
    # 出错不退出程序，回到菜单可继续
    try:
        show_answer(agents[choice].run(task))
    except Exception as e:
        print(f"\n❌ 任务执行失败: {e}，请重试或更换任务。")


def chat_loop() -> None:
    """ChatAgent 的多轮对话循环：连续提问，直到用户输入 exit / quit / 退出 才回到菜单"""
    agent = agents[CHAT_AGENT_KEY]
    print(f"\n🚀 已进入 {agent.agent_name} 多轮对话（输入 exit / quit / 退出 返回主菜单）...\n")

    while True:
        task = input("你: ").strip()
        if not task:
            continue
        if task.lower() in EXIT_WORDS:
            print("\n👋 已结束对话，返回主菜单。")
            break

        # 单轮失败不结束对话，用户可继续提问或退出
        try:
            show_answer(agent.run(task))
        except Exception as e:
            print(f"\n❌ 对话失败: {e}，请重试或更换问题。")


if __name__ == '__main__':
    try:
        while True:
            show_menu()

            # 让用户选择Agent
            choice = input("🎯 请选择要使用的Agent (输入编号): ").strip()
            if choice == "0":
                print("\n� 再见！期待下次再见～")
                break
            if choice not in agents:
                print("⚠️ 无效的选项，请重新输入！")
                continue

            # 多轮对话范式 => 进入对话循环；单次任务范式 => 执行一条任务后回到菜单
            if choice == CHAT_AGENT_KEY:
                chat_loop()
            else:
                run_single_task(choice)
    except KeyboardInterrupt:
        # input() 时按 Ctrl+C 抛 KeyboardInterrupt，优雅退出，避免打印 traceback
        print("\n👋 再见！期待下次再见～")
