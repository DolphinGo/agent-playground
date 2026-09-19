import time
import logging

from llm.llm_client import OpenAICompatibleClient
from agent.core.agent import Agent
from agent.core.config import Config, config
from agent.core.message import Message, ToolCall
from tools.tool_manager import tool_manager
from agent.core.agent_response import AgentResponse

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考并分析用户给出的任务，最终给出准确的答案。
如果你觉得任务需要调用外部工具，你可以调用合适的工具来获取信息。
可用工具如下：
{tools}

请按照以下规则解决用户的问题：
1. 如果你需要多个彼此独立的不同类型信息，可以在一轮中并行调用多个工具；
2. 当你认为信息已经足够时，请不要再调用任何工具，直接用中文输出完整的最终答案。

注意：
1、最终答案限制在200字以内。
2、但千万注意在同一轮中不要用相似参数重复调用同一个工具！！！"""


class ReActAgent(Agent):
    """ReAct（推理+行动）范式：思考 -> 行动（调用工具）-> 观察，循环往复，直到信息足够再给出最终答案。"""

    def __init__(self, name: str, config: Config):
        system_prompt = SYSTEM_PROMPT.format(tools=tool_manager.get_available_tools())
        super().__init__(name, OpenAICompatibleClient(config.api_key, config.base_url), tool_manager, config, system_prompt)
        self.model = config.react_agent_model
        self.max_steps = config.react_agent_max_steps  # agentloop 最大轮次
        self.llm_temperature = config.react_agent_llm_temperature
        self.thinking = config.react_agent_thinking  # 思考模式开关：enabled / disabled（仅 DeepSeek 生效）

    def run(self, task: str) -> AgentResponse:
        start_time = time.time()
        current_step = 0
        tool_call_names: list[str] = []  # 存放调用的工具名称
        messages: list[Message] = []  # 存储执行run()方法时与大模型的交互历史
        tools_schema = self.tool_manager.get_available_tools_openai_schema()  # 获取可用工具的OpenAI Schema
        extra_body = self._thinking_extra_body(self.model, self.config.base_url, self.thinking)  # 仅 DeepSeek 时附加请求体参数（关闭其默认开启的思考模式）

        # 标准Function Calling对话模式：
        # 1. system消息设定角色和可用工具描述（只发一次）
        # 2. user消息包含用户任务（只发一次）
        # 3. 后续轮次只追加assistant消息（含tool_calls和tool执行结果）
        # 模型通过messages自然能看到完整的对话历史，无需重复发送指令
        messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=task))

        logger.info(f"用户任务: {task}")
        print("🤖 正在处理你的问题，请稍候...")

        while current_step < self.max_steps:
            current_step += 1
            logger.info(f"--- 第 {current_step}/{self.max_steps} 步 ---")

            # 工具调用次数达到硬上限时，不再传入可调用的工具，强制模型只能输出文本答案
            if current_step == self.max_steps:
                tools_choice = "none"
                messages.append(Message(role="system", content="工具调用次数已达到限制，请根据已有信息给出最终答案。"))
            else:
                tools_choice = "auto"

            # 逐条打印本轮请求体，仅写日志文件（DEBUG 级别），排错时改日志级别即可查看
            logger.debug("📤 本轮发送给模型的消息:")
            for m in messages:
                if m.tool_calls:
                    logger.debug(f"  [{m.role}] {m.content or ''}")
                    for tc in m.tool_calls:
                        logger.debug(f"      tool_call[{tc.id}] {tc.name} {tc.arguments}")
                elif m.tool_call_id:
                    logger.debug(f"  [{m.role}] (tool_call_id={m.tool_call_id}) {m.content}")
                else:
                    logger.debug(f"  [{m.role}] {m.content}")

            # llm_client 内部对 API 失败是"打印 + raise"的透明包装，异常会向上抛出；
            # 这里必须兜底，否则网络/鉴权/模型不可用等失败会直接中断 run()，并导致调用方崩溃
            try:
                llm_response = self.llm_client.invoke_with_tools(
                    model=self.model, messages=messages, tools=tools_schema,
                    tools_choice=tools_choice, temperature=self.llm_temperature, extra_body=extra_body
                )
            except Exception as e:
                return AgentResponse(
                    agent_name=self.agent_name,
                    task=task,
                    content="错误：LLM调用失败，请稍后重试。",
                    latency_s=round(time.time()-start_time, 2),
                    status_code=500,
                    status_desc=f"错误：LLM调用失败 - {e}",
                    tool_call_names=tool_call_names
                )

            logger.info(f"⏱️ 第{current_step}步调用大模型耗时: {llm_response.latency_s}s, Token: {llm_response.usage}")

            # 本轮没有发起函数调用，说明模型认为信息已足够，直接给出最终答案
            if not llm_response.tool_calls:
                final_answer = llm_response.content or ""
                logger.info(f"💡 Agent 最终回答: {final_answer}")

                return AgentResponse(
                    agent_name=self.agent_name,
                    task=task,
                    content=final_answer,
                    latency_s=round(time.time()-start_time, 2),
                    status_code=200,
                    status_desc="成功",
                    tool_call_names=tool_call_names
                )

            # 模型决定调用工具：把思考过程写日志，便于事后复盘模型的决策依据
            if llm_response.content:
                logger.info(f"🤔 大模型思考过程: {llm_response.content}")
            elif llm_response.reasoning_content:
                logger.info(f"🤔 大模型思考过程: {llm_response.reasoning_content}")
            logger.info(f"🤔 模型决定调用 {len(llm_response.tool_calls)} 个工具")

            # 模型请求调用工具：把该消息加入上下文，执行工具，并把观察结果以tool消息回传。
            # reasoning_content 必须回传：DeepSeek 思考模式下，带工具调用的 assistant 轮次
            # 在后续所有请求中都要带上思维链字段，否则 API 会返回 400（详见官方 thinking_mode 文档）
            messages.append(Message(
                role="assistant",
                content=llm_response.content,  # 模型这轮的思考文本（只调工具不说话时是None）
                reasoning_content=llm_response.reasoning_content,
                tool_calls=[
                    ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                    for tc in llm_response.tool_calls
                ],
            ))

            for tool_call in llm_response.tool_calls:
                tool_name = tool_call.name
                tool_call_names.append(tool_name)
                # 函数调用参数是JSON字符串/字典，直接传给工具
                tool_args = tool_call.arguments

                logger.info(f"🎬 行动(id={tool_call.id}): {tool_name}[{tool_args}]")
                print(f"🔧 正在调用工具: {tool_name}...")
                observation = tool_manager.execute_tool(tool_name, tool_args)
                observation_text = str(observation)
                logger.debug(f"👀 观察: {observation_text[:200]}..." if len(observation_text) > 200 else f"👀 观察: {observation_text}")  # 工具调用返回结果只展示前200字
                messages.append(Message(role="tool", content=observation_text, tool_call_id=tool_call.id))

        logger.warning("已达到最大步数，流程终止。")
        return AgentResponse(
            agent_name=self.agent_name,
            task=task,
            content="错误：已达到最大循环数。",
            latency_s=round(time.time()-start_time, 2),
            status_code=400,
            status_desc="错误：已达到最大循环数。",
            tool_call_names=tool_call_names
        )


if __name__ == '__main__':
    # 入口处只配一次：过程日志仅写文件 log_file.txt（不进屏幕）；排错时把 level 改成 DEBUG 可看工具返回等细节。
    # 此配置需在运行(创建 ReActAgent、调用工具)之前执行；格式与 main.py 入口保持一致。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("log_file.txt", encoding="utf-8"),  # encoding 保证中文不乱码
            # 不挂 StreamHandler → 屏幕干净
        ]
    )

    react_agent = ReActAgent(name="ReactAgent", config=config)

    task = "我是一名经常出差的打工人，每月有长途旅行。请搜索一下2026年上半年适合长途出行用的高性价比手机（要求长续航、信号好、拍照够用、预算3000元左右），结合搜索结果对比2-3款，最后只给出最推荐的一款和理由（200字以内）。"

    answer = react_agent.run(task)

    print(f"\n [{react_agent.agent_name}] 给出的答案: {answer.content}","\n\n")

    print(answer)