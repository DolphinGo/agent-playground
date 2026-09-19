import time
import logging

logger = logging.getLogger(__name__)

from llm.llm_client import OpenAICompatibleClient
from agent.core.agent import Agent 
from agent.core.config import Config, config
from agent.core.message import Message, ToolCall
from tools.tool_manager import tool_manager
from agent.core.agent_response import AgentResponse


SYSTEM_PROMPT = """你是一个乐于助人的中文AI聊天助手，可以与用户进行多轮、自然的对话。如果需要信息，你可以调用工具获取。

可用工具如下：
{tools}

请按照以下规则进行回答：

[工具调用规则]
当本会话上下文缺少回答所需的信息时，先调用合适的工具补全信息，按以下优先级判断：
  1. 涉及用户过往信息时用 memory_query，常见于以下四类：
    * 用户偏好：口味、习惯、喜爱的事物等；
    * 用户事实：姓名、职业、联系方式等个人基本信息；
    * 知识快照：过去工具检索到并存下来的资料；
    * 对话纪要：更早会话聊过什么、做过什么。
  2. 问题依赖本地知识库中的既定知识时用 rag_query（当前知识库主要收录：{rag_documents}）；
  3. 需要实时/时效信息时用 search；
  4. 可同时命中多个来源时，允许并行调用多个工具。

[何时不要调用工具]
  - 常识问题、当本会话上下文或工具返回的结果已足以作答时，请直接给出友好、完整的回答，不要调用工具；
  - 所需信息已在工具返回结果中时，不要重复调用同一工具；
  - 工具无结果时，基于已有上下文和自己的知识继续回答，不要反复调用。

[作答要求]
1. 只要调用了工具，必须结合工具返回结果与已有知识，给出完整、准确的最终回答，不要机械复述；
2. 当问题缺少关键信息、且工具也无法补全时，主动向用户提问澄清，而不是猜测作答。

[示例]
用户："我上次说想喝什么咖啡来着？" → 应调用 memory_query（用户偏好）
用户："2026世界杯冠军是谁？" → 应调用 search（实时信息）
用户："知识库中收录的文档内容" → 应调用 rag_query（本地知识库）
用户："你的训练库中包含的知识" → 无需调用工具，直接使用你自己已有的知识回答"""


class ChatAgent(Agent):
    def __init__(self, name: str, config: Config):
        # 动态读取知识库文档清单，注入到系统提示词中；
        # 空库或调用出错时降级为空库描述，避免注入失败文案干扰提示词
        rag_documents = tool_manager.execute_tool("rag_list_documents", {})
        if not rag_documents or "错误" in rag_documents:
            rag_documents = "（当前为空）"

        system_prompt = SYSTEM_PROMPT.format(
            tools=tool_manager.get_available_tools(),
            rag_documents=rag_documents,
        )
        super().__init__(name, OpenAICompatibleClient(config.api_key, config.base_url), tool_manager, config, system_prompt)
        self.model = config.chat_agent_model
        self.llm_temperature = config.chat_agent_llm_temperature

    def run(self, task: str) -> AgentResponse:
        start_time = time.time()
        tool_call_names: list[str] = []  # 存放调用的工具名称

        # 第一步：把本会话的交互历史 self._history 组装进传给 LLM 的 messages，
        # 让大模型能看到本次多轮对话的上下文（同一 ChatAgent 实例多次 run 累积的历史）
        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        messages.extend(self._history)
        messages.append(Message(role="user", content=task))
        # 仅本轮对话（不含 system 提示词与历史会话），用作跨会话记忆抽取的输入，
        # 避免把系统设定和所有过往 record 混入记忆抽取，干扰对"当前轮"的判断
        messages_for_memory: list[Message] = [Message(role="user", content=task)]
        tools_schema = self.tool_manager.get_available_tools_openai_schema()  # 获取可用工具的OpenAI Schema
        extra_body = self._thinking_extra_body(self.model, self.config.base_url, self.config.chat_agent_thinking)  # 仅 DeepSeek 时附加请求体参数（关闭其默认开启的思考模式）

        self.add_history(Message(role="user", content=task))  # 当前用户输入同步记入会话历史，下一轮可见

        logger.info(f"用户任务: {task}")
        print("🤖 正在处理你的问题，请稍候...")

        # ---- 第一轮：尝试调用工具（memory_query / search / rag_query），也可以直接作答 ----
        # llm_client 内部对 API 失败是"打印 + raise"的透明包装，异常会向上抛出；
        # 这里必须兜底，否则网络/鉴权/模型不可用等失败会直接中断 run()，并导致对话循环崩溃
        try:
            llm_response = self.llm_client.invoke_with_tools(
                model=self.model, messages=messages, tools=tools_schema,
                tools_choice="auto", temperature=self.llm_temperature, extra_body=extra_body
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

        # ---- 第二轮：若第一轮调用了工具，则执行工具并把观察结果回传，强制模型给出最终答案 ----
        if llm_response.tool_calls:
            # 把该消息（含tool_calls）加入上下文，为执行工具做准备。
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

                logger.info(f"🎬 调用工具: {tool_name}{tool_args}")
                print(f"🔧 正在调用工具: {tool_name}...")
                observation = tool_manager.execute_tool(tool_name, tool_args)
                observation_text = str(observation)
                logger.debug(f"👀 工具返回: {observation_text[:200]}..." if len(observation_text) > 200 else f"👀 工具返回: {observation_text}")  # 工具调用返回结果只展示前200字
                messages.append(Message(role="tool", content=observation_text, tool_call_id=tool_call.id))
                messages_for_memory.append(Message(role="tool", content=observation_text, tool_call_id=tool_call.id))

            print("✍️ 正在生成最终回答...")

            # 第二轮：结合第一轮工具结论 + 已有知识作答，禁止再调用工具
            # 注意：此处必须继续使用 invoke_with_tools（tools_choice="none"）而不是 invoke()。
            # 原因：此时的 messages 中已含有 role="tool" 的观察结果消息（上面的 append），
            # OpenAI 兼容 API 要求 tool 角色消息必须是对前一条带 tool_calls 的 assistant 消息的响应，
            # 若改用不带 tools 参数的 invoke()，请求体中缺失 tools 定义，多个后端会直接拒绝（400 校验失败）。
            # tools_choice="none" 的语义正是「保留 tools schema 以便历史中的 tool 消息合法化，但禁止模型发起新的工具调用」。
            try:
                llm_response = self.llm_client.invoke_with_tools(
                    model=self.model, messages=messages, tools=tools_schema,
                    tools_choice="none", temperature=self.llm_temperature, extra_body=extra_body
                )
            except Exception as e:
                # 第二轮的 API 失败同样兜底，避免中断 run()
                return AgentResponse(
                    agent_name=self.agent_name,
                    task=task,
                    content="错误：LLM调用失败，请稍后重试。",
                    latency_s=round(time.time()-start_time, 2),
                    status_code=500,
                    status_desc=f"错误：LLM调用失败 - {e}",
                    tool_call_names=tool_call_names
                )

            final_answer = llm_response.content
        else:
            # 第一轮没有调用工具，说明模型认为信息已足够，直接给出最终答案
            final_answer = llm_response.content

        final_answer = final_answer or ""
        logger.info(f"💡 Agent 最终回答: {final_answer}")  # 最终答复记入日志，便于事后排查
        self.add_history(Message(role="assistant", content=final_answer))  # 回答记入会话历史，下一轮可见
        self._trim_history()  # 保持历史记录数量在最大范围内

        # run 结束前：仅用本轮对话（用户输入 + 工具返回 + 最终答复）调用 _remember
        # 自动总结并存储长期记忆（跨会话），不再把 system 提示词和整个历史都塞进去
        messages_for_memory.append(Message(role="assistant", content=final_answer))
        self._remember(messages_for_memory)

        return AgentResponse(
            agent_name=self.agent_name,
            task=task,
            content=final_answer,
            latency_s=round(time.time()-start_time, 2),
            status_code=200,
            status_desc="成功",
            tool_call_names=tool_call_names
        )
    
    def _trim_history(self):
        """控制历史总 token 数：用字符数近似 token 数（1 汉字 ≈ 1 token），
        超预算时从最老的消息开始逐一丢弃，直到总长达标。"""
        budget = self.config.max_history_tokens
        while sum(len(msg.content or "") for msg in self._history) > budget and len(self._history) > 1:
            self._history.pop(0)


if __name__ == '__main__':
    # 入口处只配一次：过程日志仅写文件 log_file.txt（不进屏幕）；排错时把 level 改成 DEBUG 可看工具返回等细节。
    # 此配置需在运行(创建 ChatAgent、调用工具)之前执行；格式与 main.py 入口保持一致。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("log_file.txt", encoding="utf-8"),  # encoding 保证中文不乱码
            # 不挂 StreamHandler → 屏幕干净
        ]
    )

    chat_agent = ChatAgent(name="ChatAgent", config=config)

    print("🤖 聊天助手已就绪，输入内容与我对话（输入 exit / quit / 退出 结束对话）...\n")

    try:
        while True:
            task = input("你: ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit", "退出"):
                break

            answer = chat_agent.run(task)
            print(f"\n [{chat_agent.agent_name}] 给出的答案: {answer.content}", "\n\n")
    except KeyboardInterrupt:
        # 终端里 Ctrl+C 会抛 KeyboardInterrupt，这里统一走正常退出流程，
        # 避免打印一串异常堆栈打断用户的退出操作
        pass

    print(f"\n👋 再见，期待下次交流！")