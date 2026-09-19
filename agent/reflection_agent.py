import time
import json
import logging

from llm.llm_client import OpenAICompatibleClient
from agent.core.agent import Agent
from agent.core.config import Config, config
from agent.core.message import Message, ToolCall
from tools.tool_manager import tool_manager
from agent.core.agent_response import AgentResponse

logger = logging.getLogger(__name__)


"""
Reflection（反思）范式的核心思想：先执行，后审阅，再修订。
执行者(Executor)借助原生函数调用(Function Calling)自主决定是否使用外部工具（如联网搜索）并生成草稿 -> 审阅者(Reviewer)对草稿提出批评意见 -> 执行者根据意见修订草稿。
如此循环往复，直到审阅者认可或达到最大反思轮数，实现"自我纠错、逐步进化"。
"""

# 1、执行器 (Executor) 提示词：生成/修订答案草稿
# 提示词中不提及任何工具：工具列表与参数由API的tools参数注入，是工具信息的唯一来源。
# 这样无论是否传入tools，提示词与API状态都保持一致，模型不会在文本中"模拟"调用工具
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家，是一个有能力调用外部工具的智能助手。

请按照以下规则解决用户的问题：
1. 如果解决用户的问题确有必要调用工具，请直接发起调用；如果不需要或已有信息足够，请直接作答。
2. 在收到工具的执行结果后，请基于执行结果直接输出最终的完整答案，注意严格限制在200字以内。

# 用户任务:
{task}

# 历次工具调用的观察结果（如果为"无"表示尚未调用过工具，请勿凭空捏造资料）:
# 注意：如果之前调用工具的结果中包含你想要的信息，请直接使用，请勿重复调用工具。
{tool_observations}

# 上一轮提交的答案草稿（如果为"无"表示这是首轮执行，尚无草稿）:
{previous_draft}

# 审阅者的反馈意见（如果反馈不是"无"，请务必针对每一条意见，结合上述观察结果与上一轮草稿修订你的答案）:
{feedback}
"""

class Executor:
    """执行者：借助原生函数调用自主决定是否使用工具，生成/修订答案草稿"""

    def __init__(self, llm_client: OpenAICompatibleClient, model: str, temperature: float, extra_body: dict | None = None):
        self.llm_client = llm_client
        self.model = model
        self.llm_temperature = temperature
        self.extra_body = extra_body  # 由 Agent 侧统一算好（仅 DeepSeek 时用于关闭默认开启的思考模式）
        self.tool_observations: list[str] = []  # 历次工具调用的观察结果（跨反思轮次累计），修订草稿时无需重复搜索
        self.previous_draft = ""  # 上一轮提交的答案草稿，让修订轮能看到自己写过的内容
        self.execute_tool_calls: list[str] = []  # 最近一次execute()调用的工具名称，供Agent层记录审计轨迹

    def reset(self):
        """每个新任务开始时重置执行器的全部状态（历史在反思轮次间保留、跨任务隔离），
        保证对同一个ReflectionAgent实例连续调用run方法解决不同问题时互不干扰。
        """
        self.tool_observations = []
        # 不保存全部历史草稿，只保留最新版，因为保留旧稿有两个负面影响：
        # （1）旧稿带着已知缺陷：它们是被审阅者批评过的版本，和新feedback一起展示会给模型冲突信号，甚至诱导它把已改掉的错误又改回来。
        # （2）feedback只对应最新稿：审阅者说"第二点论据不足"，这个"第二点"只在最新稿里有意义，旧稿在这个语境下是纯噪音。
        # 一个直观类比：人类改论文时，参考资料全部留着（对应tool_observations列表），
        # 但桌上永远只摊开最新一版稿子（对应previous_draft字符串）——没人会同时摊开三个历史版本对着改。
        self.previous_draft = ""
        self.execute_tool_calls = []

    def execute(self, task: str, feedback: str = "无") -> str:
        """执行者生成或修订答案草稿：首轮为撰写草稿，收到反馈后为修订草稿"""
        messages: list[Message] = []  # 存储本次execute()方法内与大模型的交互历史
        tools_schema = tool_manager.get_available_tools_openai_schema()  # 获取可用工具的OpenAI Schema
        self.execute_tool_calls = []  # 重置本轮的工具调用记录

        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            task=task, feedback=feedback,
            # 将历史信息注入提示词：观察结果与上一轮草稿为空时显示"无"，与提示词中的说明保持一致
            tool_observations="\n---\n".join(self.tool_observations) or "无",
            previous_draft=self.previous_draft or "无",
        )
        messages.append(Message(role="user", content=prompt))

        logger.info("--- 正在执行（撰写/修订草稿） ---")

        llm_response = self.llm_client.invoke_with_tools(
            model=self.model, messages=messages, tools=tools_schema, tools_choice="auto",
            temperature=self.llm_temperature, extra_body=self.extra_body
        )

        logger.info(f"⏱️ 执行者调用大模型耗时: {llm_response.latency_s}s, Token: {llm_response.usage}")

        # 模型自主决定：不调用工具时，直接返回文本答案（草稿同样保存，供下一轮修订参考）
        if not llm_response.tool_calls:
            if llm_response.content:
                self.previous_draft = llm_response.content
            return llm_response.content or ""

        # 模型自主决定调用工具：执行工具，并把观察结果以tool消息回传。
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
            # 函数调用参数是JSON字符串/字典，直接传给工具
            tool_args = tool_call.arguments

            logger.info(f"🎬 行动(id={tool_call.id}): {tool_name}[{tool_args}]")
            print(f"🔧 正在调用工具: {tool_name}...")
            observation = tool_manager.execute_tool(tool_name, tool_args)
            observation_text = str(observation)
            logger.debug(f"👀 观察: {observation_text[:200]}..." if len(observation_text) > 200 else f"👀 观察: {observation_text}")  # 工具调用返回结果只展示前200字

            self.execute_tool_calls.append(tool_name)  # 记录本轮调用的工具名称，供Agent层审计
            self.tool_observations.append(observation_text)  # 观察结果跨轮累计，供后续修订轮注入提示词
            messages.append(Message(role="tool", content=observation_text, tool_call_id=tool_call.id))

        # 第二次调用不再允许发起新的工具调用，并追加用户消息明确收尾，模型只能基于工具结果直接输出最终答案
        # （首条user消息提及了工具，需在结尾显式告知工具阶段结束，防止模型在文本中"模拟"调用工具）
        messages.append(Message(role="user", content="工具调用阶段已结束。请不要再尝试调用任何工具，直接基于以上所有信息输出最终的完整答案。"))
        final_response = self.llm_client.invoke_with_tools(
            model=self.model, messages=messages, tools=tools_schema, tools_choice="none",
            temperature=self.llm_temperature, extra_body=self.extra_body
        )

        logger.info(f"⏱️ 执行者收尾调用大模型耗时: {final_response.latency_s}s, Token: {final_response.usage}")

        # 保存本轮草稿，供下一轮修订参考
        if final_response.content:
            self.previous_draft = final_response.content
        return final_response.content or ""

# 2、审阅者 (Reviewer) 提示词：对草稿进行审阅，输出PASS或具体修改意见
# 输出采用结构化输出（Structured Output）方案：通过API的response_format开启JSON模式，
REVIEWER_PROMPT_TEMPLATE = """
你是一位以严苛著称的AI审阅专家。你的任务是从准确性、完整性、逻辑性和实用性等角度，对答案草稿进行审阅。

# 用户任务:
{task}

# 待审阅的答案草稿:
{draft}

请严格按照以下JSON格式输出你的审阅结果:
- 审阅通过、无需修改时: {{"result": "PASS"}}
- 需要修改时: {{"result": "FEEDBACK", "feedback": "具体的、可操作的修改意见（100字以内）"}}
"""

class Reviewer:
    """审阅者：对答案草稿进行审阅，输出PASS（通过）或FEEDBACK（具体修改意见）"""

    def __init__(self, llm_client: OpenAICompatibleClient, model: str, temperature: float, extra_body: dict | None = None):
        self.llm_client = llm_client
        self.model = model
        self.llm_temperature = temperature
        self.extra_body = extra_body  # 由 Agent 侧统一算好（仅 DeepSeek 时用于关闭默认开启的思考模式）

    def review(self, task: str, draft: str) -> str:
        """审阅答案草稿，返回"PASS"（通过）或具体修改意见（字符串）"""
        prompt = REVIEWER_PROMPT_TEMPLATE.format(task=task, draft=draft)
        messages = [Message(role="user", content=prompt)]

        logger.info("--- 正在审阅答案草稿 ---")
        # JSON模式：API层面强制模型输出合法JSON，PASS/FEEDBACK的判定不再依赖正则从自由文本中提取
        llm_response = self.llm_client.invoke(
            model=self.model, messages=messages, temperature=self.llm_temperature,
            response_format={"type": "json_object"}, extra_body=self.extra_body
        )

        logger.info(f"⏱️ 审阅者调用大模型耗时: {llm_response.latency_s}s, Token: {llm_response.usage}")

        response_text = llm_response.content or ""
        try:
            data = json.loads(response_text)
            # 审阅通过，反思循环结束
            if isinstance(data, dict) and data.get("result") == "PASS":
                return "PASS"
            # 需要修改：提取feedback字段；字段缺失或非字符串时兜底返回原文，保证反馈永远可用
            feedback = data.get("feedback") if isinstance(data, dict) else None
            if isinstance(feedback, str) and feedback.strip():
                return feedback.strip()
            return response_text
        except ValueError as e:  # json.JSONDecodeError是ValueError的子类
            logger.error(f"❌ 解析审阅结果时出错: {e}")
            logger.error(f"原始响应: {response_text}")
            return response_text.strip()


# 3、智能体 (Agent) 整合
class ReflectionAgent(Agent):
    """Reflection范式：执行者先生成草稿，审阅者提出批评意见，执行者据此修订，循环直到审阅通过或达到最大轮数。"""

    def __init__(self, name: str, config: Config):
        system_prompt = "你是一个具备自我反思能力的AI助手。你会先生成答案草稿，再由审阅者提出批评意见，然后据此修订草稿，如此循环直到审阅通过。"
        super().__init__(name, OpenAICompatibleClient(config.api_key, config.base_url), tool_manager, config, system_prompt)
        self.max_iterations = config.reflection_max_iterations  # 反思循环最大轮数，重要的安全阀，防止无限反思耗尽资源
        self.executor = Executor(
            self.llm_client, config.reflection_executor_model, config.reflection_executor_llm_temperature,
            extra_body=self._thinking_extra_body(
                config.reflection_executor_model, config.base_url, config.reflection_executor_thinking
            ),
        )
        self.reviewer = Reviewer(
            self.llm_client, config.reflection_reviewer_model, config.reflection_reviewer_llm_temperature,
            extra_body=self._thinking_extra_body(
                config.reflection_reviewer_model, config.base_url, config.reflection_reviewer_thinking
            ),
        )

    def run(self, task: str) -> AgentResponse:
        start_time = time.time()
        tool_call_names: list[str] = []  # 存放调用的工具名称

        self.executor.reset()  # 每次 run 开始时重置执行器状态（观察与草稿只在本次任务的反思轮次间保留）

        logger.info(f"用户任务: {task}")
        print("🤖 正在处理你的问题，请稍候...")

        # llm_client 内部对 API 失败是"打印 + raise"的透明包装，异常会向上抛出；
        # 执行者/审阅者的 LLM 调用统一在此兜底，避免网络/鉴权/模型不可用等失败中断调用方
        try:
            # 阶段1：执行——执行者自主决定是否调用工具，并生成初始草稿
            print("✍️ 正在生成初始草稿...")
            draft = self.executor.execute(task)
            if not draft:
                logger.error("执行者未能生成有效的草稿。")
                return AgentResponse(
                    agent_name=self.agent_name,
                    task=task,
                    content="",
                    latency_s=round(time.time()-start_time, 2),
                    status_code=400,
                    status_desc="错误：执行者未能生成有效的草稿。",
                    tool_call_names=tool_call_names
                )
            tool_call_names.extend(self.executor.execute_tool_calls)  # 记录初始执行阶段调用的工具
            logger.info(f"✅ 初始草稿已生成: {draft}")

            # 阶段2：反思循环（审阅 -> 修订 -> 再审阅 ...）
            passed = False
            for i in range(1, self.max_iterations + 1):
                print(f"🔁 正在进行第 {i}/{self.max_iterations} 轮反思...")
                logger.info(f"--- 第 {i}/{self.max_iterations} 轮反思 ---")
                feedback = self.reviewer.review(task, draft)
                logger.info(f"🧐 审阅意见: {feedback}")

                # 审阅通过，反思循环结束
                if feedback == "PASS":
                    logger.info("🎉 审阅通过，反思结束。")
                    passed = True
                    break

                # 审阅未通过，执行者带着反馈意见修订草稿（修订时同样可自主调用工具补充资料）
                revised = self.executor.execute(task, feedback=feedback)
                tool_call_names.extend(self.executor.execute_tool_calls)  # 记录修订阶段调用的工具
                # 修订返回空内容时保留上一版草稿，避免把已有的有效内容清空
                if revised:
                    draft = revised
                else:
                    logger.warning(f"⚠️ 第 {i} 轮修订未返回内容，保留上一版草稿。")
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

        if not passed:
            logger.warning(f"⚠️ 已达到最大反思轮数({self.max_iterations})，返回最后一版草稿。")

        logger.info(f"💡 Agent 最终回答: {draft}")

        return AgentResponse(
            agent_name=self.agent_name,
            task=task,
            content=draft,
            latency_s=round(time.time()-start_time, 2),
            status_code=200,
            status_desc="成功" if passed else "达到最大反思轮数，返回最后一版草稿",
            tool_call_names=tool_call_names
        )


# 4、主函数入口
if __name__ == '__main__':
    # 入口处只配一次：过程日志仅写文件 log_file.txt（不进屏幕）；排错时把 level 改成 DEBUG 可看工具返回等细节。
    # 此配置需在运行(创建 ReflectionAgent、调用工具)之前执行；格式与 main.py 入口保持一致。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("log_file.txt", encoding="utf-8"),  # encoding 保证中文不乱码
            # 不挂 StreamHandler → 屏幕干净
        ]
    )

    reflection_agent = ReflectionAgent(name="ReflectionAgent", config=config)

    task = "今天是2026年8月26日，我在天津。这周末想去看一个电影，推荐一个电影、去哪里看，什么时候去最好，再搭配一个餐厅选择，谢谢"

    answer = reflection_agent.run(task)

    print(f"\n [{reflection_agent.agent_name}] 给出的答案: {answer.content}","\n\n")

    print(answer)