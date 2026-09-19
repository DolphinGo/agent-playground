import json
import time
import logging

from llm.llm_client import OpenAICompatibleClient
from agent.core.agent import Agent
from agent.core.config import Config, config
from agent.core.message import Message
from agent.core.agent_response import AgentResponse
from tools.tool_manager import tool_manager

logger = logging.getLogger(__name__)


# 1、规划器 (Planner) 提示词：将复杂问题分解成包含多个子步骤的行动计划
# 通过API的response_format开启JSON模式，强制模型只输出合法JSON，解析零歧义
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由最多{max_steps}个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
请务必确保在{max_steps}个步骤内完成任务。

任务: {task}

请严格按照以下JSON格式输出你的计划，"plan"字段的值是一个字符串数组:
{{"plan": ["步骤1", "步骤2", "步骤3"]}}
"""

class Planner:
    """规划器：负责将复杂问题分解成行动计划"""

    def __init__(self, llm_client: OpenAICompatibleClient, model: str, temperature: float, max_steps: int, extra_body: dict | None = None):
        self.llm_client = llm_client
        self.model = model
        self.llm_temperature = temperature
        self.max_steps = max_steps
        self.extra_body = extra_body  # 由 Agent 侧统一算好（仅 DeepSeek 时用于关闭默认开启的思考模式）

    def plan(self, task: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(task=task, max_steps=self.max_steps)
        messages = [Message(role="user", content=prompt)]

        logger.info("--- 正在生成计划 ---")
        # JSON模式：API层面强制模型输出合法JSON，不再依赖对输出格式的"口头约定"
        llm_response = self.llm_client.invoke(
            model=self.model, messages=messages, temperature=self.llm_temperature,
            response_format={"type": "json_object"}, extra_body=self.extra_body
        )

        logger.info(f"⏱️ 规划阶段调用大模型耗时: {llm_response.latency_s}s, Token: {llm_response.usage}")

        # 解析LLM输出的JSON，提取计划步骤
        plan = self._parse_plan(llm_response.content or "")
        if not plan:
            logger.error("❌ 解析计划失败，无法生成有效的行动计划。")
            return []
        # 计划步骤数兜底：超过上限时截断
        if len(plan) > self.max_steps:
            logger.warning(f"⚠️ 计划步骤数({len(plan)})超过上限，截断为前{self.max_steps}步。")
            plan = plan[:self.max_steps]

        logger.info(f"📋 计划已生成: {plan}")
        return plan

    def _parse_plan(self, response_text: str) -> list[str]:
        """
        LLM输出是JSON对象，例如：{"plan": ["步骤1", "步骤2", "步骤3"]}，
        使用json.loads直接解析，再做字段校验。
        """
        try:
            data = json.loads(response_text)
            # 兼容两种输出：{"plan": [...]} 或直接输出 [...]
            steps = data.get("plan", []) if isinstance(data, dict) else data
            # 只保留非空字符串类型的步骤，过滤掉非法元素
            if isinstance(steps, list):
                return [step.strip() for step in steps if isinstance(step, str) and step.strip()]
            return []
        except ValueError as e:  # json.JSONDecodeError是ValueError的子类
            logger.error(f"❌ 解析计划时出错: {e}")
            logger.error(f"原始响应: {response_text}")
            return []

# 2、执行器 (Executor) 提示词：严格按照计划逐步执行
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始任务、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始任务:
{task}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
注意输出文字限制在200个汉字以内
"""

# 答案合成提示词：所有步骤执行完毕后，综合回答原始问题（经典Plan-and-Solve范式的收尾环节）
SYNTHESIZER_PROMPT_TEMPLATE = """
你是一位顶级的AI总结专家。给定的行动计划已经全部执行完毕。
请基于所有步骤的执行结果，针对原始任务给出一份完整、连贯的最终答案。

# 原始任务:
{task}

# 历史步骤与结果:
{history}

请直接输出最终答案:
注意输出文字限制在500个汉字以内
"""

class Executor:
    """执行器：负责按计划逐步执行，并合成最终答案"""

    def __init__(self, llm_client: OpenAICompatibleClient, model: str, temperature: float, extra_body: dict | None = None):
        self.llm_client = llm_client
        self.model = model
        self.llm_temperature = temperature
        self.extra_body = extra_body  # 由 Agent 侧统一算好（仅 DeepSeek 时用于关闭默认开启的思考模式）

    def execute_step(self, task: str, plan: list[str], history: str, current_step: str, step_num: int) -> str:
        """执行计划中的单个步骤，返回该步骤的执行结果"""
        # 计划格式化成带序号的列表，提升可读性
        plan_text = "\n".join(f"{i}. {step}" for i, step in enumerate(plan, 1))
        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            task=task, plan=plan_text, history=history if history else "无", current_step=current_step
        )
        messages = [Message(role="user", content=prompt)]

        llm_response = self.llm_client.invoke(
            model=self.model, messages=messages, temperature=self.llm_temperature, extra_body=self.extra_body
        )

        logger.info(f"⏱️ 步骤{step_num}调用大模型耗时: {llm_response.latency_s}s, Token: {llm_response.usage}")

        return llm_response.content or ""

    def synthesize(self, task: str, history: str) -> str:
        """所有步骤执行完毕后，综合全部结果回答原始任务"""
        prompt = SYNTHESIZER_PROMPT_TEMPLATE.format(task=task, history=history)
        messages = [Message(role="user", content=prompt)]

        llm_response = self.llm_client.invoke(
            model=self.model, messages=messages, temperature=self.llm_temperature, extra_body=self.extra_body
        )

        logger.info(f"⏱️ 合成阶段调用大模型耗时: {llm_response.latency_s}s, Token: {llm_response.usage}")

        return llm_response.content or ""


# 4、智能体 (Agent) 整合
class PlanAndSolveAgent(Agent):
    """Plan-and-Solve范式：先规划（Plan）把复杂任务拆成步骤，再执行（Solve）逐步完成，最后综合成最终答案。"""

    def __init__(self, name: str, config: Config):
        # Plan-and-Solve范式不需要调用工具，但保持与Agent基类一致的构造签名
        system_prompt = "你是一个先规划、后执行的AI助手。你会先将复杂问题分解成行动计划，再逐步执行计划，最后综合所有结果给出最终答案。"
        super().__init__(name, OpenAICompatibleClient(config.api_key, config.base_url), tool_manager, config, system_prompt)
        self.planner = Planner(
            self.llm_client, config.planner_model, config.planner_llm_temperature, config.planner_max_steps,
            extra_body=self._thinking_extra_body(config.planner_model, config.base_url, config.planner_thinking),
        )
        self.executor = Executor(
            self.llm_client, config.executor_model, config.executor_llm_temperature,
            extra_body=self._thinking_extra_body(config.executor_model, config.base_url, config.executor_thinking),
        )

    def run(self, task: str) -> AgentResponse:
        start_time = time.time()

        logger.info(f"用户任务: {task}")
        print("🤖 正在处理你的问题，请稍候...")

        # llm_client 内部对 API 失败是"打印 + raise"的透明包装，异常会向上抛出；
        # 规划/执行/合成三处的 LLM 调用统一在此兜底，避免网络/鉴权/模型不可用等失败中断调用方
        try:
            # 阶段1：规划（Plan）—— 将复杂问题分解成行动计划
            print("🧩 正在生成行动计划...")
            plan = self.planner.plan(task)
            if not plan:
                logger.error("无法生成有效的行动计划。")
                return AgentResponse(
                    agent_name=self.agent_name,
                    task=task,
                    content="",
                    latency_s=round(time.time()-start_time, 2),
                    status_code=400,
                    status_desc="错误：无法生成有效的行动计划。",
                    tool_call_names=[]
                )

            # 阶段2：执行（Solve）—— 逐步执行计划，历史结果作为上下文传递给后续步骤
            history = ""
            for i, step in enumerate(plan, 1):
                print(f"⚙️ 正在执行步骤 {i}/{len(plan)}...")
                logger.info(f"-> 正在执行步骤 {i}/{len(plan)}: {step}")
                result = self.executor.execute_step(task, plan, history, step, step_num=i)

                # 步骤执行失败时熔断，避免错误结果污染后续步骤
                if not result:
                    logger.error(f"❌ 步骤 {i} 未返回有效结果，任务终止。")
                    return AgentResponse(
                        agent_name=self.agent_name,
                        task=task,
                        content="",
                        latency_s=round(time.time()-start_time, 2),
                        status_code=400,
                        status_desc=f"错误：步骤{i}未返回有效结果。",
                        tool_call_names=[]
                    )

                history += f"步骤 {i}: {step}\n结果: {result}\n\n"
                logger.info(f"✅ 步骤 {i} 已完成，结果: {result}")

            # 阶段3：合成—— 综合所有步骤结果，针对原始问题输出最终答案
            print("✍️ 正在生成最终回答...")
            final_answer = self.executor.synthesize(task, history)

            # 合成失败时与规划/步骤失败保持一致，避免返回 200 + 空内容
            if not final_answer:
                logger.error("❌ 合成阶段未返回有效结果。")
                return AgentResponse(
                    agent_name=self.agent_name,
                    task=task,
                    content="",
                    latency_s=round(time.time()-start_time, 2),
                    status_code=400,
                    status_desc="错误：合成阶段未返回有效结果。",
                    tool_call_names=[]
                )
        except Exception as e:
            return AgentResponse(
                agent_name=self.agent_name,
                task=task,
                content="错误：LLM调用失败，请稍后重试。",
                latency_s=round(time.time()-start_time, 2),
                status_code=500,
                status_desc=f"错误：LLM调用失败 - {e}",
                tool_call_names=[]
            )

        logger.info(f"💡 Agent 最终回答: {final_answer}")

        return AgentResponse(
            agent_name=self.agent_name,
            task=task,
            content=final_answer,
            latency_s=round(time.time()-start_time, 2),
            status_code=200,
            status_desc="成功",
            tool_call_names=[]
        )


# 5、主函数入口
if __name__ == '__main__':
    # 入口处只配一次：过程日志仅写文件 log_file.txt（不进屏幕）；排错时把 level 改成 DEBUG 可看工具返回等细节。
    # 此配置需在运行(创建 PlanAndSolveAgent、调用工具)之前执行；格式与 main.py 入口保持一致。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("log_file.txt", encoding="utf-8"),  # encoding 保证中文不乱码
            # 不挂 StreamHandler → 屏幕干净
        ]
    )

    ps_agent = PlanAndSolveAgent(name="PlanAndSolveAgent", config=config)

    task = "我准备在天津开一家超市，帮我制定一个计划，并整合成一份详细计划书"

    answer = ps_agent.run(task)

    print(f"\n [{ps_agent.agent_name}] 给出的答案: {answer.content}","\n\n")

    print(answer)