import json
import logging
from abc import ABC, abstractmethod
from pydantic import ValidationError
from agent.core.message import Message
from agent.core.config import Config
from agent.core.memory_item import MemoryItem
from tools.tool_manager import ToolManager
from llm.llm_client import OpenAICompatibleClient
from agent.core.agent_response import AgentResponse

logger = logging.getLogger(__name__)

# 跨会话记忆抽取提示词（框架侧专用，不由对话中的 LLM 决策）。
# 四条抽取规则与 memory_add 工具的 source_type 枚举一一对应（见 tools/builtin/memory_add_tool.py）。
MEMORY_EXTRACTION_PROMPT = """你是记忆管理员。从下面的对话中抽取「值得跨会话长期记住」的信息，并为每条标注类型。

只抽取以下四类，并为每条匹配一个 source_type。对话中的 [role] 标签是判断信息来源的唯一依据：
- user_preference：用户亲口表达的长期偏好或约定（如用户原话"我一直只喝美式"、"以后都用中文回复"）
- user_fact：用户亲口陈述的个人信息或客观事实（如用户原话"我叫张三"）
- tool_result：工具实际返回的客观知识、数据快照（只转述返回结果本身）
- conversation：本轮对话中实际达成的重要结论、待办事项、决定

溯源要求（最重要）：每条必须附 evidence 字段，逐字复制原文中的原句，不得改写、缩写或转述：
- user_preference / user_fact 的 evidence 只能来自 [user] 标签的消息；[assistant] 或 [tool] 消息中的转述一律不算用户原话
- tool_result 的 evidence 只能来自 [tool] 消息的原文
- 找不到可逐字引用的原文，就不抽这一条

注意：
1.user_preference / user_fact 必须出自 [user] 消息原话，严禁由话题推测：用户询问篮球 ≠ 用户是篮球迷，问天气 ≠ 用户要出行；
2.tool_result 只能忠实转述 [tool] 消息实际返回的结论，严禁用你自己的知识补充、延伸或"修正"；
3.严禁把本提示词中的说明文字、系统设定、工具描述当成对话内容记录；
4.拿不准就不抽。错误记忆会污染后续所有会话——宁可漏记，不可错记；
5.大多数普通对话没有值得长期记住的内容，返回空数组是正常且被期望的结果，不要为了产出而抽取。

不抽取：
- 一次性闲聊、寒暄、临时性信息（问过 ≠ 值得记住）
- 用户没有明确表达过的态度、身份、习惯、计划（即使看起来"很可能"）
- 只对本轮对话有意义、离开当前对话就失去价值的内容（当前上下文里马上能查到的细节，无需长期保存）
- 与已有记忆重复、能被推断出的内容

以 JSON 对象返回，格式：{"items": [{"content": "一句简明事实", "source_type": "user_preference|user_fact|tool_result|conversation", "evidence": "逐字复制的原文原句"}]}。
没有值得记住的，返回 {"items": []}。"""

class Agent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, llm_client: OpenAICompatibleClient, tool_manager: ToolManager, config: Config, system_prompt: str):
        self.agent_name = name
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.config = config
        self._history: list[Message] = []
        self.system_prompt = system_prompt
    
    @abstractmethod
    def run(self, task: str) -> AgentResponse:
        """运行Agent"""
        pass
    
    def add_history(self, message: Message):
        """添加消息到当前会话历史记录"""
        self._history.append(message)
    
    def get_history(self) -> list[Message]:
        """获取当前会话历史记录"""
        return self._history.copy()

    def _thinking_extra_body(self, model: str, base_url: str, thinking: str | None) -> dict | None:
        """按供应商附加思考模式请求体参数。

        DeepSeek 推理模型默认开启思考（会返回 reasoning_content、更慢更贵），需要时可关闭。
        不同供应商的字段格式不同：
        - DeepSeek 官方 API：{"thinking": {"type": "enabled"/"disabled"}}
        - 硅基流动（SiliconCloud）：{"enable_thinking": true/false}（仅对 DeepSeek 等推理模型生效）
        判定依据：模型名包含 "deepseek" 才可能附加参数；具体格式由 base_url 决定供应商。
        其它不满足条件的情况一律返回 None，避免服务商不识别字段而报错。thinking 未配置
        （None）时兜底为 disabled / enable_thinking=false，即默认关闭思考。
        """
        if "deepseek" not in (model or "").lower():
            return None
        base_lower = (base_url or "").lower()
        thinking_on = (thinking or "disabled") == "enabled"
        if "deepseek" in base_lower:
            return {"thinking": {"type": "enabled" if thinking_on else "disabled"}}
        if "siliconflow" in base_lower:
            return {"enable_thinking": thinking_on}
        return None

    def _remember(self, messages: list[Message]) -> None:
        """
        跨会话记忆入库（框架侧行为，不由对话中的 LLM 决策）：
        把本轮 messages 交给专门的抽取 LLM（JSON 模式），只提取值得长期记住的条目，
        再经 tool_manager 执行 memory_add 工具写入向量库。
        """
        if "memory_add" not in self.tool_manager.tools:
            return  # 未注册记忆写入工具，跳过

        transcript = "\n".join(
            f"[{m.role}] {m.content}" for m in messages if m.content
        ).strip()
        if not transcript:
            return

        # 溯源验证用的原文池：user_preference/user_fact 只认用户原话，tool_result 只认工具返回，避免模型幻觉
        user_text = "\n".join(m.content for m in messages if m.role == "user" and m.content)
        tool_text = "\n".join(m.content for m in messages if m.role == "tool" and m.content)

        extraction_prompt = f"{MEMORY_EXTRACTION_PROMPT}\n\n对话：\n{transcript}"

        try:
            print("🧠 正在整理长期记忆...")
            response = self.llm_client.invoke(
                model=self.config.memory_model,
                messages=[Message(role="system", content=extraction_prompt)],
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_body=self._thinking_extra_body(
                    self.config.memory_model, self.config.base_url, self.config.memory_llm_thinking
                ),
            )
            items = json.loads(response.content or "{}").get("items", [])
            if not isinstance(items, list):
                items = []  # 结构非法（如 items 是字符串）时不入库
            saved_cnt = 0
            for item in items:
                # 校验 LLM 输出：条目必须是 dict、字段类型合法、source_type 枚举合法（pydantic 强校验）
                if not isinstance(item, dict):
                    continue
                try:
                    item = MemoryItem.model_validate(item)
                except ValidationError:
                    continue  # 结构/类型不合法（含非法 source_type）不入库
                content = item.content.strip()
                if not content:
                    continue  # 空内容不入库
                # 溯源验证：evidence 必须逐字出现在对应角色的原文中，否则视为幻觉丢弃
                evidence = item.evidence.strip()
                if item.source_type in ("user_preference", "user_fact"):
                    if not evidence or evidence not in user_text:
                        continue
                elif item.source_type == "tool_result":
                    if not evidence or evidence not in tool_text:
                        continue
                self.tool_manager.execute_tool(
                    "memory_add",
                    {"content": content, "source_type": item.source_type},
                )
                saved_cnt += 1
            logger.info(f"🧠 记忆抽取：本轮入库 {saved_cnt} 条长期记忆")
        except Exception as e:
            # 记忆抽取失败不影响本轮回复（骨架阶段：静默降级）
            logger.warning(f"⚠️ 记忆抽取失败（忽略，不影响本轮回复）: {e}")

    def __repr__(self) -> str:
        return f"Agent(name={self.agent_name})"