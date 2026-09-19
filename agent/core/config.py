import os
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

class Config(BaseModel):
    # API配置
    api_key: str = ""
    base_url: str = ""

    # ReactAgent配置
    react_agent_model: str
    react_agent_max_steps: int = 5
    react_agent_llm_temperature: float = 0.3
    react_agent_thinking: str | None = None  # 思考模式开关：enabled / disabled（未配置时为 None，仅对 DeepSeek 生效且默认关闭思考）

    # PlanAndSolveAgent配置
    planner_model: str
    planner_llm_temperature: float = 0.3
    planner_thinking: str | None = None  # 规划器思考模式开关，语义同 chat_agent_thinking
    planner_max_steps: int = 3  # 计划最多包含的步骤数
    executor_model: str
    executor_llm_temperature: float = 0.3
    executor_thinking: str | None = None  # 执行器思考模式开关，语义同 chat_agent_thinking

    # ReflectionAgent配置
    reflection_executor_model: str
    reflection_executor_llm_temperature: float = 0.3
    reflection_executor_thinking: str | None = None  # 执行者思考模式开关，语义同 chat_agent_thinking
    reflection_reviewer_model: str
    reflection_reviewer_llm_temperature: float = 0.3
    reflection_reviewer_thinking: str | None = None  # 审阅者思考模式开关，语义同 chat_agent_thinking
    reflection_max_iterations: int = 3  # 反思循环最大轮数（安全阀，防止无限反思耗尽资源）

    # 记忆配置
    memory_model: str = "deepseek-flash"  # 记忆抽取用的模型（可单独用更便宜的模型，控制成本）
    memory_llm_thinking: str | None = None  # 记忆抽取 LLM 的思考模式开关：enabled / disabled（未配置时为 None，仅对 DeepSeek 生效）

    # ChatAgent 配置
    chat_agent_model: str
    chat_agent_llm_temperature: float = 0.3
    chat_agent_thinking: str | None = None  # 思考模式开关：enabled / disabled（未配置时为 None，仅对 DeepSeek 生效且默认关闭思考）
    memory_query_results: int = 3
    memory_reuse_threshold: float = 0.8
    max_history_tokens: int = 4000  # 最大历史记录token预算（按字符数近似）

    @classmethod
    def from_env(cls) -> "Config": # 因为 Python 在解析这个函数时，Config 类可能还没有完全定义完成。
        """从环境变量创建配置"""
        _ = load_dotenv(find_dotenv())  # 显式加载 .env，避免依赖其它模块 import 的副作用
        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", ""),
            react_agent_model=os.getenv("REACT_AGENT_MODEL", "deepseek-flash"),
            react_agent_max_steps=int(os.getenv("REACT_AGENT_MAX_STEPS", "5")),
            react_agent_llm_temperature=float(os.getenv("REACT_AGENT_LLM_TEMPERATURE", "0.3")),
            react_agent_thinking=os.getenv("REACT_AGENT_THINKING"),
            planner_model=os.getenv("PLANNER_MODEL", "deepseek-flash"),
            planner_llm_temperature=float(os.getenv("PLANNER_LLM_TEMPERATURE", "0.3")),
            planner_thinking=os.getenv("PLANNER_THINKING"),
            planner_max_steps=int(os.getenv("PLANNER_MAX_STEPS", "3")),
            executor_model=os.getenv("EXECUTOR_MODEL", "deepseek-flash"),
            executor_llm_temperature=float(os.getenv("EXECUTOR_LLM_TEMPERATURE", "0.3")),
            executor_thinking=os.getenv("EXECUTOR_THINKING"),
            reflection_executor_model=os.getenv("REFLECTION_EXECUTOR_MODEL", "deepseek-flash"),
            reflection_executor_llm_temperature=float(os.getenv("REFLECTION_EXECUTOR_LLM_TEMPERATURE", "0.3")),
            reflection_executor_thinking=os.getenv("REFLECTION_EXECUTOR_THINKING"),
            reflection_reviewer_model=os.getenv("REFLECTION_REVIEWER_MODEL", "deepseek-flash"),
            reflection_reviewer_llm_temperature=float(os.getenv("REFLECTION_REVIEWER_LLM_TEMPERATURE", "0.3")),
            reflection_reviewer_thinking=os.getenv("REFLECTION_REVIEWER_THINKING"),
            reflection_max_iterations=int(os.getenv("REFLECTION_MAX_ITERATIONS", "3")),
            memory_model=os.getenv("MEMORY_MODEL", "deepseek-flash"),
            memory_llm_thinking=os.getenv("MEMORY_LLM_THINKING"),
            chat_agent_model=os.getenv("CHAT_AGENT_MODEL", "deepseek-flash"),
            chat_agent_llm_temperature=float(os.getenv("CHAT_AGENT_LLM_TEMPERATURE", "0.3")),
            chat_agent_thinking=os.getenv("CHAT_AGENT_THINKING"),
            memory_query_results=int(os.getenv("MEMORY_QUERY_RESULTS", "3")),
            memory_reuse_threshold=float(os.getenv("MEMORY_REUSE_THRESHOLD", "0.8")),
            max_history_tokens=int(os.getenv("MAX_HISTORY_TOKENS", "4000"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


# 模块级单例：全局只读一份配置，任何地方 from agent.core.config import config 拿到的都是同一份
config: Config = Config.from_env()