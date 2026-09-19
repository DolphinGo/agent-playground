from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentResponse:
    """Agent响应"""

    agent_name: str
    task: str
    content: str
    latency_s: float
    status_code: int
    status_desc: str
    tool_call_names: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")) # 默认生成当前时间戳

    def __repr__(self) -> str:
        """友好展示Agent的执行结果"""
        # 长文本截断，避免刷屏
        task_desc = str(self.task)[:100] + "......" if len(str(self.task)) > 100 else str(self.task)
        content_desc = str(self.content)[:100] + "......" if len(str(self.content)) > 100 else str(self.content)
        tool_calls_desc = ", ".join(self.tool_call_names) or ""
        return "\n".join([
            f"AgentResponse(",
            f"  agent        = {self.agent_name}",
            f"  状态          = {self.status_code} [{self.status_desc}]",
            f"  任务          = {task_desc}",
            f"  回答          = {content_desc}",
            f"  工具调用次数   = {len(self.tool_call_names)} [{tool_calls_desc}]",
            f"  耗时          = {self.latency_s:.2f}s",
            f"  时间          = {self.timestamp}",
            f")",
        ])
