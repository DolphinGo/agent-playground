from abc import ABC, abstractmethod


class Tool(ABC):
    """
    所有工具的抽象基类。

    一个 Tool 实例对应 LLM 可见的一个函数（与 OpenAI Function Calling 一一对应）。
    子类需要声明三个类属性，并实现 run() 方法：

    - name:        工具名（LLM 调用时使用的函数名，需全局唯一）
    - description: 工具描述（告诉 LLM 什么时候该用这个工具）
    - parameters:  参数定义（JSON Schema 格式）
    - run():       统一执行入口，返回 list[str]（ToolManager 只认这一个方法）
    """

    name: str
    description: str
    parameters: dict
    # 是否暴露给 LLM：False 表示仅供框架内部调用（如 memory_add），
    # 不会出现在 get_available_tools* 的返回结果中
    expose_to_llm: bool = True

    @abstractmethod
    def run(self, **kwargs) -> list[str]:
        """
        工具的统一执行入口。
        ToolManager 会把解析后的参数（dict）以关键字参数形式传入。
        约定：返回 list[str]，列表中的每个元素是一条独立结果（如一条记忆、一条搜索结果）；
        错误信息也以单元素列表返回（如 ["错误：..."]）
        """
        ...

    def to_openai_schema(self) -> dict:
        """
        生成该工具的 OpenAI Function Calling Schema。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
