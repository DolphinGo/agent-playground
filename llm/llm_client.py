import time
import logging
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from typing import Dict, List, Any
from llm.llm_response import LLMResponse
from agent.core.message import Message, ToolCall

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """
    通用的OpenAI API兼容的LLM客户端
    """

    def __init__(self, api_key: str, base_url: str):
        """
        初始化客户端。参数：API密钥、服务地址。
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def invoke(self, model: str, messages: List[Message], temperature: float, response_format: Dict[str, str] | None = None, extra_body: dict | None = None) -> LLMResponse:
        """
        调用大语言模型进行生成，并返回其响应。
        参数：消息列表、模型名称、采样温度、响应格式（如JSON模式：{"type": "json_object"}）、额外请求体参数（如关闭思考模式）。
        """
        logger.info(f"🧠 正在调用 {model} 模型...")
        print(f"🧠 正在调用 {model} 模型，请稍候...")
        start_time = time.time()
        try:
            request_kwargs = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature
            }
            if response_format:
                request_kwargs["response_format"] = response_format # plan_and_solve_agent 配置中定义的响应格式
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = self.client.chat.completions.create(**request_kwargs)
            logger.info(f"✅ 大语言模型响应成功")

            latency_s = round(time.time()-start_time, 2) # 单位：秒
            message = response.choices[0].message

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return LLMResponse(
                model=model,
                finish_reason=getattr(response.choices[0], "finish_reason", None),
                content=message.content,
                reasoning_content=getattr(message, "reasoning_content", None), # reasoning_content是推理模型的扩展字段，可能不存在
                tool_calls=[],
                usage=usage,
                latency_s=latency_s
            )

        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise


    def invoke_with_tools(self, model: str, messages: List[Message], tools: list[Dict[str, Any]], tools_choice: str, temperature: float, extra_body: dict | None = None) -> LLMResponse:
        """
        调用大语言模型（支持工具调用）进行生成，并返回其响应。
        参数：消息列表、模型名称、工具列表、工具选择、采样温度、额外请求体参数（如关闭思考模式）。
        """
        logger.info(f"🧠 正在调用 {model} 模型...")
        print(f"🧠 正在调用 {model} 模型，请稍候...")

        start_time = time.time()
        try:
            request_kwargs = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "tools": tools,
                "tool_choice": tools_choice
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = self.client.chat.completions.create(**request_kwargs)
            logger.info(f"✅ 大语言模型响应成功")

            # print("\n","模型响应:", response,"\n")

            latency_s = round(time.time()-start_time, 2) # 单位：秒
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments
                    ))

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return LLMResponse(
                model=model,
                finish_reason=finish_reason,
                content=message.content,
                reasoning_content=getattr(message, "reasoning_content", "") or "", # 只有推理模型才会有这个字段
                tool_calls=tool_calls,
                usage=usage,
                latency_s=latency_s
            )

        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise

# --- 客户端使用示例 ---
if __name__ == '__main__':
    import os
    from tools.tool_manager import tool_manager
    _ = load_dotenv(find_dotenv())

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = "deepseek-ai/DeepSeek-V4-Flash"
    if not api_key or not base_url:
        raise ValueError("环境变量中缺少名为LLM_API_KEY或LLM_BASE_URL的变量")

    try:
        llmClient = OpenAICompatibleClient(api_key=api_key, base_url=base_url)

        exampleMessages = [
            Message(role="user", content="你好，我在天津，请我这里天气如何？")
        ]

        print("--- 调用LLM ---")
        response = llmClient.invoke_with_tools(
            model=model_name,
            messages=exampleMessages,
            tools=tool_manager.get_available_tools_openai_schema(),
            tools_choice="auto",
            temperature=0.3
        )

        print(response,'\n\n')
        print(response.to_dict())

    except ValueError as e:
        print(e)