from typing import List
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

# 加载 .env 文件中的环境变量
_ = load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

class Embedder():

    def __init__(self):
        api_key = os.environ.get('EMBEDDING_API_KEY')
        base_url = os.environ.get('EMBEDDING_BASE_URL',"https://api.siliconflow.cn/v1")
        model = os.environ.get('EMBEDDING_MODEL',"BAAI/bge-large-zh-v1.5")
        
        if not api_key:
            logger.error("EMBEDDING_API_KEY未在环境变量中配置")
            raise ValueError("Embedding 模型API配置错误")

        self.model: str = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本转向量"""
        if not texts:
            return []
        result = []
        # 保守起见按 16 一批切分（社区常用 32 以内作为安全批量，官方未明文规定数组上限）
        # bge-large-zh-v1.5 单文本 token 上限为512，每个中文字符约等于1个token（国产LLM，一个token约等于1~1.5个汉字）
        # 即单文本最多容纳约 250~500 个中文字符，超长部分会被静默截断（不报错）
        for i in range(0, len(texts), 16):
            resp = self.client.embeddings.create(
                model=self.model,
                input=texts[i:i+16]
            )
            result.extend([item.embedding for item in resp.data])
        return result

    def embed_query(self, text: str) -> list[float]:
        """单文本转向量"""
        text = text.strip()
        if not text:
            logger.error("embed_query 的输入不能为空")
            raise ValueError("No query text provided")
        return self.embed_documents([text])[0]
