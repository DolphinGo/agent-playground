import os
import logging
import requests
from dotenv import load_dotenv, find_dotenv
from rag.rag_query_item import RagQueryItem

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

class Reranker:

    def __init__(self):
        self.api_key = os.getenv("RAG_RERANK_API_KEY", "")
        if not self.api_key:
            logger.error("RAG_RERANK_API_KEY未在环境变量中配置")
            raise ValueError("Reranker 模型API配置错误")

        self.base_url = os.getenv("RAG_RERANK_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        self.model = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[RagQueryItem]:
        """
        对 query 与候选文档做相关性重排序，返回按分数降序的 top_n 条。

        args:
            query:       用户查询
            documents:   召回阶段的候选文档列表
            top_n:       返回的最相关条数
        returns:
            [RagQueryItem(index=原列表下标, score=相关性分数, text=文档文本), ...]
        """
        if not documents:
            return []
        if not self.api_key:
            # 未配置 RAG_RERANK_API_KEY 时退化为"按原始顺序取前 top_n"，保证流程不中断
            return [
                RagQueryItem(text=doc, score=0.0, index=i)
                for i, doc in enumerate(documents[:top_n])
            ]

        url = f"{self.base_url}/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            ordered = []
            for r in results:
                idx = r.get("index", 0)
                text = (r.get("document") or {}).get("text", "") if isinstance(r.get("document"), dict) else ""
                ordered.append(RagQueryItem(
                    text=text or (documents[idx] if idx < len(documents) else ""),
                    score=float(r.get("relevance_score", 0.0)),
                    # 接口未回传文本时，从原列表取值兜底
                    index=idx,
                ))
            # 已按 relevance_score 降序（接口 Promise top_n 有序），安全起见显式排序
            ordered.sort(key=lambda x: x.score, reverse=True)
            return ordered
        except Exception as e:  # noqa: BLE001
            # 重排序失败不阻断召回：退化为按原始顺序取前 top_n
            logger.warning(f"⚠️  Reranker 调用失败，退化为原始顺序: {e}")
            return [
                RagQueryItem(text=doc, score=0.0, index=i)
                for i, doc in enumerate(documents[:top_n])
            ]