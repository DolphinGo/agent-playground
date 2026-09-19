"""RagStore：RAG 知识库的领域层（操作向量数据库）。

继承 VectorStore 基类（复用 Chroma / embedding 原语），实现 RAG 特有的业务：
- 入库：把切分后的 chunk 批量 embedding 写入 rag collection
- 查询：召回升级为两阶段 ——「向量召回 top_k」→「rerank 精选 top_n」。
  召回用 bge embedding 语义相似度召回大量候选；重排用 bge-reranker 相关性精排，
  更贴合"与用户意图真正相关"的片段。

所需配置来自 .env 的 "# Rag 功能配置" 段（RAG_TOP_K / RAG_TOP_N 等）。
"""

import os
import logging
from dotenv import load_dotenv, find_dotenv
from database.vector_store import VectorStore
from rag.reranker import Reranker
from rag.rag_query_item import RagQueryItem

load_dotenv(find_dotenv())  # 保证未掉 .env（不依赖导入顺序）

logger = logging.getLogger(__name__)


class RagStore(VectorStore):
    """RAG 知识库领域层：chunk 向量化入库 + 召回重排查询。"""

    collection_name = "rag_documents"
    # 召回阶段取回的候选条数
    top_k = int(os.getenv("RAG_TOP_K", "10"))
    # 重排序后真正返回的条数
    top_n = int(os.getenv("RAG_TOP_N", "3"))

    def __init__(self):
        super().__init__()  # 基类已持有 collection 与 Embedder，这里只需补 Reranker
        # Rerank 模型：由 RagStore 持有，供两阶段检索的重排阶段使用
        self.reranker = Reranker()

    def add_documents(self, chunks: list[str], source: str, chunk_indices: list[int] | None = None) -> int:
        """
        把某文档切分出的 chunks 一次性写入知识库。

        args:
            chunks:         文档切分后的文本块列表
            source:         文档标识（通常用文件名），写入 metadata 用于删改定位
            chunk_indices:  每个 chunk 在原文中的序号（可选，默认 0..n-1）
        returns:
            实际写入条数
        """
        if not chunks:
            return 0
        embeddings = self._embed_documents(chunks)

        indices = chunk_indices or list(range(len(chunks)))
        ids = [f"{source}#{idx}" for idx in indices]  # 稳定 id：source+序号，便于更新
        metadatas = [{"source": source, "chunk_index": idx} for idx in indices]

        self._add(ids, list(chunks), embeddings, metadatas)

        logger.info(f"📄 RAG 入库 {len(chunks)} 个 chunk (source={source})")
        return len(chunks)

    def update_document(self, chunks: list[str], source: str) -> int:
        """重新入库某文档：先删该 source 的全部旧 chunk，再写入新 chunks（先删后建，保证最新）。"""
        self.delete_document(source)
        return self.add_documents(chunks, source=source)

    def delete_document(self, source: str) -> int:
        """按 source（文档标识）删除该文档全部 chunk，返回删除条数。"""
        return self._delete(where={"source": source})

    def delete_all(self) -> int:
        """清空整个知识库，返回删除条数。"""
        if self.collection.count() == 0:
            return 0
        data = self.collection.get(include=[])
        ids = data.get("ids", []) or []
        return self._delete(ids=ids)

    def list_sources(self) -> dict[str, int]:
        """统计当前知识库有哪些文档（source）及其 chunk 数量。"""
        data = self.collection.get(include=["metadatas"]) if self.collection.count() else {} # 只看 metadatas
        metas = data.get("metadatas", []) or []
        counts: dict[str, int] = {}
        for m in metas:
            src = str((m or {}).get("source", "unknown"))
            counts[src] = counts.get(src, 0) + 1
        return counts

    # 查询：召回 + 重排序
    def query(self, query: str, top_k: int | None = None, top_n: int | None = None) -> list[RagQueryItem]:
        """
        RAG 经典两阶段检索：向量召回 top_k 候选 → rerank 精选 top_n。

        args:
            query:  用户查询
            top_k:  召回候选数（默认 RAG_TOP_K）
            top_n:  重排后返回条数（默认 RAG_TOP_N）
        returns:
            list[RagQueryItem]：按相关度降序
        """
        top_k = top_k or self.top_k
        top_n = top_n or self.top_n

        if self.collection.count() == 0:
            logger.info("⏳ RAG 知识库为空，未检索到内容。")
            return []

        # 阶段一：向量召回（语义相似度取前 top_k 候选）
        result = self._query_raw(query, n_results=min(top_k, self.collection.count()))
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        if not docs:
            logger.info(f"🔎 RAG 召回阶段未命中任何 chunk (query={query})")
            return []

        logger.info(f"🔎 RAG 召回 {len(docs)} 条 (top_k={top_k})，进入重排序...")

        # 阶段二：rerank 精排取 top_n
        reranked = self.reranker.rerank(query, docs, top_n=top_n)
        results = []
        for item in reranked:
            idx = item.index
            meta = metas[idx] if idx < len(metas) else {}
            results.append(RagQueryItem(
                text=item.text or docs[idx],
                score=item.score,
                source=meta.get("source", ""),
                chunk_index=meta.get("chunk_index", idx),
                recall_distance=distances[idx] if idx < len(distances) else None,
                index=idx,
            ))
        logger.info(f"🎯 RAG 重排序后返回 top_n={len(results)} 条")
        return results


# 模块级单例：RagQueryTool / RagManager 共享同一个 RagStore
rag_store = RagStore()