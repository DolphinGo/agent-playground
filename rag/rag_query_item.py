"""RAG 检索结果的数据结构（内部数据对象，跨 reranker / rag_store / rag_query_tool / rag_manager 共用）。"""
from dataclasses import dataclass


@dataclass
class RagQueryItem:
    """RAG 两阶段检索（向量召回 → rerank 重排）共用的单条结果结构。

    属性：
        text:            命中的文档文本
        score:           相关性分数（重排后按此降序）
        source:          文档来源标识（文件名）
        chunk_index:     该 chunk 在原文中的序号（召回阶段填充）
        recall_distance: 向量召回的原始距离（未重排时为 None）
        index:           该条在召回候选列表中的原始下标（重排阶段使用）
    """
    text: str
    score: float = 0.0
    source: str = ""
    chunk_index: int = -1
    recall_distance: float | None = None
    index: int = -1