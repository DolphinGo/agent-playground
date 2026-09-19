"""VectorStore：向量数据库领域层基类。
把 MemoryStore 与 RagStore 的共同点抽象到此处，具体子系统只继承并实现各自的语义：
- 持有对应的 Chroma collection（经全局 DbConnection 单例创建/获取）
- 各自持有一个 Embedder 实例
- 封装底层增删改查原语（_add / _update / _query_embeddings / count）
"""

from database.db_connection import db_connection
from database.embedder import Embedder

class VectorStore:
    """向量库领域层基类：暴露 Chroma + embedding 的通用底层原语。"""

    # 子类必须指定使用的 collection 名
    collection_name: str = ""

    def __init__(self):
        if not self.collection_name:
            raise ValueError(f"{type(self).__name__} 必须指定 collection_name")
        self.collection = db_connection.get_or_create_collection(self.collection_name)
        self.embedder = Embedder()

    # ---------- embedding ----------
    def _embed_query(self, text: str) -> list[float]:
        """单条文本语义向量（用于查询）。"""
        return self.embedder.embed_query(text)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本语义向量（用于入库）。"""
        return self.embedder.embed_documents(texts)

    # ---------- 写入原语 ----------
    # metadata 中的 dict value只能是str、int、float、bool，不支持list、dict等复杂类型
    def _add(self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """写入若干条记录（upsert：id 不存在则插入、存在则覆盖，天然幂等）。
        空 documents 直接跳过（Chroma 硬约束）。
        用 upsert 而非 add：add 遇到已存在 id 时各版本行为不一（旧版静默插入重复记录、
        新版可能报错），rag_store 的稳定 id "source#idx" 重复入库时依赖覆盖语义。
        """
        if not documents:
            return
        # 旧版这里用的 self.collection.add
        self.collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas, # type: ignore[arg-type]  # Chroma 存根的不变性缺陷：拒绝 list[list[float]]，运行时合法
        )

    def _update(self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """按 id 更新记录。"""
        self.collection.update(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas, # type: ignore[arg-type]  # 同上：Chroma 存根不变性缺陷
        )

    def _delete(self, ids: list[str] | None = None, where: dict | None = None) -> int:
        """按 ids 或 where 过滤条件删除记录，返回删除条数（ids 与 where 至少提供其一）。

        Chroma 的 collection.delete 不返回删除数量，为统计准确条数，先按 where 查出
        命中的 id 再删除。ids 优先级高于 where：只给 where 时先查该批 id。
        """
        if not ids and where is not None:
            data = self.collection.get(where=where, include=[])
            ids = data.get("ids", []) or []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    # ---------- 查询原语 ----------
    def _query_embeddings(self, query_embedding: list[float], n_results: int, where: dict | None = None) -> dict:
        """
        用一条查询向量去 collection 里做相似度召回，返回 Chroma 原始结果 dict
        （含 ids / distances / documents / metadatas）。

        n_results：期望返回条数；传 0 或负数会被替换为1（Chroma 不允许 0）。
        """
        if n_results <= 0:
            n_results = 1
        return self.collection.query(
            # type: ignore[arg-type]  # 同上：Chroma 存根不变性缺陷
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

    def _query_raw(self, query_text: str, n_results: int, where: dict | None = None) -> dict:
        """便捷：直接用查询文本做召回并返回 Chroma 原始结果 dict。"""
        embedding = self._embed_query(query_text)
        return self._query_embeddings(embedding, n_results, where)

    # ---------- 其它 ----------
    def count(self) -> int:
        """当前 collection 记录总数。"""
        return self.collection.count()