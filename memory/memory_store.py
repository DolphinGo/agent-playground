import os
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from database.vector_store import VectorStore
from rag.document_preprocess.text_splitter import split_text

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)


class MemoryStore(VectorStore):
    """
    跨会话长期记忆的领域层：封装对 Chroma 向量库的读写。

    注意它不继承 Tool——记忆是一个"服务/子系统"，而不是"LLM 可见的单个函数"；
    真正暴露给外界的入口是 MemoryAddTool / MemoryQueryTool 两个薄工具，
    它们共享本类的实例（模块级单例）与同一套 Chroma / embedder 基础设施。

    继承自 VectorStore 基类（抽象出 Memory 与 RAG 共有的向量库/embedding 原语）。
    """

    collection_name = "agent_memory"
    # 记忆检索一次返回的条数
    memory_query_results = int(os.getenv("MEMORY_QUERY_RESULTS", "3"))
    # 同类型记忆判定为"同一主题"的相似度阈值
    memory_reuse_threshold = float(os.getenv("MEMORY_REUSE_THRESHOLD", "0.8"))
    # 记忆内容切分上限：每块不超过 chunk_size 字符、带 overlap，防止超长 embedding 被静默截断
    memory_chunk_size = int(os.getenv("MEMORY_CHUNK_SIZE", "250"))
    memory_chunk_overlap = int(os.getenv("MEMORY_CHUNK_OVERLAP", "50"))

    # source_type -> 生命周期策略（策略路由表：骨架核心，后续可扩展新类型）
    # overwrite：同类型最相似记忆覆盖更新（永远最新）；append：追加快照（保留历史多版本）
    MEMORY_POLICIES = {
        "user_preference": "overwrite",
        "user_fact": "overwrite",
        "tool_result": "append",
        "conversation": "append",
    }

    def __init__(self):
        super().__init__()  # 由基类创建 collection（collection_name = "agent_memory"）

    def add(self, content: str, source_type: str | None = None) -> list[str]:
        """
        存入一条记忆。source_type 决定生命周期策略（见 MEMORY_POLICIES）：
        - overwrite（user_preference / user_fact）：同类型最相似记忆相似度 >= 阈值时覆盖更新，永远最新；
        - append（tool_result / conversation）：直接追加快照，可多条并存。

        入库前先用 MEMORY_CHUNK_SIZE / MEMORY_CHUNK_OVERLAP 做切分，保证每条入库文本不超过安全长度。
        """
        if not content or not content.strip():
            return ["错误：记忆内容为空"]

        # 说明：由 agent._remember 校验后传入的 source_type 必为 MEMORY_POLICIES 中的合法值，故无需再做"置空串"兜底，库中不会存在未分类（source_type=""）的记忆。
        try:
            # 超长记忆切分：每条 chunk 大写约 chunk_size 字符，避免 embedding 截断
            chunks = split_text(content, self.memory_chunk_size, self.memory_chunk_overlap)
            results: list[str] = []
            for chunk in chunks:
                embedding = self._embed_query(chunk)

                # overwrite 策略：先在同类型记忆里找最相似的一条，命中则覆盖
                if source_type in self.MEMORY_POLICIES and self.MEMORY_POLICIES[source_type] == "overwrite":
                    existing_id, similarity = self._find_most_similar(embedding, source_type)
                    if existing_id and similarity >= self.memory_reuse_threshold:
                        self._update(
                            [existing_id], [chunk], [embedding],
                            [{"source_type": source_type, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}],
                        )
                        logger.info(f"🧠 更新长期记忆: {chunk[:30]}...（同类型覆盖，相似度 {similarity:.2f}）")
                        results.append(f"已更新同类型记忆（相似度 {similarity:.2f}）。")
                        continue

                # append 策略（或未命中覆盖）→ 新增一条快照
                self._add(
                    [uuid.uuid4().hex], [chunk], [embedding],
                    [{"source_type": source_type, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}],
                )
                logger.info(f"🧠 已存入长期记忆: {chunk[:30]}...")
                results.append("记忆已成功存入。")

            return results if results else ["记忆已成功存入。"]
        except Exception as e:
            return [f"错误：存入记忆失败 - {e}"]

    def _find_most_similar(self, embedding: list[float], source_type: str) -> tuple[str | None, float]:
        """在指定 source_type 内查询与给定向量最相似的一条记忆，返回 (id, 余弦相似度)。"""
        if self.collection.count() == 0:
            return None, 0.0
        result = self._query_embeddings(embedding, n_results=1, where={"source_type": source_type})
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if not ids:
            return None, 0.0
        # hnsw 的 cosine 空间下 distance = 1 - 余弦相似度(取值[-1,1])，转换为真正的余弦相似度
        return ids[0], 1.0 - distances[0]

    def query(self, query: str, source_type: str | None = None) -> list[str]:
        """按语义相似度检索最相关的记忆（返回多条，条数由 MEMORY_QUERY_RESULTS 控制），可选按 source_type 过滤。

        框架侧 agent._remember 已丢弃 source_type="" 的条目（见 agent/core/memory_item.py 的 MemoryItem 强校验），
        且 memory_add 工具仅由 _remember 调用，因此库中不存在未分类记忆，这里无需特判 ""。
        """
        try:
            if source_type is None:
                where = None  # 不指定类型：全量检索
            else:
                where = {"source_type": source_type}
            result = self._query_raw(
                query,
                n_results=min(self.memory_query_results, self.collection.count() or 1),
                where=where,
            )
            documents = (result.get("documents") or [[]])[0]
            if not documents:
                return ["没有找到相关记忆。"]
            logger.info(f"🧠 检索长期记忆: {query} -> 命中 {len(documents)} 条")
            return list(documents)
        except Exception as e:
            return [f"错误：检索记忆失败 - {e}"]


# 模块级单例：MemoryAddTool / MemoryQueryTool（以及后续 RAG）共享同一个 MemoryStore
memory_store = MemoryStore()