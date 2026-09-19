import os
import chromadb

# 数据库文件路径配置（相对本文件所在目录，避免受运行目录影响）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "vector_db")

class DbConnection():
    """数据库连接类：负责 Chroma 持久化连接与 collection 获取。"""

    def __init__(self):
        # Chroma 持久化客户端，数据实际存储于 database/vector_db
        self.client = chromadb.PersistentClient(path=DB_PATH)

    def get_or_create_collection(self, name: str):
        """
        按名获取 collection，不存在则创建。
        注意：不绑定 embedding function（显式传 None，避免 Chroma 启用默认 ONNX Embedding），
        由调用方先经 embedder 生成向量，再随 embeddings= / query_embeddings= 传入。
        """        
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

# 模块级单例：全项目共享同一个 DbConnection
db_connection = DbConnection()
