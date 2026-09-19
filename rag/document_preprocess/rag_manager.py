"""RAG 知识库后台管理工具：

扫描 ./rag/docs 下的原始文档（PDF / txt / md 等），解析 → 切分 → 向量化入库；
后台维护：新增文档重新入库、按文档删除、清空知识库、查看已入库文档清单；
提供 query 快速检索，便于单独测试检索效果。

调用示例（在 Chapter08 目录下执行）：
    python -m rag.document_preprocess.rag_manager            # 交互式菜单
    python -m rag.document_preprocess.rag_manager --index    # 一键把 docs 下所有文档建库
    python -m rag.document_preprocess.rag_manager --query "NBA 常规赛规则"
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from rag.document_preprocess.document_parser import DocumentParser
from rag.document_preprocess.text_splitter import split_text
from rag.rag_store import rag_store  # 复用全局单例
from rag.rag_query_item import RagQueryItem

load_dotenv(find_dotenv())

# 知识库原始文档目录（在 rag/docs；本文件位于 rag/document_preprocess/，需向上两级：parent.parent 为 rag/）
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
# 切分参数：取 .env 中 Rag 配置
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

# 复用 rag_store 单例，与 RagQueryTool 连的是同一个 Chroma 集合
parser = DocumentParser()


def index_all() -> list[str]:
    """把 DOCS_DIR 下的所有原始文档逐一解析切分并入库（重复文档会先删旧再重建）。

    注意：这里仅遍历 docs 根目录下的一级文件（iterdir + is_file），
    不会递归处理 docs 子文件夹中的文档；若需支持子目录，应改用 rglob("*.pdf") 等方式。
    """
    files = sorted([
        p for p in DOCS_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    ])
    if not files:
        print("📂 docs 目录下没有文档。")
        return []
    return [index_file(p.name) for p in files]


def index_file(filename: str) -> str:
    """解析并入库 docs 目录下的单个文档（按文件名定位）。同名文档会"先删后建"，保证最新。"""
    path = DOCS_DIR / filename
    if not path.exists():
        return f"错误：{filename} 不存在于 {DOCS_DIR}"
    try:
        doc = parser.parse(path)
        if not doc.text or not doc.text.strip():
            return f"错误：{filename} 解析后无有效文本"
        chunks = split_text(doc.text, CHUNK_SIZE, CHUNK_OVERLAP)
        rag_store.update_document(chunks, source=filename)
        return f"✅ 已入库 {filename}（{len(chunks)} 个 chunk）"
    except Exception as e:
        return f"错误：入库 {filename} 失败 - {e}"


def delete_document(filename: str) -> str:
    """按文件名从知识库删除该文档。"""
    count = rag_store.delete_document(filename)
    return f"🗑️  已删除 {filename}（{count} 个 chunk）" if count else f"⚠️  {filename} 不在知识库中"


def clear_all() -> str:
    """清空整个知识库。"""
    count = rag_store.delete_all()
    return f"🗑️  已清空知识库（{count} 条）"


def list_documents() -> str:
    """列出当前知识库已入库的文档及其 chunk 数。"""
    counts = rag_store.list_sources()
    if not counts:
        return "当前知识库为空。"
    lines = [f"- {src}: {n} 个 chunk" for src, n in counts.items()]
    return "📚 已入库文档:\n" + "\n".join(lines)


def query(query: str, top_n: int | None = None) -> list[RagQueryItem]:
    """检索知识库并返回最相关片段（纯查询，不打印额外新媒体）。"""
    return rag_store.query(query, top_n=top_n)


def _run_cli() -> None:
    """交互式菜单（或命令行参数快捷操作）。"""
    args = sys.argv[1:]
    if "--index" in args:
        for msg in index_all():
            print(msg)
        return
    if "--clear" in args:
        print(clear_all())
        return
    if "--list" in args:
        print(list_documents())
        return
    if "--query" in args:
        q = args[args.index("--query") + 1]
        for r in query(q):
            print(f"[{r.score:.3f}] {r.source}: {r.text}\n")
        return

    print("📚 RAG 知识库管理后台")
    while True:
        print("\n请选择操作:")
        print("  1. 一键建库（解析 docs 下全部文档）")
        print("  2. 按文件名入库")
        print("  3. 按文件名删除")
        print("  4. 清空知识库")
        print("  5. 查看已入库文档清单")
        print("  6. 检索知识库")
        print("  0. 退出")
        choice = input("> ").strip()
        if choice == "1":
            for msg in index_all():
                print(msg)
        elif choice == "2":
            name = input("文档文件名（位于 docs/ 下）: ").strip()
            print(index_file(name))
        elif choice == "3":
            name = input("文档文件名: ").strip()
            print(delete_document(name))
        elif choice == "4":
            print(clear_all())
        elif choice == "5":
            print(list_documents())
        elif choice == "6":
            q = input("检索问题: ").strip()
            results = query(q)
            if not results:
                print("未找到相关内容。")
            print(f"共找到 {len(results)} 条相关文档：")
            for r in results:
                print(f"[{r.score:.3f}] {r.source}: {r.text}\n")
        elif choice == "0":
            print("再见！")
            break
        else:
            print("无效选择，请重试。")


if __name__ == "__main__":
    _run_cli()
