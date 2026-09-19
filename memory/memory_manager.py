"""长期记忆（agent_memory 集合）维护小工具：

用在测试重跑前需要清空向量库、或想查看当前已存入了哪些记忆的场景。

调用示例（在根目录下执行）：
    python -m memory.memory_manager            # 交互式菜单
    python -m memory.memory_manager --list     # 查看全部记忆
    python -m memory.memory_manager --clear    # 一键清空全部记忆
"""
import sys

from memory.memory_store import memory_store  # 复用同一 Chroma 集合，保证连的是同一个库


def list_memory() -> None:
    """查看 agent_memory 集合中已插入的所有内容。"""
    collection = memory_store.collection
    data = collection.get()  # 不过滤地取出该集合全部记录
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    if not ids:
        print("\n📂 集合 agent_memory 当前为空，还没有任何记忆。\n")
        return

    print(f"\n📂 集合 agent_memory 中共有 {len(ids)} 条记忆：\n")
    for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas), 1):
        source_type = (meta or {}).get("source_type", "-")
        timestamp = (meta or {}).get("timestamp", "-")
        print(f"[{i}] {doc or ''}")
        print(f"    source_type: {source_type} | timestamp: {timestamp} | id: {cid}")


def clear_memory() -> None:
    """一键清空 agent_memory 集合中的全部内容。"""
    collection = memory_store.collection
    count = collection.count()
    if count == 0:
        print("✅ 集合 agent_memory 本来就是空的，无需清空。")
        return

    data = collection.get()
    ids = data.get("ids") or []
    if ids:  # 防御：count>0 但未取到 id 时不误删
        collection.delete(ids=ids)
    print(f"🧹 已删除集合 agent_memory 中的 {count} 条记忆。")


def _run_cli() -> None:
    """交互式菜单（或命令行参数快捷操作）。"""
    args = sys.argv[1:]
    if "--list" in args:
        list_memory()
        return
    if "--clear" in args:
        clear_memory()
        return

    print("🧠 长期记忆维护后台")
    while True:
        print("\n请选择操作:")
        print("  1. 查看全部记忆")
        print("  2. 一键清空全部记忆")
        print("  0. 退出")
        choice = input("> ").strip()
        if choice == "1":
            list_memory()
        elif choice == "2":
            confirm = input("确认清空 agent_memory 全部记忆？(y/n): ").strip().lower()
            clear_memory() if confirm == "y" else print("已取消。")
        elif choice == "0":
            print("再见！")
            break
        else:
            print("无效选择，请重试。")


if __name__ == "__main__":
    _run_cli()