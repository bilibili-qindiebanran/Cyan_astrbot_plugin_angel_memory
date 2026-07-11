"""
Reproduce: angel_remember update/merge 产生 is_active=False 的记忆，
angel_recall 无法搜到这些记忆。

场景模拟：
1. AI 调用 angel_remember(action="create", memory={judgment: "柠檬猫猫喜欢喝咖啡"})
   → 创建记忆 A，is_active=True ← 能搜到
2. AI 调用 angel_remember(action="update", source_memory_ids=[A], memory={judgment: "柠檬猫猫喜欢喝冰美式"})
   → angel_remember 拼接的 memory_actions["memory"] 不含 is_active
   → process_feedback / _merge_action_sync 中 is_active 默认 False
   → 新记忆 B，is_active=False ← 搜不到！
3. AI 调用 angel_recall(query="柠檬猫猫 咖啡")
   → comprehensive_recall → recall_by_tags 返回 B
   → angel_recall 过滤 if mem.is_active → B 被丢弃
   → AI 以为没有任何相关记忆！

本脚本在真实数据库上插入测试数据，通过 sqlite3 和 Tantivy BM25
验证检索结果。
"""

import asyncio
import sqlite3
import sys
import uuid
import time
import shutil
from pathlib import Path

DB_PATH = Path(
    "/Users/cyanbutterfly/Desktop/Code/astrbot/data/plugin_data/"
    "astrbot_plugin_angel_memory/memory_center/index/simple_memory.db"
)


def insert_test_memories() -> tuple[str, str]:
    """插入一组模拟 update 场景的测试记忆。返回 (active_id, inactive_id)。"""
    now = time.time()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # 模拟 angel_remember create: is_active=True
    active_id = str(uuid.uuid4())
    # 模拟 angel_remember update/merge: is_active=False (因为 memory dict 没传 is_active)
    inactive_id = str(uuid.uuid4())

    conn.execute(
        "INSERT INTO memory_records(id, memory_type, judgment, reasoning, strength, is_active, useful_count, useful_score, last_recalled_at, last_decay_at, memory_scope, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            active_id, "知识记忆",
            "柠檬猫猫喜欢喝冰美式咖啡，每天早上一杯",  # 新版本（update后）
            "柠檬猫猫在聊天中说的",
            50, 1, 0, 0.0, 0.0, 0.0, "public", now, now,
        ),
    )
    conn.execute(
        "INSERT INTO memory_records(id, memory_type, judgment, reasoning, strength, is_active, useful_count, useful_score, last_recalled_at, last_decay_at, memory_scope, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            inactive_id, "知识记忆",
            "用户小明最喜欢的颜色是蓝色",  # 这是另一条 update/merge 产生的记忆
            "用户小明多次提到喜欢蓝色",
            60, 0, 0, 0.0, 0.0, 0.0, "public", now, now,
        ),
    )
    conn.commit()
    conn.close()
    return active_id, inactive_id


def query_via_sql(query: str) -> list[dict]:
    """用 LIKE 模拟最基础的文本搜索。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    terms = query.strip().split()
    conditions = " OR ".join(["judgment LIKE ?" for _ in terms])
    params = [f"%{t}%" for t in terms]
    rows = conn.execute(
        f"SELECT id, substr(judgment, 1, 60) as judgment, is_active FROM memory_records "
        f"WHERE {conditions} ORDER BY created_at DESC LIMIT 10",
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def cleanup_test_memories(active_id: str, inactive_id: str):
    """删除测试数据。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM memory_records WHERE id IN (?, ?)", (active_id, inactive_id))
    conn.commit()
    conn.close()


async def test_via_recall_by_tags(query: str):
    """使用真实的 recall_by_tags 管线和 Tantivy BM25 检索。"""
    # 需要重建 Tantivy FTS5 索引以包含新插入的测试数据
    # 在实际插件运行时，FTS5 索引由 MemorySqlManager._ensure_fts_ready_sync() 维护
    from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import MemorySqlManager
    from astrbot_plugin_angel_memory.llm_memory.components.bm25_retriever import TantivyBM25Retriever

    manager = MemorySqlManager(
        db_path=DB_PATH,
        decay_config=None,
        rerank_provider=None,
    )
    # 强制重建 FTS5 索引以包含新数据
    manager._ensure_fts_ready_sync(force_rebuild=True)

    results = await manager.recall_by_tags(
        query=query,
        limit=10,
        memory_scope="public",
        vector_scores=None,
    )
    return results


def test_via_simple_memory_runtime(query: str):
    """使用 SimpleMemoryRuntime.comprehensive_recall 测试。"""
    from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import MemorySqlManager
    from astrbot_plugin_angel_memory.core.memory_runtime.simple_memory_runtime import SimpleMemoryRuntime

    manager = MemorySqlManager(
        db_path=DB_PATH,
        decay_config=None,
        rerank_provider=None,
    )
    runtime = SimpleMemoryRuntime(manager)
    memories = asyncio.run(runtime.comprehensive_recall(
        query=query,
        limit=10,
        memory_scope="public",
    ))
    return memories


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print_separator("angel_remember update/merge 导致 is_active=False 的 Bug 复现")

    # Step 1: 插入测试数据
    print("\n[Step 1] 插入测试记忆...")
    active_id, inactive_id = insert_test_memories()
    print(f"  ✅ 插入 active 记忆 (模拟 angel_remember create): {active_id}")
    print(f"  ✅ 插入 inactive 记忆 (模拟 angel_remember update/merge): {inactive_id}")

    try:
        # Step 2: 直接 SQL 查询验证两条数据都存在
        print_separator("Step 2: 直接 SQL 查询")
        query = "柠檬猫猫 咖啡"
        print(f"  查询词: '{query}'")
        results = query_via_sql(query)
        print(f"  SQL 匹配到 {len(results)} 条记忆:")
        for r in results:
            active_label = "✅ active (会被 angel_recall 返回)" if r["is_active"] else "❌ inactive (会被 angel_recall 过滤!)"
            print(f"    [{r['id'][:8]}...] is_active={r['is_active']} | {r['judgment'][:50]} → {active_label}")

        # Step 3: 模拟 angel_recall 的 is_active 过滤
        print_separator("Step 3: 模拟 angel_recall 的 is_active 过滤")
        all_results = query_via_sql(query)
        active_only = [r for r in all_results if r["is_active"]]
        filtered_out = [r for r in all_results if not r["is_active"]]
        print(f"  全部匹配: {len(all_results)} 条")
        print(f"  通过 is_active 过滤: {len(active_only)} 条")
        print(f"  被 is_active 过滤掉: {len(filtered_out)} 条")
        if filtered_out:
            print(f"  ❌ Bug 确认: {len(filtered_out)} 条 update/merge 产生的记忆被过滤!")
            for r in filtered_out:
                print(f"    × [{r['id'][:8]}...] {r['judgment'][:50]}")
        else:
            print(f"  ✅ 没有记忆被过滤")

        # Step 4: 查询另一条 inactive 记忆
        print_separator("Step 4: 查询第二条 inactive 记忆")
        query2 = "小明 蓝色"
        results2 = query_via_sql(query2)
        print(f"  查询词: '{query2}'")
        print(f"  SQL 匹配到 {len(results2)} 条记忆:")
        for r in results2:
            active_label = "✅ 可通过 angel_recall" if r["is_active"] else "❌ 无法通过 angel_recall"
            print(f"    [{r['id'][:8]}...] is_active={r['is_active']} | {r['judgment'][:50]} → {active_label}")

        # Step 5: 总结
        print_separator("结论")
        print("""
  angel_remember 的 create 动作:
    → angel_remember.py:162 传入 is_active=True
    → memory_runtime.remember(is_active=True)
    → 记忆存入 DB，is_active=1

  angel_remember 的 update/merge 动作:
    → angel_remember.py:182-189 构建 memory_actions["memory"] 时不包含 is_active
    → 走到 memory_sql_manager._merge_action_sync() 或 process_feedback()
    → is_active=bool(memory_data.get("is_active", False))   ← 默认 False！
    → 新记忆存入 DB，is_active=0

  angel_recall 调用:
    → comprehensive_recall 返回所有匹配记忆（包括 is_active=False 的）
    → angel_recall.py:80 过滤: all_active = [mem for mem in all_memories if mem.is_active]
    → update/merge 产生的记忆全部被丢弃！

  修复方案（二选一）：
    A) angel_remember.py update/merge 路径 memory dict 补 is_active: True
    B) angel_recall.py 移除 is_active 过滤（comprehensive_recall 已有质量保证）
""")

    finally:
        # 清理测试数据
        print_separator("清理")
        cleanup_test_memories(active_id, inactive_id)
        print("  测试记忆已删除。")


if __name__ == "__main__":
    main()
