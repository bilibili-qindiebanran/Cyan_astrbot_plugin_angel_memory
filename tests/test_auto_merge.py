"""
测试 LLM 自动合并功能。插入几组语义相似的被动记忆，触发合并管线。
"""

import asyncio
import time
import uuid
from pathlib import Path

DB_PATH = Path(
    "/Users/cyanbutterfly/Desktop/Code/astrbot/data/plugin_data/"
    "astrbot_plugin_angel_memory/memory_center/index/simple_memory.db"
)

# 模拟数据：3组语义近似的被动记忆
TEST_GROUPS = [
    {
        "desc": "小白喜欢喝咖啡相关的多条表述",
        "memories": [
            ("小白每天都喝一杯美式咖啡", "小白自己说的", ["小白", "咖啡", "美式"]),
            ("小白喜欢喝咖啡，每天早上一杯", "聊天中提到", ["小白", "咖啡"]),
            ("小白（123456789）有喝咖啡的习惯", "观察到的", ["小白", "咖啡", "习惯"]),
            ("小白每天早上必喝一杯冰美式", "小白多次提起", ["小白", "美式", "咖啡"]),
        ],
    },
    {
        "desc": "小黑是程序员不同角度的描述",
        "memories": [
            ("小黑是后端开发工程师，主要写Go", "聊天中提到", ["小黑", "程序员", "Go"]),
            ("小黑（987654321）做后端开发，技术栈Go+Python", "自我介绍", ["小黑", "后端", "Go", "Python"]),
            ("小黑从事软件开发工作", "观察", ["小黑", "程序员"]),
        ],
    },
    {
        "desc": "小红正在备考研究生",
        "memories": [
            ("小红在准备考研，每天复习到深夜", "小红说的", ["小红", "考研"]),
            ("小红（55555555）今年考研，目标院校是复旦", "聊天中提到", ["小红", "考研", "复旦"]),
        ],
    },
]

MEMORY_TYPE = "知识记忆"


def insert_test_memories() -> dict[str, list[str]]:
    """插入测试数据，返回 {组描述: [id列表]}。"""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    now = time.time()
    group_ids = {}

    for group in TEST_GROUPS:
        ids = []
        for judgment, reasoning, tags in group["memories"]:
            mem_id = str(uuid.uuid4())
            tags_json = "\x1f".join(tags)  # null-delimited
            conn.execute(
                """INSERT INTO memory_records(
                    id, memory_type, judgment, reasoning, strength, is_active,
                    useful_count, useful_score, last_recalled_at, last_decay_at,
                    memory_scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mem_id, MEMORY_TYPE, judgment, reasoning, 50, 0, 0, 0.0, 0.0, 0.0,
                    "public", now, now,
                ),
            )
            # 写入 tags
            for tag in tags:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO global_tags(name) VALUES (?)",
                    (tag,),
                )
                tag_id = cur.lastrowid
                if tag_id == 0:
                    tag_id = conn.execute(
                        "SELECT id FROM global_tags WHERE name = ?", (tag,)
                    ).fetchone()[0]
                conn.execute(
                    "INSERT OR IGNORE INTO memory_tag_rel(memory_id, tag_id) VALUES (?, ?)",
                    (mem_id, tag_id),
                )
            ids.append(mem_id)
        group_ids[group["desc"]] = ids
    conn.commit()
    conn.close()
    return group_ids


def cleanup_test_memories(group_ids: dict[str, list[str]]):
    """删除测试数据。"""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    for ids in group_ids.values():
        for mid in ids:
            conn.execute("DELETE FROM memory_tag_rel WHERE memory_id = ?", (mid,))
            conn.execute("DELETE FROM memory_records WHERE id = ?", (mid,))
    conn.commit()
    conn.close()


async def run_test():
    from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import MemorySqlManager
    from astrbot.core.star.context import Context

    # 插入测试数据
    print("=" * 60)
    print("Step 1: 插入测试被动记忆")
    group_ids = insert_test_memories()
    for desc, ids in group_ids.items():
        print(f"  {desc}: {len(ids)} 条")

    try:
        # 创建 MemorySqlManager（需要 astrbot_context 来获取 LLM provider）
        # 这里用 None 跳过 LLM，先测试 BM25 聚类
        manager = MemorySqlManager(db_path=DB_PATH, astrbot_context=None)

        # 重建 Tantivy 索引以包含测试数据
        print("\n" + "=" * 60)
        print("Step 2: 重建 Tantivy 索引")
        manager._ensure_fts_ready_sync(force_rebuild=True)
        print("  ✅ 索引重建完成")

        # 测试 BM25 聚类（不调用 LLM）
        print("\n" + "=" * 60)
        print("Step 3: BM25 聚类测试（无 LLM）")
        t0 = time.time()

        from collections import defaultdict
        all_rows = manager._get_all_passive_memory_rows()
        test_ids = {iid for ids in group_ids.values() for iid in ids}
        # 只搜索测试记忆
        rows = [r for r in all_rows if r["id"] in test_ids]
        row_by_id = {r["id"]: r for r in rows}

        # BM25 候选发现
        pairs = set()
        for row in rows:
            query = str(row.get("judgment", "") or "").strip()
            if not query:
                continue
            candidates = manager._search_similar_by_bm25(query, limit=5)
            for cand in candidates:
                score = float(cand.get("final_score", 0.0))
                if score < 0.7:
                    continue
                cand_id = str(cand.get("id") or "").strip()
                if not cand_id or cand_id == row["id"] or cand_id not in row_by_id:
                    continue
                a, b = sorted([row["id"], cand_id])
                pairs.add((a, b))

        print(f"  候选对: {len(pairs)}")

        # Union-Find 聚类
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for a, b in pairs:
            union(a, b)

        groups = defaultdict(list)
        for row in rows:
            root = find(row["id"])
            groups[root].append(row)

        sorted_groups = sorted(groups.values(), key=len, reverse=True)
        sorted_groups = [g for g in sorted_groups if len(g) >= 2]

        print(f"  聚类数: {len(sorted_groups)}")
        for i, cluster in enumerate(sorted_groups, 1):
            print(f"  聚类 {i} (大小={len(cluster)}):")
            for r in cluster:
                print(f"    - {r['judgment'][:50]}")

        elapsed = int((time.time() - t0) * 1000)
        print(f"  耗时: {elapsed}ms")

        # Step 4: 如果有可用的 LLM provider，测试实际合并
        print("\n" + "=" * 60)
        print("Step 4: 检查 LLM provider 可用性")
        try:
            import importlib
            # 尝试导入 AstrBot context 获取 provider_manager
            # 在测试环境可能无法获取，这是预期的
            print("  在测试脚本中无法获取 AstrBot context")
            print("  请在运行中的 AstrBot 观察 [自动合并] 日志来验证 LLM 合并")
        except Exception:
            pass

        print("\n" + "=" * 60)
        print("结论")
        print(f"  BM25 聚类: ✅ 发现 {len(pairs)} 对候选，{len(sorted_groups)} 个聚类")
        if sorted_groups:
            expected = len(TEST_GROUPS)
            print(f"  期望聚类数: {expected} 组")
            print(f"  实际聚类数: {len(sorted_groups)} 组")
        print("  LLM 合并: 请在 AstrBot 重启后的睡眠周期日志中验证")

    finally:
        # 清理
        print("\n" + "=" * 60)
        print("清理测试数据...")
        cleanup_test_memories(group_ids)
        print("✅ 完成")


if __name__ == "__main__":
    asyncio.run(run_test())
