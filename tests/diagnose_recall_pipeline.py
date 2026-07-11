"""
诊断 angel_recall 为什么搜不到 "青蝶半染"。
逐步追踪 comprehensive_recall → BM25 → 分数过滤 每个环节。
"""

import asyncio
import time
from pathlib import Path

DB_PATH = Path(
    "/Users/cyanbutterfly/Desktop/Code/astrbot/data/plugin_data/"
    "astrbot_plugin_angel_memory/memory_center/index/simple_memory.db"
)


async def main():
    from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import MemorySqlManager

    manager = MemorySqlManager(
        db_path=DB_PATH,
        decay_config=None,
        rerank_provider=None,
    )

    query = "青蝶半染"

    # Step 1: 确保 Tantivy 索引最新
    print("=" * 60)
    print("Step 1: 重建 Tantivy FTS5 索引")
    start = time.time()
    manager._ensure_fts_ready_sync(force_rebuild=True)
    print(f"  索引重建完成，耗时 {int((time.time() - start) * 1000)}ms")

    # Step 2: BM25 仅搜索 (Tantivy)
    print("=" * 60)
    print(f"Step 2: Tantivy BM25 搜索 query='{query}'")
    bm25_hits = manager._fts_retriever.search_memory_bm25_only(query=query, limit=20)
    print(f"  BM25 命中: {len(bm25_hits)} 条")
    for h in bm25_hits[:5]:
        print(f"    id={h['id'][:16]}... bm25_score={h.get('bm25_score',0):.4f} norm_score={h.get('bm25_normalized_score',0):.4f} final={h.get('final_score',0):.4f}")

    # Step 3: _staged_search 内部细节
    print("=" * 60)
    print("Step 3: _staged_search CJK bigram 分词")
    from astrbot_plugin_angel_memory.llm_memory.components.note_chunk_search import (
        select_required_cjk_ngrams, iter_cjk_fragments, _build_all_terms_query, _build_any_terms_query
    )
    required = select_required_cjk_ngrams(query)
    all_terms = iter_cjk_fragments(query, include_unigram=True, max_ngram=2)
    strict_query = _build_all_terms_query(required)
    loose_query = _build_any_terms_query(all_terms)
    print(f"  required_terms (AND): {required}")
    print(f"  all_terms (OR): {all_terms}")
    print(f"  strict(AND) query: '{strict_query}'")
    print(f"  loose(OR) query: '{loose_query}'")

    # Step 4: 检查 Tantivy 索引中文档的 token 化
    print("=" * 60)
    print("Step 4: 检查目标记忆在 Tantivy 索引中是否存在")
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    target_rows = conn.execute(
        "SELECT id, substr(judgment, 1, 80) as judgment FROM memory_records WHERE judgment LIKE '%青蝶半染%' LIMIT 5"
    ).fetchall()
    conn.close()

    for row in target_rows:
        mem_id = row["id"]
        judgment = row["judgment"]
        # 在 BM25 结果中查找
        found = any(h["id"] == mem_id for h in bm25_hits)
        status = "✅ 命中" if found else "❌ 未命中"
        print(f"  [{mem_id[:16]}...] {status} | {judgment[:50]}")

    # Step 5: recall_by_tags 完整管线
    print("=" * 60)
    print("Step 5: recall_by_tags 完整管线 (无向量分数)")
    results = await manager.recall_by_tags(
        query=query,
        limit=20,
        memory_scope="public",
        vector_scores=None,
    )
    print(f"  最终返回: {len(results)} 条")
    for r in results:
        print(f"    id={r.id[:16]}... sim={getattr(r,'similarity',0):.4f} | {getattr(r,'judgment','')[:50]}")

    # Step 6: 分数过滤检查
    print("=" * 60)
    print("Step 6: 分数过滤分析 (score_kind != 'rrf' and final_score < 0.5 → 丢弃)")

    # 直接调用 hybrid_engine 看所有候选
    text = query.strip()
    # 模仿 recall_by_tags 中的候选策略
    candidate_limit = max(20, 20 * 3)  # limit=20 → candidate=60
    bm25_limit = max(50, 20 * 3)  # =60

    hits = await manager._hybrid_engine.search_with_strategy(
        query=text,
        limit=candidate_limit,
        candidate_limit=candidate_limit,
        bm25_limit=bm25_limit,
        vector_scores=None,  # 无向量
        bm25_only_search=lambda q, k: manager._fts_retriever.search_memory_bm25_only(query=q, limit=k),
        fusion_search=lambda q, k, bk, scores: manager._fts_retriever.search_memory(
            query=q, limit=k, fts_limit=bk, fts_weight=0.3, vector_weight=0.7, vector_scores=scores,
        ),
        build_doc_text_map=manager._build_memory_doc_text_map_by_ids,
    )
    print(f"  hybrid_engine 返回候选: {len(hits)} 条")
    for h in hits[:10]:
        score_kind = h.get("score_kind", "?")
        final = float(h.get("final_score", 0.0))
        passed = not (score_kind != "rrf" and final < 0.5)
        status = "✅ 通过" if passed else "❌ 被过滤(score<0.5)"
        print(f"    id={h['id'][:16]}... kind={score_kind} final={final:.4f} {status}")

    if not hits:
        print("  ❌ hybrid_engine 返回空！问题在 BM25/Tantivy 层面")


if __name__ == "__main__":
    asyncio.run(main())
