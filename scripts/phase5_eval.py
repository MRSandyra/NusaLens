"""Phase 5: Evaluate five retrieval configurations and produce comparison report."""
import argparse
import json
import math
import time
import uuid
from pathlib import Path

import psycopg

from nusalens.config import PROJECT_ROOT, settings
from nusalens.retrieval.hybrid_search import HybridSearcher


def recall_at_k(ranked_ids, gold_ids, k):
    if not gold_ids:
        return float("nan")
    top = set(ranked_ids[:k])
    return 1.0 if top & gold_ids else 0.0


def mrr_at_k(ranked_ids, gold_ids, k):
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, gold_ids, k):
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def load_golden_retrieval():
    golden_path = PROJECT_ROOT / "eval" / "golden.jsonl"
    items = [json.loads(l) for l in golden_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in items if r["type"] in ("DOC", "HYBRID") and r["gold_chunk_ids"]]


def evaluate_mode(mode, tag):
    """Evaluate a single retrieval mode and return metrics dict."""
    items = load_golden_retrieval()
    searcher = HybridSearcher(settings.db_dsn, device="cpu")

    per_item = []
    latencies = []

    for item in items:
        gold_ids = set(item["gold_chunk_ids"])
        q = item["question"]
        t0 = time.perf_counter()

        if mode == "dense":
            hits = searcher.dense.search(q, k=10)
        elif mode == "sparse":
            hits = searcher.sparse.search(q, k=10)
        elif mode == "rrf":
            hits = searcher.search_rrf(q, k=10)
        elif mode == "rrf_rerank":
            # k=10, not the production final_top_k=8, so recall@10 is comparable across modes.
            hits = searcher.search_rrf_rerank(q, k=10)
        elif mode == "rrf_rerank_exp":
            hits = searcher.search_rrf_rerank_exp(q, k=10)
        else:
            raise ValueError(f"unknown mode {mode}")

        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [h.chunk_id for h in hits]
        per_item.append({
            "qid": item["qid"],
            "question": q,
            "gold_chunk_ids": sorted(gold_ids),
            "ranked": ranked,
            "recall_1": recall_at_k(ranked, gold_ids, 1),
            "recall_10": recall_at_k(ranked, gold_ids, 10),
            "mrr_10": mrr_at_k(ranked, gold_ids, 10),
            "ndcg_10": ndcg_at_k(ranked, gold_ids, 10),
        })

    def avg(key):
        vals = [p[key] for p in per_item if not math.isnan(p[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    metrics = {
        "mode": mode,
        "tag": tag,
        "n_items": len(items),
        "recall_at_1": avg("recall_1"),
        "recall_at_10": avg("recall_10"),
        "mrr_at_10": avg("mrr_10"),
        "ndcg_at_10": avg("ndcg_10"),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }
    return metrics, per_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", nargs="+",
        default=["dense", "sparse", "rrf", "rrf_rerank", "rrf_rerank_exp"],
    )
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in args.modes:
        tag = f"phase5-{mode}"
        print(f"Evaluating {mode}...")
        metrics, per_item = evaluate_mode(mode, tag)
        results.append(metrics)
        print(f"  recall@1={metrics['recall_at_1']:.2%}, recall@10={metrics['recall_at_10']:.2%}, ndcg@10={metrics['ndcg_at_10']:.3f}")

        run_id = str(uuid.uuid4())
        detail_path = results_dir / f"{tag}.json"
        detail_path.write_text(json.dumps({
            "run_id": run_id,
            "tag": tag,
            "config": {"mode": mode},
            "metrics": metrics,
            "per_item": per_item,
        }, indent=2), encoding="utf-8")

        with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO eval_run (run_id, tag, finished_at, metrics) "
                "VALUES (%s, %s, NOW(), %s) "
                "ON CONFLICT (run_id) DO NOTHING",
                (run_id, tag, json.dumps(metrics)),
            )
            conn.commit()

    # Save summary
    summary_path = PROJECT_ROOT / "reports" / "raw" / "05_retrieval_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {summary_path}")

    # Print comparison table
    print("\n=== Comparison Table ===")
    print("Mode       | Recall@1 | Recall@10 | MRR@10 | nDCG@10 | p50ms | p95ms")
    print("-----------|----------|-----------|--------|---------|-------|-------")
    for r in results:
        print(
            f"{r['mode']:10} | {r['recall_at_1']:8.1%} | {r['recall_at_10']:9.1%} | {r['mrr_at_10']:6.3f} | {r['ndcg_at_10']:7.3f} | {r['latency_p50_ms']:5.1f} | {r['latency_p95_ms']:6.1f}"
        )


if __name__ == "__main__":
    main()
