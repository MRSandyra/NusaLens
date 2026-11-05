"""Phase 13: profile hotspot #1 -- reranking is the single largest DOC/HYBRID latency cost
(Phase 11 measured rerank_ms mean ~9.1s, dwarfing retrieve_ms ~0.2s). k_rerank (how many RRF
candidates get fed to the CPU cross-encoder) is the one tunable lever; this sweeps it against the
real golden set and measures both latency and the nDCG@10 cost, reusing Phase 5's exact metric
functions so the numbers are directly comparable to reports/05_hybrid_search.md.
"""
import json
import math

from nusalens.config import PROJECT_ROOT, settings
from nusalens.retrieval.hybrid_search import HybridSearcher

# Duplicated from scripts/phase5_eval.py rather than imported: scripts/ is not a package, and every
# other phase script in this project follows the same standalone-file convention (see D11).


def recall_at_k(ranked_ids, gold_ids, k):
    if not gold_ids:
        return float("nan")
    top = set(ranked_ids[:k])
    return 1.0 if top & gold_ids else 0.0


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


def profile(k_rerank: int) -> dict:
    items = load_golden_retrieval()
    searcher = HybridSearcher(settings.db_dsn, device="cpu")

    ndcgs, recalls, rerank_times = [], [], []
    for item in items:
        gold_ids = set(item["gold_chunk_ids"])
        q = item["question"]
        hits_scored, retrieve_ms, rerank_ms = searcher.search_rrf_rerank_exp_timed(q, k=10, k_rerank=k_rerank)
        rerank_times.append(rerank_ms)
        ranked = [h.chunk_id for h, _ in hits_scored]
        ndcgs.append(ndcg_at_k(ranked, gold_ids, 10))
        recalls.append(recall_at_k(ranked, gold_ids, 10))

    rerank_times.sort()
    return {
        "k_rerank": k_rerank,
        "n_items": len(items),
        "mean_ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "mean_recall_at_10": sum(r for r in recalls if not math.isnan(r)) / len([r for r in recalls if not math.isnan(r)]),
        "rerank_ms_mean": sum(rerank_times) / len(rerank_times),
        "rerank_ms_p50": rerank_times[len(rerank_times) // 2],
        "rerank_ms_p95": rerank_times[int(len(rerank_times) * 0.95)],
    }


def main() -> None:
    results = [profile(k) for k in (30, 15, 10)]
    for r in results:
        print(json.dumps(r, indent=2))
    out_path = PROJECT_ROOT / "reports" / "raw" / "13_profile_rerank.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
