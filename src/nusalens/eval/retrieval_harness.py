import argparse
import json
import math
import time
import uuid
from pathlib import Path

from nusalens.config import PROJECT_ROOT, settings
from nusalens.retrieval.search_dense import DenseSearcher

import psycopg


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    top = set(ranked_ids[:k])
    return 1.0 if top & gold_ids else 0.0


def mrr_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def load_golden(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["type"] in ("DOC", "HYBRID") and r["gold_chunk_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline-dense")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    golden_path = PROJECT_ROOT / "eval" / "golden.jsonl"
    items = load_golden(golden_path)
    print(f"evaluating retrieval over {len(items)} DOC/HYBRID items (k={args.k})")

    searcher = DenseSearcher(settings.db_dsn, device="cpu")

    per_item = []
    latencies = []
    for item in items:
        gold_ids = set(item["gold_chunk_ids"])
        t0 = time.perf_counter()
        hits = searcher.search(item["question"], k=args.k)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [h.chunk_id for h in hits]
        per_item.append({
            "qid": item["qid"],
            "recall_1": recall_at_k(ranked, gold_ids, 1),
            "recall_3": recall_at_k(ranked, gold_ids, 3),
            "recall_5": recall_at_k(ranked, gold_ids, 5),
            "recall_10": recall_at_k(ranked, gold_ids, 10),
            "mrr_10": mrr_at_k(ranked, gold_ids, 10),
            "ndcg_10": ndcg_at_k(ranked, gold_ids, 10),
            "hit": 1.0 if any(r in gold_ids for r in ranked) else 0.0,
        })

    def avg(key: str) -> float:
        vals = [p[key] for p in per_item if not math.isnan(p[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    metrics = {
        "n_items": len(items),
        "recall_at_1": avg("recall_1"),
        "recall_at_3": avg("recall_3"),
        "recall_at_5": avg("recall_5"),
        "recall_at_10": avg("recall_10"),
        "mrr_at_10": avg("mrr_10"),
        "ndcg_at_10": avg("ndcg_10"),
        "hit_rate": avg("hit"),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }

    print(json.dumps(metrics, indent=2))

    run_id = str(uuid.uuid4())
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{args.tag}.json"
    result_path.write_text(json.dumps({
        "run_id": run_id,
        "tag": args.tag,
        "config": {"mode": "dense", "model": "intfloat/multilingual-e5-small", "k": args.k},
        "metrics": metrics,
        "per_item": per_item,
    }, indent=2), encoding="utf-8")
    print(f"wrote {result_path}")

    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO eval_run (run_id, tag, finished_at, metrics) "
            "VALUES (%s, %s, NOW(), %s) "
            "ON CONFLICT (run_id) DO NOTHING",
            (run_id, args.tag, json.dumps(metrics)),
        )
        conn.commit()
    print(f"recorded eval_run row run_id={run_id}")


if __name__ == "__main__":
    main()
