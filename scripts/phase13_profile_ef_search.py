"""Phase 13: profile hotspot #3 -- HNSW ef_search (query-time candidate list size), named as a
candidate in the walkthrough's optimization list. Default is pgvector's own default (40, confirmed
via SHOW); sweeps against the real golden set's DOC/HYBRID dense-only recall@10, same methodology
as scripts/phase13_profile_rerank.py.
"""
import json
import math
import time

import psycopg
from sentence_transformers import SentenceTransformer

from nusalens.config import PROJECT_ROOT, settings

MODEL_NAME = "intfloat/multilingual-e5-small"


def recall_at_k(ranked_ids, gold_ids, k):
    if not gold_ids:
        return float("nan")
    top = set(ranked_ids[:k])
    return 1.0 if top & gold_ids else 0.0


def load_golden_retrieval():
    golden_path = PROJECT_ROOT / "eval" / "golden.jsonl"
    items = [json.loads(l) for l in golden_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in items if r["type"] in ("DOC", "HYBRID") and r["gold_chunk_ids"]]


def profile(ef_search: int) -> dict:
    items = load_golden_retrieval()
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    recalls, latencies = [], []
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
        for item in items:
            gold_ids = set(item["gold_chunk_ids"])
            qvec = model.encode(f"query: {item['question']}", normalize_embeddings=True)
            vec_str = "[" + ",".join(str(v) for v in qvec.tolist()) + "]"
            t0 = time.perf_counter()
            cur.execute(
                "SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT 10",
                (vec_str,),
            )
            ranked = [r[0] for r in cur.fetchall()]
            latencies.append((time.perf_counter() - t0) * 1000)
            recalls.append(recall_at_k(ranked, gold_ids, 10))

    latencies.sort()
    valid_recalls = [r for r in recalls if not math.isnan(r)]
    return {
        "ef_search": ef_search,
        "n_items": len(items),
        "mean_recall_at_10": sum(valid_recalls) / len(valid_recalls),
        "query_ms_mean": sum(latencies) / len(latencies),
        "query_ms_p50": latencies[len(latencies) // 2],
        "query_ms_p95": latencies[int(len(latencies) * 0.95)],
    }


def main() -> None:
    results = [profile(ef) for ef in (20, 40, 100)]
    for r in results:
        print(json.dumps(r, indent=2))
    out_path = PROJECT_ROOT / "reports" / "raw" / "13_profile_ef_search.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
