"""Phase 6: four-condition comparison plus the Matryoshka dimension-truncation test."""
import json
import math
import time

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer

from nusalens.config import PROJECT_ROOT, settings
from nusalens.retrieval.hybrid_search import HybridSearcher
from nusalens.retrieval.search_dense import DenseSearcher

MODEL_PATH = str(PROJECT_ROOT / "models" / "e5-nusalens-v1")
LARGER_STOCK_MODEL = "intfloat/multilingual-e5-base"
MATRYOSHKA_DIMS = [384, 256, 128, 64]


def recall_at_k(ranked, gold, k):
    if not gold:
        return float("nan")
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr_at_k(ranked, gold, k):
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, gold, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in gold)
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def load_golden():
    path = PROJECT_ROOT / "eval" / "golden.jsonl"
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in items if r["type"] in ("DOC", "HYBRID") and r["gold_chunk_ids"]]


def score_ranked_lists(items, rank_fn, k=10):
    per_item = []
    latencies = []
    for item in items:
        gold = set(item["gold_chunk_ids"])
        t0 = time.perf_counter()
        ranked = rank_fn(item["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        per_item.append({
            "recall_1": recall_at_k(ranked, gold, 1),
            "recall_10": recall_at_k(ranked, gold, 10),
            "mrr_10": mrr_at_k(ranked, gold, 10),
            "ndcg_10": ndcg_at_k(ranked, gold, k),
        })

    def avg(key):
        vals = [p[key] for p in per_item if not math.isnan(p[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    latencies.sort()
    return {
        "n_items": len(items),
        "recall_at_1": avg("recall_1"),
        "recall_at_10": avg("recall_10"),
        "mrr_at_10": avg("mrr_10"),
        "ndcg_at_10": avg("ndcg_10"),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
    }


def eval_condition_1_stock_dense(items):
    s = DenseSearcher(settings.db_dsn, device="cpu")
    return score_ranked_lists(items, lambda q: [h.chunk_id for h in s.search(q, k=10)])


def eval_condition_2_finetuned_dense(items):
    s = DenseSearcher(settings.db_dsn, device="cpu", model_name=MODEL_PATH, embedding_column="embedding_ft")
    return score_ranked_lists(items, lambda q: [h.chunk_id for h in s.search(q, k=10)])


def eval_condition_3_finetuned_rrf_rerank(items):
    s = HybridSearcher(settings.db_dsn, device="cpu", dense_model_name=MODEL_PATH, dense_embedding_column="embedding_ft")
    return score_ranked_lists(items, lambda q: [h.chunk_id for h in s.search_rrf_rerank(q, k=10)])


def eval_condition_4_larger_stock_dense(items, corpus_ids, corpus_emb):
    model = SentenceTransformer(LARGER_STOCK_MODEL, device="cuda")

    def rank_fn(q):
        qvec = model.encode(f"query: {q}", normalize_embeddings=True)
        sims = corpus_emb @ qvec
        top = np.argsort(-sims)[:10]
        return [corpus_ids[i] for i in top]

    return score_ranked_lists(items, rank_fn)


def fetch_embedding_ft():
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT chunk_id, embedding_ft FROM chunks WHERE embedding_ft IS NOT NULL")
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    emb = np.array([[float(x) for x in r[1].strip("[]").split(",")] for r in rows], dtype=np.float32)
    return ids, emb


def matryoshka_test(items):
    ids, full_emb = fetch_embedding_ft()
    model = SentenceTransformer(MODEL_PATH, device="cuda")
    query_full = model.encode([f"query: {it['question']}" for it in items], normalize_embeddings=True)

    results = {}
    for dim in MATRYOSHKA_DIMS:
        corpus_trunc = full_emb[:, :dim]
        corpus_trunc = corpus_trunc / np.linalg.norm(corpus_trunc, axis=1, keepdims=True)
        query_trunc = query_full[:, :dim]
        query_trunc = query_trunc / np.linalg.norm(query_trunc, axis=1, keepdims=True)

        per_item = []
        for item, qvec in zip(items, query_trunc):
            gold = set(item["gold_chunk_ids"])
            sims = corpus_trunc @ qvec
            top = np.argsort(-sims)[:10]
            ranked = [ids[i] for i in top]
            per_item.append(ndcg_at_k(ranked, gold, 10))
        ndcg = sum(per_item) / len(per_item)
        storage_bytes_per_vec = dim * 4
        results[dim] = {
            "ndcg_at_10": ndcg,
            "storage_bytes_per_vector": storage_bytes_per_vec,
            "storage_saving_pct": round(100 * (1 - dim / 384), 1),
        }
    return results


def main() -> None:
    items = load_golden()

    print("condition 1: stock e5-small, dense")
    c1 = eval_condition_1_stock_dense(items)
    print(json.dumps(c1, indent=2))

    print("condition 2: fine-tuned e5-small, dense")
    c2 = eval_condition_2_finetuned_dense(items)
    print(json.dumps(c2, indent=2))

    print("condition 3: fine-tuned e5-small, RRF + rerank")
    c3 = eval_condition_3_finetuned_rrf_rerank(items)
    print(json.dumps(c3, indent=2))

    print("loading larger stock comparison model and encoding corpus (one-off, in-memory)")
    import pyarrow.parquet as pq
    chunks_df = pq.read_table(PROJECT_ROOT / "data" / "gold" / "chunks.parquet").to_pandas()
    base_model = SentenceTransformer(LARGER_STOCK_MODEL, device="cuda")
    corpus_texts = [f"passage: {t}" for t in chunks_df["text"].tolist()]
    corpus_emb = base_model.encode(corpus_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    corpus_ids = chunks_df["chunk_id"].tolist()

    print("condition 4: larger stock model (multilingual-e5-base), dense")
    c4 = eval_condition_4_larger_stock_dense(items, corpus_ids, corpus_emb)
    print(json.dumps(c4, indent=2))

    print("matryoshka dimension test")
    mat = matryoshka_test(items)
    print(json.dumps(mat, indent=2))

    out = {
        "stock_e5_small_dense": c1,
        "finetuned_e5_small_dense": c2,
        "finetuned_e5_small_rrf_rerank": c3,
        "larger_stock_e5_base_dense": c4,
        "matryoshka": mat,
    }
    out_path = PROJECT_ROOT / "reports" / "raw" / "06_finetune_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
