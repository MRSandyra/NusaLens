"""Phase 6: quality filter the synthetic (question, chunk) pairs, with pass-rate accounting."""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from nusalens.config import PROJECT_ROOT, settings
from nusalens.retrieval.search_sparse import SparseSearcher

META_PHRASES = [
    "berdasarkan teks di atas", "dokumen ini", "paragraf ini", "teks ini",
    "teks di atas", "artikel ini", "disebutkan dalam teks", "disajikan dalam teks",
    "dalam teks ini", "dalam cerita ini", "disebutkan dalam cerita ini",
    "bacaan ini", "dalam bacaan ini",
]


def word_count_ok(question: str) -> bool:
    n = len(question.split())
    return 5 <= n <= 40


def has_meta_phrase(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in META_PHRASES)


def round_trip_ok(question: str, chunk_id: str, searcher: SparseSearcher, k: int = 50) -> bool:
    hits = searcher.search(question, k=k)
    return any(h.chunk_id == chunk_id for h in hits)


def dedup_above_threshold(records: list[dict], model: SentenceTransformer, threshold: float = 0.97) -> list[dict]:
    if not records:
        return records
    texts = [f"query: {r['question']}" for r in records]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    keep_mask = np.ones(len(records), dtype=bool)
    kept_indices: list[int] = []
    for i in range(len(records)):
        if not keep_mask[i]:
            continue
        if kept_indices:
            sims = emb[kept_indices] @ emb[i]
            if np.any(sims > threshold):
                keep_mask[i] = False
                continue
        kept_indices.append(i)
    return [r for r, keep in zip(records, keep_mask) if keep]


def main() -> None:
    raw_path = PROJECT_ROOT / "data" / "silver" / "pairs_raw.jsonl"
    records = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    stats = {"raw": len(records)}

    records = [r for r in records if word_count_ok(r["question"])]
    stats["after_word_count"] = len(records)

    records = [r for r in records if not has_meta_phrase(r["question"])]
    stats["after_meta_phrase"] = len(records)

    searcher = SparseSearcher(settings.db_dsn)
    survivors = []
    for r in records:
        if round_trip_ok(r["question"], r["chunk_id"], searcher):
            survivors.append(r)
    records = survivors
    stats["after_round_trip_bm25_top50"] = len(records)

    model = SentenceTransformer(settings.embedding_model, device="cuda")
    records = dedup_above_threshold(records, model, threshold=0.97)
    stats["after_dedup_0.97"] = len(records)

    out_path = PROJECT_ROOT / "data" / "silver" / "pairs_filtered.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats_path = PROJECT_ROOT / "data" / "silver" / "pairs_filter_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"wrote {out_path} ({len(records)} pairs)")


if __name__ == "__main__":
    main()
