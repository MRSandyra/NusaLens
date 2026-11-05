"""Phase 6: hard negative mining for the train split, full corpus as the negative pool."""
import json

import pyarrow.parquet as pq
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

from nusalens.config import PROJECT_ROOT, settings


def main() -> None:
    split_path = PROJECT_ROOT / "data" / "silver" / "pairs_split.jsonl"
    pairs = [json.loads(l) for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    train_pairs = [p for p in pairs if p["split"] == "train"]

    chunks_df = pq.read_table(PROJECT_ROOT / "data" / "gold" / "chunks.parquet").to_pandas()
    chunk_text = dict(zip(chunks_df["chunk_id"], chunks_df["text"]))
    corpus = chunks_df["text"].tolist()

    dataset = Dataset.from_dict({
        "anchor": [p["question"] for p in train_pairs],
        "positive": [chunk_text[p["chunk_id"]] for p in train_pairs],
    })

    model = SentenceTransformer(settings.embedding_model, device="cuda")
    triplets = mine_hard_negatives(
        dataset,
        model,
        corpus=corpus,
        range_min=10,
        range_max=50,
        max_score=0.95,
        num_negatives=3,
        query_prompt="query: ",
        corpus_prompt="passage: ",
        output_format="triplet",
        batch_size=128,
        use_faiss=False,
        verbose=True,
    )

    out_path = PROJECT_ROOT / "data" / "silver" / "pairs_train_triplets.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in triplets:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(triplets)} triplets from {len(train_pairs)} train pairs)")


if __name__ == "__main__":
    main()
