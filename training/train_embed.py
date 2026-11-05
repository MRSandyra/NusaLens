"""Phase 6: fine-tune multilingual-e5-small with MatryoshkaLoss(MultipleNegativesRankingLoss)."""
import json
import time

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

from nusalens.config import PROJECT_ROOT, settings

MATRYOSHKA_DIMS = [384, 256, 128, 64]
EFFECTIVE_BATCH = 128
EPOCHS = 2
OUTPUT_DIR = PROJECT_ROOT / "models" / "e5-nusalens-v1"
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints" / "e5-nusalens-v1"


def load_train_dataset() -> Dataset:
    path = PROJECT_ROOT / "data" / "silver" / "pairs_train_triplets.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return Dataset.from_dict({
        "anchor": [f"query: {r['anchor']}" for r in rows],
        "positive": [f"passage: {r['positive']}" for r in rows],
        "negative": [f"passage: {r['negative']}" for r in rows],
    })


def build_ir_evaluator(split: str, name: str, chunk_text: dict) -> InformationRetrievalEvaluator:
    split_path = PROJECT_ROOT / "data" / "silver" / "pairs_split.jsonl"
    pairs = [json.loads(l) for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [p for p in pairs if p["split"] == split]

    queries = {f"{split}-{i}": r["question"] for i, r in enumerate(rows)}
    relevant_docs = {f"{split}-{i}": {r["chunk_id"]} for i, r in enumerate(rows)}
    corpus = dict(chunk_text)  # full chunk corpus, so ranking reflects real retrieval difficulty

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        query_prompt="query: ",
        corpus_prompt="passage: ",
        truncate_dim=None,
        show_progress_bar=False,
        batch_size=128,
    )


def try_train(model, train_dataset, evaluator, batch_size: int, use_cached: bool) -> None:
    base_loss = (
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=4)
        if use_cached
        else MultipleNegativesRankingLoss(model)
    )
    loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=MATRYOSHKA_DIMS)

    grad_accum = max(1, EFFECTIVE_BATCH // batch_size)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        gradient_checkpointing=use_cached,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()


def main() -> None:
    chunks_df = pq.read_table(PROJECT_ROOT / "data" / "gold" / "chunks.parquet").to_pandas()
    chunk_text = dict(zip(chunks_df["chunk_id"], chunks_df["text"]))

    model = SentenceTransformer(settings.embedding_model, device="cuda")
    train_dataset = load_train_dataset()
    val_evaluator = build_ir_evaluator("val", "nusalens-val", chunk_text)

    peak_vram_mb = 0.0
    batch_size = 32
    use_cached = False
    start = time.time()
    for attempt in range(3):
        try:
            torch.cuda.reset_peak_memory_stats()
            try_train(model, train_dataset, val_evaluator, batch_size, use_cached)
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not use_cached and batch_size > 8:
                batch_size //= 2
                print(f"OOM, retrying at batch_size={batch_size}")
            elif not use_cached:
                use_cached = True
                print("persistent OOM, switching to CachedMultipleNegativesRankingLoss(mini_batch_size=4)")
            else:
                raise
    wall_time_s = time.time() - start

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))

    final_scores = val_evaluator(model)
    run_meta = {
        "batch_size": batch_size,
        "effective_batch": EFFECTIVE_BATCH,
        "used_cached_loss": use_cached,
        "matryoshka_dims": MATRYOSHKA_DIMS,
        "epochs": EPOCHS,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "wall_time_s": round(wall_time_s, 1),
        "n_train_triplets": len(train_dataset),
        "val_ir_scores": final_scores,
    }
    meta_path = PROJECT_ROOT / "reports" / "raw" / "06_train_run.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2))
    print(f"saved model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
