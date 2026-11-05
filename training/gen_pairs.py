"""Phase 6: generate synthetic Indonesian (question, chunk) pairs with the local LLM."""
import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from nusalens.config import PROJECT_ROOT, settings

FEW_SHOT = """Kamu membuat satu pertanyaan Bahasa Indonesia yang jawabannya benar-benar ada di dalam potongan teks yang diberikan.
Aturan:
- Pertanyaan harus spesifik terhadap isi teks, bukan pertanyaan umum.
- Jangan gunakan frasa seperti "berdasarkan teks di atas", "dokumen ini", "paragraf ini", atau sejenisnya.
- Pertanyaan antara 5 dan 40 kata.
- Keluarkan hanya satu baris pertanyaan, tanpa penjelasan, tanpa tanda kutip.

Contoh 1
Teks: "Gunung Bromo adalah gunung berapi aktif di Jawa Timur dengan ketinggian 2.329 meter di atas permukaan laut, terletak di dalam kaldera besar bernama Laut Pasir."
Pertanyaan: Berapa ketinggian Gunung Bromo di atas permukaan laut?

Contoh 2
Teks: "Bank Indonesia menetapkan suku bunga acuan BI7DRR sebagai instrumen utama kebijakan moneter sejak Agustus 2016, menggantikan BI Rate."
Pertanyaan: Instrumen apa yang menggantikan BI Rate sebagai acuan kebijakan moneter Bank Indonesia?

Sekarang buat satu pertanyaan untuk teks berikut.
Teks: "{chunk_text}"
Pertanyaan:"""


async def generate_one(client: httpx.AsyncClient, chunk_text: str, model: str, endpoint: str) -> str:
    resp = await client.post(
        f"{endpoint}/api/generate",
        json={
            "model": model,
            "prompt": FEW_SHOT.format(chunk_text=chunk_text[:1500]),
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 80},
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    text = resp.json()["response"].strip()
    # Keep the first non-empty line, strip wrapping quotes the model sometimes adds.
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return line.strip('"“” ')


async def worker(queue: asyncio.Queue, client: httpx.AsyncClient, model: str, endpoint: str,
                  out_fh, lock: asyncio.Lock, counters: dict) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        chunk_id, doc_id, text = item
        try:
            question = await generate_one(client, text, model, endpoint)
        except Exception as exc:
            question = ""
            print(f"generation failed for {chunk_id}: {exc}")
        if question:
            record = {"chunk_id": chunk_id, "doc_id": doc_id, "question": question}
            async with lock:
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
                counters["done"] += 1
                if counters["done"] % 200 == 0:
                    print(f"generated {counters['done']}/{counters['total']}")
        queue.task_done()


async def run(chunks: list[tuple], model: str, endpoint: str, out_path: Path, concurrency: int) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for c in chunks:
        queue.put_nowait(c)
    for _ in range(concurrency):
        queue.put_nowait(None)

    lock = asyncio.Lock()
    counters = {"done": 0, "total": len(chunks)}
    with out_path.open("a", encoding="utf-8") as out_fh:
        async with httpx.AsyncClient() as client:
            workers = [
                asyncio.create_task(worker(queue, client, model, endpoint, out_fh, lock, counters))
                for _ in range(concurrency)
            ]
            await queue.join()
            for w in workers:
                await w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # qwen3:4b (the product's configured generator) spends its token budget on
    # chain-of-thought and never emits a clean one-line answer at this budget;
    # qwen2.5:3b-instruct has no reasoning mode and answers directly. Recorded
    # as a training-tooling choice, not a change to the product's generator model.
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / "data" / "silver" / "pairs_raw.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                already.add(json.loads(line)["chunk_id"])
    print(f"{len(already)} chunks already have a generated question, resuming")

    table = pq.read_table(PROJECT_ROOT / "data" / "gold" / "chunks.parquet")
    df = table.to_pandas()
    df = df[df["token_count"] >= 80]

    rng = random.Random(args.seed)
    candidate_ids = sorted(df["chunk_id"].tolist())
    rng.shuffle(candidate_ids)
    selected_ids = candidate_ids[: args.target]
    remaining = [cid for cid in selected_ids if cid not in already]

    id_to_row = df.set_index("chunk_id")
    todo = [(cid, id_to_row.loc[cid, "doc_id"], id_to_row.loc[cid, "text"]) for cid in remaining]
    print(f"{len(todo)} chunks left to generate out of target {args.target}")

    if todo:
        asyncio.run(run(todo, args.model, settings.llm_endpoint, out_path, args.concurrency))

    total_lines = sum(1 for _ in out_path.open(encoding="utf-8"))
    print(f"wrote {out_path} ({total_lines} total pairs)")


if __name__ == "__main__":
    main()
