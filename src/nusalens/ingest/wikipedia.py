import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .base import sha256_file
from .wikipedia_filters import filter_articles


def fetch_wikipedia(bronze_root, hf_config, target_max):
    """Stream the Hugging Face wikimedia/wikipedia dataset (Indonesian config) and filter articles."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(f"datasets library unavailable: {exc}")
    ds = load_dataset(
        "wikimedia/wikipedia",
        hf_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    kept = []
    scanned = 0
    for art in ds:
        scanned += 1
        title = art.get("title", "")
        text = art.get("text", "")
        url = art.get("url", "")
        filtered = filter_articles([{"title": title, "text": text, "url": url}])
        if filtered:
            kept.append(filtered[0])
        if scanned % 2000 == 0:
            print(f"  wikipedia: scanned {scanned} kept {len(kept)}")
        if len(kept) >= target_max:
            break
    out = bronze_root / "wikipedia_raw.json"
    out.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    return out, scanned, len(kept)


def normalize_wikipedia(bronze_path, silver_path, gold_path):
    raw = json.loads(Path(bronze_path).read_text(encoding="utf-8"))
    rows = []
    for art in raw:
        rows.append({
            "doc_id": f"wikipedia:{art['title'].replace(' ', '_')[:200]}",
            "source": "S2_wikipedia_id",
            "title": art["title"],
            "url": art["url"],
            "lang": "id",
            "published_at": None,
            "license": "CC BY-SA 4.0",
            "text": art["text"],
        })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, silver_path)
    pq.write_table(table, gold_path)
    return len(rows)


def run_wikipedia(bronze_root, gold_root, target_max):
    bronze_path, scanned, kept = fetch_wikipedia(bronze_root, "20231101.id", target_max)
    silver_path = bronze_root / "wikipedia_silver.parquet"
    gold_path = gold_root / "documents.parquet"
    count = normalize_wikipedia(bronze_path, silver_path, gold_path)
    sha = sha256_file(gold_path)
    return gold_path, count, sha, scanned