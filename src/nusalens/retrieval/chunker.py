import json
import time
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

def chunk_documents(doc_parquet: Path, target_token: int = 450, overlap: int = 80) -> Path:
    table = pq.read_table(doc_parquet)
    df = table.to_pandas()
    chunks = []
    for _, row in df.iterrows():
        text = row["text"]
        # naive token approximation: split on whitespace (real tokenization would use the embedding tokenizer)
        tokens = text.split()
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + target_token]
            chunk_text = " ".join(chunk_tokens)
            if len(chunk_tokens) < 80:
                break
            chunks.append({
                "chunk_id": f"{row['doc_id']}_{i}",
                "doc_id": row["doc_id"],
                "ord": i,
                "heading_path": row.get("title", ""),
                "text": chunk_text,
                "token_count": len(chunk_tokens),
            })
            i += target_token - overlap
    out = doc_parquet.parent / "chunks.parquet"
    pq.write_table(pa.Table.from_pylist(chunks), out)
    return out

if __name__ == "__main__":
    start = time.time()
    out_path = chunk_documents(Path("data/gold/documents.parquet"))
    print(f"Chunked into {out_path} with {pq.ParquetFile(out_path).metadata.num_rows} rows in {time.time() - start:.1f}s")
