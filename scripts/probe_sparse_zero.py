import json
import psycopg
from nusalens.config import PROJECT_ROOT, settings
from nusalens.retrieval.search_dense import DenseSearcher
from nusalens.retrieval.search_sparse import SparseSearcher

golden = [json.loads(l) for l in (PROJECT_ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
doc_items = [r for r in golden if r["type"] in ("DOC", "HYBRID") and r["gold_chunk_ids"]]
item = doc_items[0]

print("question:", item["question"])
print("gold_chunk_ids:", item["gold_chunk_ids"][:3], [type(x).__name__ for x in item["gold_chunk_ids"][:3]])

d = DenseSearcher(settings.db_dsn, device="cpu")
dh = d.search(item["question"], k=5)
print("\ndense hits:", [(h.chunk_id, type(h.chunk_id).__name__) for h in dh[:3]])

s = SparseSearcher(settings.db_dsn)
sh = s.search(item["question"], k=5)
print("sparse hits:", [(h.chunk_id, type(h.chunk_id).__name__) for h in sh[:3]])
print("sparse hit count:", len(sh))

with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
    cur.execute("SELECT websearch_to_tsquery('simple', %s)::text", (item["question"],))
    print("\ntsquery:", cur.fetchone()[0])
    cur.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ websearch_to_tsquery('simple', %s)",
        (item["question"],),
    )
    print("matching rows for full question:", cur.fetchone()[0])
