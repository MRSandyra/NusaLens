import hashlib
import json
from collections import Counter
from pathlib import Path

from nusalens.config import PROJECT_ROOT

golden = PROJECT_ROOT / "eval" / "golden.jsonl"
sha_file = PROJECT_ROOT / "eval" / "golden.sha256"

rows = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
print("rows:", len(rows))
print("distribution:", dict(Counter(r["type"] for r in rows)))

qids = [r["qid"] for r in rows]
print("unique qid:", len(set(qids)) == len(qids))
questions = [r["question"] for r in rows]
print("unique question:", len(set(questions)) == len(questions))

doc_like = [r for r in rows if r["type"] in ("DOC", "HYBRID")]
print("DOC/HYBRID with gold_chunk_ids:", sum(1 for r in doc_like if r["gold_chunk_ids"]), "/", len(doc_like))

sql_like = [r for r in rows if r["type"] in ("SQL", "HYBRID")]
print("SQL/HYBRID with gold_sql:", sum(1 for r in sql_like if r["gold_sql"]), "/", len(sql_like))

abstain = [r for r in rows if r["type"] == "ABSTAIN"]
print("ABSTAIN with no chunk and no sql:",
      sum(1 for r in abstain if not r["gold_chunk_ids"] and not r["gold_sql"]), "/", len(abstain))

with_facts = sum(1 for r in rows if r.get("gold_answer_facts"))
print("items carrying gold_answer_facts:", with_facts)

digest = hashlib.sha256(golden.read_bytes()).hexdigest()
recorded = sha_file.read_text(encoding="utf-8").split()[0]
print("recomputed sha256:", digest)
print("recorded  sha256:", recorded)
print("FROZEN HASH MATCHES:", digest == recorded)
