"""Phase 9: full end-to-end evaluation of the Pipeline over all 120 golden items.

Metrics:
- routing accuracy: Pipeline.answer().route vs the golden type, over all 120 items.
- SQL/HYBRID numeric correctness: same execution-accuracy method as scripts/phase7_sql_eval.py
  (flatten gold vs agent rows, iso3 mapped to name_id, floats rounded to 2 decimals, gold subset
  of agent values), run against gold_sql for every non-abstained SQL/HYBRID item.
- DOC/HYBRID groundedness: two independent signals, since Phase 8 (docs/DECISIONS.md D16) found the
  local LLM judge only reaches 60% agreement with hand labels: (1) judge() v3 grounded rate, and
  (2) fact_coverage, the fraction of gold_answer_facts found verbatim (case-insensitive) in the
  final answer text, which is deterministic and does not depend on the judge model.
- abstain recall: fraction of the 15 ABSTAIN golden items where the pipeline abstained.
- latency: wall-clock per item, p50/p95 overall and per route.
"""
import json
import re
import time

import psycopg

from nusalens.config import PROJECT_ROOT, settings
from nusalens.llm.judge import judge
from nusalens.pipeline import Pipeline

_ISO3 = re.compile(r"^[A-Z]{3}$")


def load_golden() -> list[dict]:
    path = PROJECT_ROOT / "eval" / "golden.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def iso3_to_name(dsn: str) -> dict[str, str]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT iso3, name_id FROM dim_country")
        return dict(cur.fetchall())


def flatten(rows: list, iso3_map: dict[str, str]) -> set:
    flat = set()
    for row in rows:
        for v in row:
            if v is None:
                continue
            if isinstance(v, str) and _ISO3.match(v) and v in iso3_map:
                flat.add(iso3_map[v].strip().lower())
            elif isinstance(v, (int, float)):
                flat.add(round(float(v), 2))
            else:
                flat.add(str(v).strip().lower())
    return flat


def run_gold_sql(sql: str, dsn: str) -> list[tuple]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def chunk_texts(chunk_ids: list[str], dsn: str) -> dict[str, str]:
    if not chunk_ids:
        return {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
        return dict(cur.fetchall())


def fact_coverage(answer: str, facts: list[str]) -> float:
    if not facts:
        return float("nan")
    a = answer.lower()
    hits = sum(1 for f in facts if f.lower() in a)
    return hits / len(facts)


def main() -> None:
    items = load_golden()
    iso3_map = iso3_to_name(settings.db_dsn)
    pipeline = Pipeline()

    per_item = []
    for n, item in enumerate(items, start=1):
        t0 = time.perf_counter()
        try:
            result = pipeline.answer(item["question"])
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"[{n}/{len(items)}] {item['qid']} gold={item['type']} ERROR: {type(exc).__name__}: {exc}")
            per_item.append({
                "qid": item["qid"], "gold_type": item["type"], "predicted_route": None,
                "route_source": None, "routing_correct": False, "abstained": None,
                "is_abstain_item": item["type"] == "ABSTAIN", "abstained_correctly": None,
                "sql_correct": None, "fact_coverage": None, "judge_grounded": None,
                "confidence": None, "latency_ms": latency_ms, "question": item["question"],
                "answer": None, "sql": None, "retrieved_chunk_ids": [], "error": str(exc),
            })
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        routing_correct = result.route == item["type"]

        sql_correct = None
        if item["type"] in ("SQL", "HYBRID") and item.get("gold_sql") and not result.abstained:
            gold_rows = run_gold_sql(item["gold_sql"], settings.db_dsn)
            gold_values = flatten(gold_rows, iso3_map)
            agent_values = flatten(result.sql_rows, iso3_map)
            sql_correct = bool(agent_values) and gold_values.issubset(agent_values)

        judge_grounded = None
        cov = None
        if item["type"] in ("DOC", "HYBRID") and not result.abstained:
            cov = fact_coverage(result.summary, item["gold_answer_facts"])
            texts = chunk_texts([c for c in result.retrieved_chunk_ids], settings.db_dsn)
            context_text = "\n\n".join(f"[{cid}] {texts[cid]}" for cid in result.retrieved_chunk_ids if cid in texts)
            verdict = judge(item["question"], context_text, result.summary, version="v3")
            judge_grounded = verdict.grounded

        is_abstain_item = item["type"] == "ABSTAIN"
        abstained_correctly = is_abstain_item and (result.route == "ABSTAIN" or result.abstained)

        per_item.append({
            "qid": item["qid"], "gold_type": item["type"], "predicted_route": result.route,
            "route_source": result.route_source, "routing_correct": routing_correct,
            "abstained": result.abstained, "is_abstain_item": is_abstain_item,
            "abstained_correctly": abstained_correctly if is_abstain_item else None,
            "sql_correct": sql_correct, "fact_coverage": cov, "judge_grounded": judge_grounded,
            "confidence": result.confidence, "latency_ms": latency_ms,
            "question": item["question"], "answer": result.summary, "sql": result.sql,
            "retrieved_chunk_ids": result.retrieved_chunk_ids,
        })
        print(f"[{n}/{len(items)}] {item['qid']} gold={item['type']} pred={result.route} "
              f"({result.route_source}) {latency_ms:.0f}ms")

    routing_accuracy = sum(1 for p in per_item if p["routing_correct"]) / len(per_item)

    abstain_items = [p for p in per_item if p["is_abstain_item"]]
    abstain_recall = sum(1 for p in abstain_items if p["abstained_correctly"]) / len(abstain_items)

    sql_checked = [p for p in per_item if p["sql_correct"] is not None]
    sql_accuracy = sum(1 for p in sql_checked if p["sql_correct"]) / len(sql_checked) if sql_checked else float("nan")

    doc_checked = [p for p in per_item if p["fact_coverage"] is not None and p["fact_coverage"] == p["fact_coverage"]]
    mean_fact_coverage = sum(p["fact_coverage"] for p in doc_checked) / len(doc_checked) if doc_checked else float("nan")
    judge_checked = [p for p in per_item if p["judge_grounded"] is not None]
    judge_grounded_rate = sum(1 for p in judge_checked if p["judge_grounded"]) / len(judge_checked) if judge_checked else float("nan")

    latencies = sorted(p["latency_ms"] for p in per_item)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    per_route_latency = {}
    for route in ("SQL", "DOC", "HYBRID", "ABSTAIN"):
        lats = sorted(p["latency_ms"] for p in per_item if p["predicted_route"] == route)
        if lats:
            per_route_latency[route] = {"p50_ms": lats[len(lats) // 2], "p95_ms": lats[int(len(lats) * 0.95)], "n": len(lats)}

    summary = {
        "n_items": len(per_item),
        "routing_accuracy": routing_accuracy,
        "abstain_recall": abstain_recall,
        "sql_execution_accuracy": sql_accuracy,
        "mean_fact_coverage_doc_hybrid": mean_fact_coverage,
        "judge_grounded_rate_doc_hybrid": judge_grounded_rate,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "per_route_latency": per_route_latency,
    }
    print(json.dumps(summary, indent=2))

    out_path = PROJECT_ROOT / "reports" / "raw" / "09_e2e_eval.json"
    out_path.write_text(json.dumps({"summary": summary, "per_item": per_item}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
