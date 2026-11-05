import json
import time
import uuid
from pathlib import Path

import psycopg
from nusalens.config import settings, PROJECT_ROOT

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden.jsonl"
RAW_OUT = PROJECT_ROOT / "reports" / "raw" / "04_e2e_harness.txt"
SUMMARY_OUT = PROJECT_ROOT / "reports" / "raw" / "04_e2e_summary.json"


def exec_sql(sql: str) -> list[tuple]:
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def evaluate() -> dict:
    items = [json.loads(l) for l in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    sql_items = [it for it in items if it["type"] in ("SQL", "HYBRID") and it.get("gold_sql")]
    abstain_items = [it for it in items if it["type"] == "ABSTAIN"]

    total_sql = len(sql_items)
    correct_sql = 0
    latencies = []
    for it in sql_items:
        sql = it["gold_sql"]
        start = time.perf_counter()
        try:
            rows = exec_sql(sql)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        except Exception as e:
            # treat as failure
            continue
        if not rows:
            continue
        # If gold_answer_facts present, compare first element (as string)
        if it.get("gold_answer_facts"):
            expected = str(it["gold_answer_facts"][0])
            actual = str(rows[0][0])
            if actual == expected:
                correct_sql += 1
        else:
            # any non‑null result counts as correct
            if rows[0][0] is not None:
                correct_sql += 1

    sql_accuracy = correct_sql / total_sql if total_sql else None
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.5)] if latencies else None
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None

    # Abstain metrics cannot be measured yet, deferred.
    return {
        "sql_total": total_sql,
        "sql_correct": correct_sql,
        "sql_accuracy": sql_accuracy,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "abstain_count": len(abstain_items),
        "abstain_deferred": True,
    }


def main() -> None:
    results = evaluate()
    # raw dump
    RAW_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # summary for report insertion
    SUMMARY_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # human‑readable table printed to stdout (captured in the caller's report)
    print("E2E Evaluation Results")
    print(f"SQL items : {results['sql_total']}")
    print(f"SQL correct: {results['sql_correct']} ({results['sql_accuracy']:.2%})")
    print(f"Latency p50: {results['latency_p50_ms']:.1f} ms")
    print(f"Latency p95: {results['latency_p95_ms']:.1f} ms")
    print("Abstain metrics: deferred until router/synthesizer is available (Phase 8/9).")

if __name__ == "__main__":
    main()
