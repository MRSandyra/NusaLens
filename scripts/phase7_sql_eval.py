"""Phase 7: execution accuracy of the Text2SQL agent against the 35 SQL golden items."""
import argparse
import json
import re
import time

import psycopg

from nusalens.agents.text2sql import Text2SQLAgent
from nusalens.config import PROJECT_ROOT, settings

_ISO3 = re.compile(r"^[A-Z]{3}$")


def load_sql_golden() -> list[dict]:
    path = PROJECT_ROOT / "eval" / "golden.jsonl"
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in items if r["type"] == "SQL" and r["gold_sql"]]


def _iso3_to_name(dsn: str) -> dict[str, str]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT iso3, name_id FROM dim_country")
        return dict(cur.fetchall())


def _flatten(rows: list[tuple], iso3_map: dict[str, str]) -> set:
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


def run_gold(sql: str, dsn: str) -> list[tuple]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def classify_error(agent_error: str | None, agent_sql: str | None, gold_values: set, agent_values: set) -> str:
    if agent_error:
        if "EXPLAIN" in agent_error or "parse" in agent_error:
            return "invalid_sql"
        return "execution_error"
    if agent_sql is None:
        return "no_sql_generated"
    if not agent_values:
        return "empty_result"
    if not gold_values.issubset(agent_values):
        return "wrong_value"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--tag", default="phase7-v1")
    args = parser.parse_args()

    items = load_sql_golden()
    agent = Text2SQLAgent(prompt_version=args.prompt_version)
    iso3_map = _iso3_to_name(settings.db_dsn)

    records = []
    correct = 0
    latencies = []
    for item in items:
        t0 = time.perf_counter()
        result = agent.ask(item["question"])
        latencies.append((time.perf_counter() - t0) * 1000)

        gold_rows = run_gold(item["gold_sql"], settings.db_dsn)
        gold_values = _flatten(gold_rows, iso3_map)
        agent_values = _flatten(result.rows, iso3_map)

        is_correct = bool(gold_values) and gold_values.issubset(agent_values)
        if is_correct:
            correct += 1

        record = {
            "qid": item["qid"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
            "agent_sql": result.sql,
            "correct": is_correct,
            "attempts": result.attempts,
            "error": result.error,
        }
        if not is_correct:
            record["error_category"] = classify_error(result.error, result.sql, gold_values, agent_values)
            record["gold_values"] = sorted(str(v) for v in gold_values)[:10]
            record["agent_values"] = sorted(str(v) for v in agent_values)[:10]
        records.append(record)
        print(f"{item['qid']}: {'OK' if is_correct else 'FAIL'}")

    accuracy = correct / len(items)
    latencies.sort()

    taxonomy: dict[str, int] = {}
    for r in records:
        if not r["correct"]:
            cat = r["error_category"]
            taxonomy[cat] = taxonomy.get(cat, 0) + 1

    summary = {
        "tag": args.tag,
        "prompt_version": args.prompt_version,
        "n_items": len(items),
        "correct": correct,
        "execution_accuracy": accuracy,
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
        "error_taxonomy": taxonomy,
    }
    print(json.dumps(summary, indent=2))

    out_path = PROJECT_ROOT / "reports" / "raw" / f"07_text2sql_eval_{args.tag}.json"
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
