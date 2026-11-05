"""Phase 8: A/B the router prompt versions on all 120 golden items. Fixed temperature=0."""
import json
import time

from nusalens.agents.router import classify
from nusalens.config import PROJECT_ROOT


def load_golden() -> list[dict]:
    path = PROJECT_ROOT / "eval" / "golden.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(version: str, items: list[dict]) -> dict:
    per_item = []
    latencies = []
    for item in items:
        t0 = time.perf_counter()
        result = classify(item["question"], version=version)
        latencies.append((time.perf_counter() - t0) * 1000)
        per_item.append({
            "qid": item["qid"],
            "gold_type": item["type"],
            "predicted_type": result.type,
            "correct": result.type == item["type"],
        })

    n = len(per_item)
    accuracy = sum(1 for p in per_item if p["correct"]) / n

    tp = sum(1 for p in per_item if p["gold_type"] == "ABSTAIN" and p["predicted_type"] == "ABSTAIN")
    fp = sum(1 for p in per_item if p["gold_type"] != "ABSTAIN" and p["predicted_type"] == "ABSTAIN")
    fn = sum(1 for p in per_item if p["gold_type"] == "ABSTAIN" and p["predicted_type"] != "ABSTAIN")
    abstain_precision = tp / (tp + fp) if (tp + fp) else float("nan")
    abstain_recall = tp / (tp + fn) if (tp + fn) else float("nan")

    latencies.sort()
    return {
        "version": version,
        "n_items": n,
        "accuracy": accuracy,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
        "per_item": per_item,
    }


def main() -> None:
    items = load_golden()
    results = {}
    for version in ["v1", "v2"]:
        print(f"evaluating router {version}...")
        res = evaluate(version, items)
        results[version] = res
        print(f"  accuracy={res['accuracy']:.1%} abstain_precision={res['abstain_precision']:.1%} "
              f"abstain_recall={res['abstain_recall']:.1%} p50={res['latency_p50_ms']:.0f}ms")

    out_path = PROJECT_ROOT / "reports" / "raw" / "08_router_eval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
