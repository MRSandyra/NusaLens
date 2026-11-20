"""Summarize the checked-in benchmark artifacts without rerunning models or changing metrics."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_PATH = ROOT / "reports" / "raw" / "05_retrieval_summary.json"
E2E_PATH = ROOT / "reports" / "raw" / "09_e2e_eval.json"
FACT_COVERAGE_TARGET = 0.80


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"required benchmark artifact is missing: {path.relative_to(ROOT)}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-fact-coverage-target", action="store_true",
                        help="exit non-zero when the recorded fact-coverage target is not met")
    args = parser.parse_args()

    retrieval = load_json(RETRIEVAL_PATH)
    e2e = load_json(E2E_PATH)
    serving = next((row for row in retrieval if row.get("mode") == "rrf_rerank_exp"), None)
    if serving is None:
        raise ValueError("retrieval artifact has no rrf_rerank_exp serving configuration")
    summary = e2e.get("summary", {})
    required_e2e = {"routing_accuracy", "abstain_recall", "mean_fact_coverage_doc_hybrid"}
    missing = required_e2e - summary.keys()
    if missing:
        raise ValueError(f"end-to-end artifact is missing fields: {', '.join(sorted(missing))}")

    fact_coverage = float(summary["mean_fact_coverage_doc_hybrid"])
    output = {
        "source": {
            "retrieval": str(RETRIEVAL_PATH.relative_to(ROOT)),
            "end_to_end": str(E2E_PATH.relative_to(ROOT)),
            "note": "Recorded benchmark artifacts; this command does not rerun or improve the benchmark.",
        },
        "retrieval": {
            "mode": serving["mode"],
            "recall_at_10": serving["recall_at_10"],
            "mrr_at_10": serving["mrr_at_10"],
            "ndcg_at_10": serving["ndcg_at_10"],
        },
        "routing": {"accuracy": summary["routing_accuracy"], "n_items": summary.get("n_items")},
        "abstention": {"recall": summary["abstain_recall"]},
        "evidence_coverage": {
            "fact_coverage_doc_hybrid": fact_coverage,
            "target": FACT_COVERAGE_TARGET,
            "target_met": fact_coverage >= FACT_COVERAGE_TARGET,
            "production_gate": "PASS" if fact_coverage >= FACT_COVERAGE_TARGET else "NOT_MET",
        },
    }
    print(json.dumps(output, indent=2))
    return 1 if args.require_fact_coverage_target and fact_coverage < FACT_COVERAGE_TARGET else 0


if __name__ == "__main__":
    raise SystemExit(main())
