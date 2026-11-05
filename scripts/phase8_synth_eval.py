"""Phase 8: A/B the synthesizer prompt versions (v1/v2/v3) on a stratified sample of the golden set.

Retrieval is run once per question and shared across all three prompt versions, since it does not
depend on the synthesizer prompt; only the synthesis LLM call varies.
"""
import json
import random
import re
import time
from argparse import ArgumentParser
from pathlib import Path

from nusalens.agents.synthesizer import synthesize
from nusalens.config import PROJECT_ROOT, settings
from nusalens.retrieval.hybrid_search import HybridSearcher

ABSTAIN_PHRASES = ["tidak cukup bukti", "tidak dapat menjawab", "tidak ditemukan", "tidak tersedia data"]


def load_golden() -> list[dict]:
    path = PROJECT_ROOT / "eval" / "golden.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def stratified_sample(items: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_type = {"DOC": [], "HYBRID": [], "ABSTAIN": []}
    for it in items:
        if it["type"] in by_type:
            by_type[it["type"]].append(it)
    sample = (
        rng.sample(by_type["DOC"], 20)
        + rng.sample(by_type["HYBRID"], 15)
        + by_type["ABSTAIN"]  # all 15, small and important category
    )
    return sample


def fact_coverage(answer: str, facts: list[str]) -> float:
    if not facts:
        return float("nan")
    a = answer.lower()
    hits = sum(1 for f in facts if f.lower() in a)
    return hits / len(facts)


def predicted_abstain(answer: str, abstained_flag: bool) -> bool:
    if abstained_flag:
        return True
    a = answer.lower().strip()
    return not a or any(p in a for p in ABSTAIN_PHRASES)


def format_compliant(version: str, answer: str, citations: list[str], valid_chunk_ids: set, json_parsed) -> bool:
    if version in {"v1", "v4"}:
        return True
    if version == "v2":
        if not citations:
            return False
        return all(c in valid_chunk_ids for c in citations)
    if version == "v3":
        return bool(json_parsed)
    return False


def parse_args() -> tuple[list[str], Path]:
    parser = ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "raw" / "08_synth_eval.json")
    args = parser.parse_args()
    return args.versions, args.out


def main() -> None:
    versions, out_path = parse_args()
    items = load_golden()
    sample = stratified_sample(items)
    print(f"sample: {len(sample)} items "
          f"({sum(1 for i in sample if i['type']=='DOC')} DOC, "
          f"{sum(1 for i in sample if i['type']=='HYBRID')} HYBRID, "
          f"{sum(1 for i in sample if i['type']=='ABSTAIN')} ABSTAIN)")

    searcher = HybridSearcher(settings.db_dsn, device="cpu")

    cached_chunks = {}
    for item in sample:
        hits = searcher.search_rrf_rerank_exp(item["question"], k=8)
        cached_chunks[item["qid"]] = [{"chunk_id": h.chunk_id, "text": h.text} for h in hits]
        print(f"retrieved for {item['qid']}")

    results = {}
    for version in versions:
        print(f"synthesizing with {version}...")
        per_item = []
        latencies = []
        for item in sample:
            chunks = cached_chunks[item["qid"]]
            valid_ids = {c["chunk_id"] for c in chunks}
            t0 = time.perf_counter()
            r = synthesize(item["question"], chunks, version=version)
            latencies.append((time.perf_counter() - t0) * 1000)

            coverage = fact_coverage(r.answer, item["gold_answer_facts"])
            pred_abstain = predicted_abstain(r.answer, r.abstained)
            compliant = format_compliant(version, r.answer, r.citations, valid_ids, r.json_parsed)

            per_item.append({
                "qid": item["qid"],
                "type": item["type"],
                "fact_coverage": coverage,
                "predicted_abstain": pred_abstain,
                "expected_abstain": item["type"] == "ABSTAIN",
                "format_compliant": compliant,
                "answer_word_count": len(r.answer.split()),
                "answer": r.answer,
            })
        results[version] = {"per_item": per_item, "latencies_ms": latencies}
        print(f"  done, {len(per_item)} items")

    summary = {}
    for version, data in results.items():
        per_item = data["per_item"]
        latencies = sorted(data["latencies_ms"])
        coverages = [p["fact_coverage"] for p in per_item if p["fact_coverage"] == p["fact_coverage"]]  # drop NaN

        tp = sum(1 for p in per_item if p["expected_abstain"] and p["predicted_abstain"])
        fp = sum(1 for p in per_item if not p["expected_abstain"] and p["predicted_abstain"])
        fn = sum(1 for p in per_item if p["expected_abstain"] and not p["predicted_abstain"])

        summary[version] = {
            "n_items": len(per_item),
            "mean_fact_coverage": sum(coverages) / len(coverages) if coverages else float("nan"),
            "abstain_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "abstain_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
            "format_compliance_rate": sum(1 for p in per_item if p["format_compliant"]) / len(per_item),
            "mean_answer_word_count": sum(p["answer_word_count"] for p in per_item) / len(per_item),
            "latency_p50_ms": latencies[len(latencies) // 2],
            "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
        }
        print(f"{version}: {json.dumps(summary[version], indent=2)}")

    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
