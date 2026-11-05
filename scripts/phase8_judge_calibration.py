"""Phase 8: generate (question, context, answer) transcripts for manual grounded/not-grounded
labeling, then separately score them with the LLM judge and report agreement.

Step 1 (this script, mode=gen): produce transcripts for a 20-item sample using the production
synthesizer (v1) and write them to reports/raw/08_judge_calibration_transcripts.json for a human
to read and label.

Step 2 (this script, mode=score): read back the same file once the "human_label" field has been
filled in for every item, run the LLM judge on each transcript, and compute agreement.
"""
import json
import random
from argparse import ArgumentParser
from pathlib import Path

from nusalens.agents.synthesizer import synthesize
from nusalens.config import PROJECT_ROOT, settings
from nusalens.llm.judge import judge
from nusalens.retrieval.hybrid_search import HybridSearcher

OUT_PATH = PROJECT_ROOT / "reports" / "raw" / "08_judge_calibration_transcripts.json"


def load_golden() -> list[dict]:
    path = PROJECT_ROOT / "eval" / "golden.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sample_20(items: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_type = {"DOC": [], "HYBRID": [], "ABSTAIN": []}
    for it in items:
        if it["type"] in by_type:
            by_type[it["type"]].append(it)
    return (
        rng.sample(by_type["DOC"], 10)
        + rng.sample(by_type["HYBRID"], 5)
        + rng.sample(by_type["ABSTAIN"], 5)
    )


def gen() -> None:
    items = load_golden()
    sample = sample_20(items)
    searcher = HybridSearcher(settings.db_dsn, device="cpu")

    transcripts = []
    for item in sample:
        hits = searcher.search_rrf_rerank_exp(item["question"], k=8)
        chunks = [{"chunk_id": h.chunk_id, "text": h.text} for h in hits]
        r = synthesize(item["question"], chunks, version="v1")
        transcripts.append({
            "qid": item["qid"],
            "type": item["type"],
            "question": item["question"],
            "gold_answer_facts": item["gold_answer_facts"],
            "context": [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks],
            "answer": r.answer,
            "human_label": None,
            "human_reason": None,
        })
        print(f"generated transcript for {item['qid']} ({item['type']})")

    OUT_PATH.write_text(json.dumps(transcripts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print("fill in human_label (true/false) and human_reason for each item, then rerun with 'score'")


def score(version: str | None, out_path: Path) -> None:
    transcripts = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    missing = [t["qid"] for t in transcripts if t["human_label"] is None]
    if missing:
        print(f"missing human_label for: {missing}")
        sys.exit(1)

    agree = 0
    rows = []
    for t in transcripts:
        context_text = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in t["context"])
        verdict = judge(t["question"], context_text, t["answer"], version=version)
        matched = verdict.grounded == t["human_label"]
        agree += matched
        rows.append({
            "qid": t["qid"], "type": t["type"],
            "human_label": t["human_label"], "judge_grounded": verdict.grounded,
            "judge_score": verdict.score, "judge_reason": verdict.reason, "agree": matched,
        })
        print(f"{t['qid']}: human={t['human_label']} judge={verdict.grounded} "
              f"score={verdict.score} agree={matched}")

    agreement_rate = agree / len(transcripts)
    out = {"agreement_rate": agreement_rate, "n_items": len(transcripts), "rows": rows}
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"agreement_rate={agreement_rate:.1%}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["gen", "score"], default="gen")
    parser.add_argument("--version")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "reports" / "raw" / "08_judge_calibration.json")
    args = parser.parse_args()
    if args.mode == "gen":
        gen()
    else:
        score(args.version, args.out)
