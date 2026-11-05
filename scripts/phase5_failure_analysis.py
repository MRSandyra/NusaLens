"""Phase 5: classify the ten worst nDCG@10 items of the winning configuration."""
import json

from nusalens.config import PROJECT_ROOT

CAUSE = {
    "doc-007": "entity confusion (Kanada vs Provinsi dan wilayah di Kanada)",
    "hybrid-081": "dual-intent query: indicator clause drags retrieval to indicator-topic chunks",
    "hybrid-084": "dual-intent query: indicator clause drags retrieval to indicator-topic chunks",
    "hybrid-085": "dual-intent query: indicator clause drags retrieval to indicator-topic chunks",
    "hybrid-086": "entity confusion (Johannes Leimena vs Johannes Latuharhary, similar biographies)",
    "hybrid-087": "dual-intent query: indicator clause drags retrieval to indicator-topic chunks",
    "hybrid-090": "weak lexical signal: rare content word, no strong term overlap with gold chunk",
    "hybrid-091": "dual-intent query: indicator clause drags retrieval to indicator-topic chunks",
    "hybrid-092": "near-homonym confusion (Manga vs Marga Minahasa)",
    "hybrid-096": "near-homonym confusion (Purwokerto vs Purwakarta)",
}


def main() -> None:
    result_path = PROJECT_ROOT / "eval" / "results" / "phase5-rrf_rerank_exp.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    worst = sorted(data["per_item"], key=lambda p: p["ndcg_10"])[:10]

    rows = []
    for p in worst:
        cause = CAUSE.get(p["qid"], "uncategorized")
        rows.append({
            "qid": p["qid"],
            "question": p["question"],
            "ndcg_10": p["ndcg_10"],
            "gold_chunk_ids": p["gold_chunk_ids"],
            "top5_retrieved": p["ranked"][:5],
            "cause": cause,
        })

    out_path = PROJECT_ROOT / "reports" / "raw" / "05_failure_analysis.json"
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")

    counts: dict[str, int] = {}
    for r in rows:
        key = r["cause"].split(":")[0].split("(")[0].strip()
        counts[key] = counts.get(key, 0) + 1
    for cause, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{n}x {cause}")


if __name__ == "__main__":
    main()
