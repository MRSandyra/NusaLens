"""Phase 5: bar chart comparing the five retrieval configurations."""
import json

import matplotlib.pyplot as plt

from nusalens.config import PROJECT_ROOT

MODE_LABELS = {
    "dense": "dense",
    "sparse": "sparse",
    "rrf": "rrf",
    "rrf_rerank": "rrf+rerank",
    "rrf_rerank_exp": "rrf+rerank+exp",
}


def main() -> None:
    summary_path = PROJECT_ROOT / "reports" / "raw" / "05_retrieval_summary.json"
    results = json.loads(summary_path.read_text(encoding="utf-8"))

    modes = [r["mode"] for r in results]
    labels = [MODE_LABELS[m] for m in modes]
    recall_1 = [r["recall_at_1"] for r in results]
    recall_10 = [r["recall_at_10"] for r in results]
    mrr_10 = [r["mrr_at_10"] for r in results]
    ndcg_10 = [r["ndcg_at_10"] for r in results]

    x = range(len(modes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 1.5 * width for i in x], recall_1, width, label="Recall@1")
    ax.bar([i - 0.5 * width for i in x], recall_10, width, label="Recall@10")
    ax.bar([i + 0.5 * width for i in x], mrr_10, width, label="MRR@10")
    ax.bar([i + 1.5 * width for i in x], ndcg_10, width, label="nDCG@10")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title("Phase 5 retrieval configuration comparison (70 DOC/HYBRID golden items, k=10)")
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "reports" / "figures" / "05_retrieval_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
