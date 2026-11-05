"""Phase 6: bar chart of the four conditions plus the Matryoshka truncation curve."""
import json

import matplotlib.pyplot as plt

from nusalens.config import PROJECT_ROOT

COND_LABELS = {
    "stock_e5_small_dense": "stock e5-small\ndense",
    "finetuned_e5_small_dense": "fine-tuned e5-small\ndense",
    "finetuned_e5_small_rrf_rerank": "fine-tuned e5-small\nRRF+rerank",
    "larger_stock_e5_base_dense": "stock e5-base\n(278M) dense",
}


def main() -> None:
    data = json.loads((PROJECT_ROOT / "reports" / "raw" / "06_finetune_eval.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    keys = list(COND_LABELS.keys())
    ndcg = [data[k]["ndcg_at_10"] for k in keys]
    labels = [COND_LABELS[k] for k in keys]
    bars = ax1.bar(labels, ndcg, color=["#888", "#4c72b0", "#2a6b2a", "#aa6622"])
    ax1.set_ylabel("nDCG@10")
    ax1.set_title("Four-condition comparison\n(70 DOC/HYBRID golden items)")
    ax1.set_ylim(0, 0.8)
    for b, v in zip(bars, ndcg):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    mat = data["matryoshka"]
    dims = sorted((int(d) for d in mat.keys()), reverse=True)
    ndcg_by_dim = [mat[str(d)]["ndcg_at_10"] for d in dims]
    saving = [mat[str(d)]["storage_saving_pct"] for d in dims]
    ax2.plot(dims, ndcg_by_dim, marker="o", color="#2a6b2a")
    ax2.set_xlabel("embedding dimension")
    ax2.set_ylabel("nDCG@10")
    ax2.set_title("Matryoshka dimension truncation\n(fine-tuned model, brute-force)")
    ax2.invert_xaxis()
    for d, n, s in zip(dims, ndcg_by_dim, saving):
        ax2.annotate(f"-{s}%", (d, n), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    fig.tight_layout()
    out_path = PROJECT_ROOT / "reports" / "figures" / "06_finetune_gain.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
