"""Phase 6: strict doc_id split (80/10/10), golden-set docs forced into test, leakage verified."""
import json
import random

from nusalens.config import PROJECT_ROOT


def golden_doc_ids() -> set[str]:
    golden_path = PROJECT_ROOT / "eval" / "golden.jsonl"
    items = [json.loads(l) for l in golden_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    doc_ids = set()
    for item in items:
        for cid in item.get("gold_chunk_ids") or []:
            parts = cid.rsplit("_", 1)
            doc_ids.add(parts[0] if len(parts) == 2 and parts[1].isdigit() else cid)
    return doc_ids


def main(seed: int = 42) -> None:
    filtered_path = PROJECT_ROOT / "data" / "silver" / "pairs_filtered.jsonl"
    pairs = [json.loads(l) for l in filtered_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    pool_docs = sorted({p["doc_id"] for p in pairs})
    golden_docs = golden_doc_ids()
    forced_test = sorted(set(pool_docs) & golden_docs)

    rng = random.Random(seed)
    remaining = [d for d in pool_docs if d not in forced_test]
    rng.shuffle(remaining)

    total = len(pool_docs)
    target_test = round(0.10 * total)
    target_val = round(0.10 * total)

    test_docs = set(forced_test)
    need_more_test = max(0, target_test - len(forced_test))
    test_docs |= set(remaining[:need_more_test])
    remaining = remaining[need_more_test:]

    val_docs = set(remaining[:target_val])
    remaining = remaining[target_val:]

    train_docs = set(remaining)

    pool_docs_set = set(pool_docs)
    assert train_docs.isdisjoint(val_docs)
    assert train_docs.isdisjoint(test_docs)
    assert val_docs.isdisjoint(test_docs)
    leaked_golden = (golden_docs & pool_docs_set) - test_docs
    assert not leaked_golden, f"golden docs leaked out of test: {leaked_golden}"

    doc_to_split = {}
    for d in train_docs:
        doc_to_split[d] = "train"
    for d in val_docs:
        doc_to_split[d] = "val"
    for d in test_docs:
        doc_to_split[d] = "test"

    out_path = PROJECT_ROOT / "data" / "silver" / "pairs_split.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            p = dict(p)
            p["split"] = doc_to_split[p["doc_id"]]
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    report = {
        "total_docs": total,
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "test_docs": len(test_docs),
        "forced_golden_docs_in_test": len(forced_test),
        "train_pairs": sum(1 for p in pairs if doc_to_split[p["doc_id"]] == "train"),
        "val_pairs": sum(1 for p in pairs if doc_to_split[p["doc_id"]] == "val"),
        "test_pairs": sum(1 for p in pairs if doc_to_split[p["doc_id"]] == "test"),
        "leakage_check": "PASS: no doc_id appears in more than one split; all golden-set docs present in the pool are in test",
    }
    stats_path = PROJECT_ROOT / "data" / "silver" / "split_stats.json"
    stats_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
