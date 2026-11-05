# Phase 12: QLoRA NL-to-SQL (stretch, skipped)

## Decision: skip

Phase 12 is explicitly optional. This is not a resources-forced skip -- the opposite is true: this machine has 16,311 MiB of VRAM (`nvidia-smi`), far above the 6 GB budget the spec was originally written against, and `peft`/`bitsandbytes`/`unsloth` are simply not installed yet, confirming no attempt was made and abandoned. The skip is a cost-benefit decision, made before spending the time, not after a failed attempt.

## Why: the baseline this phase would have to beat is already at its ceiling

Phase 7 measured the prompt-engineered baseline (`qwen2.5:3b-instruct` + `prompts/text2sql/v2.yaml`, no fine-tuning) at **100% execution accuracy (35/35)** on the frozen SQL golden set ([`reports/07_text2sql.md`](07_text2sql.md)). Phase 9's full 120-item end-to-end run reconfirmed the same **100% (35/35)** through the live pipeline ([`reports/09_e2e.md`](09_e2e.md)). There is no accuracy headroom left for a fine-tuned adapter to capture on the only evaluation set this project has: it cannot score above 100%.

Building the ~3,000-pair template-plus-paraphrase dataset, validating every pair by actual execution (per the checklist's own requirement, which takes real wall time against the live database, not just generation time), running QLoRA on `Qwen3-1.7B-Instruct`, and then measuring the comparison is real, non-trivial effort -- realistically several hours between generation, training, and evaluation -- for an outcome that is one of only two possibilities:
1. The adapter also reaches 100% on the 35-item set: no measured accuracy gain, but now the serving path carries an extra adapter to load, version, and maintain.
2. The adapter does worse: synthetic template-plus-paraphrase data at this scale is a plausible source of distribution mismatch against the real golden set's phrasing, and a worse result on a bounded 35-item set is a real risk, not a hypothetical one.

Neither outcome justifies the cost. This is exactly the cost-benefit analysis the checklist itself asks for ("is the accuracy gain worth loading an additional adapter") -- conducted before building anything, using the baseline's own already-measured ceiling, rather than discovered after the fact.

## What would change this decision

If a future phase expanded the SQL golden set with harder items the current prompt-engineered agent measurably fails on (the 35-item set is deliberately templated per Phase 4's construction method, same as the router/DOC/HYBRID sets -- see [DECISIONS.md D17](../docs/DECISIONS.md) on what that does and doesn't prove), QLoRA would have real headroom to demonstrate a gain against, and this decision should be revisited then. Phase 6 already proved the fine-tuning pipeline mechanics work end to end (a full embedding model fine-tune, measured 326.7s wall time, 9009 MB peak VRAM, +65.6% nDCG@10 gain over stock on a case that genuinely had headroom) -- so there is no technical-capability gap to close here, only a currently-absent reason to spend the time.

## DoD

1. This report explains the skip with the VRAM and time figures that justify it. **Met.** VRAM: 16,311 MiB available vs. the original 6 GB budget (not a constraint). Time: the baseline this phase would compare against is already at 100%/100% (Phase 7, Phase 9), measured twice, so the comparison has no accuracy headroom to find regardless of time spent.
2. If executed, peak VRAM and wall time are recorded. N/A -- not executed, per the decision above.
