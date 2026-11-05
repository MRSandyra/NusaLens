# Phase 8: Prompt Engineering

Versioned prompt registry (`prompts/<role>/vN.yaml`), A/B tested per role against the frozen 120-item golden set (`eval/golden.jsonl`) or a stratified sample of it. Raw outputs under `reports/raw/08_*.json`.

## Router (classify question as SQL / DOC / HYBRID / ABSTAIN)

Full 120-item golden set, both versions, qwen2.5:3b-instruct (`router_model`).

| Version | Prompt technique | Accuracy | Abstain precision | Abstain recall | Latency p50 |
|---|---|---|---|---|---|
| [v1](../prompts/router/v1.yaml) | plain one-word label | **62.5%** | 22.5% | **60.0%** | 2120 ms |
| [v2](../prompts/router/v2.yaml) | structured JSON `{type, reason}` | 55.0% | 33.3% | 53.3% | 2472 ms |

Winner: **v1**. Structured output (v2) did not help this model classify better and cost more latency. Confusion-matrix analysis (`reports/raw/08_router_eval.json`): both versions' dominant error is the HYBRID class: v1 misclassifies HYBRID as ABSTAIN 16 times, v2 misclassifies SQL as HYBRID 19 times (v2 is "trigger-happy" on HYBRID). v1 chosen on overall accuracy and higher abstain recall, which matters more for safety (missing an abstain-worthy question is worse than one false alarm). Set as `config/settings.yaml` `prompts.router: v1`.

## Text2SQL (natural language to SQL)

Carried over from Phase 7 ([reports/07_text2sql.md](07_text2sql.md)), included here for the full per-role picture.

| Version | Prompt technique | Execution accuracy |
|---|---|---|
| [v1](../prompts/text2sql/v1.yaml) | few-shot + schema view, loose entity snap | 68.6% (24/35), 71.4% (25/35) after an entity-resolution code fix |
| [v2](../prompts/text2sql/v2.yaml) | + explicit repeated rules ("always include `value`", "trend needs `year, value`") | **100% (35/35)** |

Winner: **v2**. Repeating the two rules the model kept missing, instead of stating them once, closed the gap entirely. Set as `prompts.text2sql: v2`.

## Synthesizer (answer from retrieved chunks)

Stratified 50-item sample (20 DOC, 15 HYBRID, all 15 ABSTAIN, seed 42), retrieval cached once per question and shared across versions, qwen3:4b (`generator_model`).

| Version | Prompt technique | Fact coverage | Abstain precision | Abstain recall | Format compliance | Latency p50 |
|---|---|---|---|---|---|---|
| [v1](../prompts/synthesizer/v1.yaml) | plain instruction | **0.533** | 0.0 | 0.0 | 100% (trivial, no format rule) | 25206 ms |
| [v2](../prompts/synthesizer/v2.yaml) | + mandatory `[chunk_id]` citation per sentence | 0.543 | 0.0 | 0.0 | **2%** | 25082 ms |
| [v3](../prompts/synthesizer/v3.yaml) | + abstain clause + anti-speculation + JSON output | 0.038 | 0.311 | 0.933 | 100% | 14058 ms |

Winner: **v1**. v2's only addition (the citation rule) fires correctly only 2% of the time, so it adds complexity for no measured benefit over v1: fact coverage is the same within noise. v3's abstain clause over-corrects catastrophically: `predicted_abstain` is 80% on DOC questions and 100% on HYBRID questions, including plainly-answerable ones (a direct re-run confirmed the model returns `{"answer": "", "abstained": true}` for a question whose answer is verbatim in the retrieved context). Set as `prompts.synthesizer: v1`. See [DECISIONS.md D15](DECISIONS.md) for the full root-cause writeup.

**Open problem, not solved by picking a version**: the project has no working synthesizer abstain mechanism. v1/v2 never abstain (0% recall, including on all 15 fictional-country ABSTAIN questions by this script's phrase-matching heuristic: though manual inspection during judge calibration below shows the model does in fact decline those questions correctly in natural language, just not using the four exact trigger phrases the eval script checks for). v3 abstains on everything. Neither is production-ready; a milder abstain instruction (flag low confidence rather than demand certainty) is the next thing to try, out of scope for this phase.

## LLM-as-judge calibration

20 hand-labeled transcripts (10 DOC, 5 HYBRID, 5 ABSTAIN; `reports/raw/08_judge_calibration_transcripts.json`), each read against its actual retrieved context and labeled grounded/not-grounded by hand, then scored by `judge()` across four prompt iterations.

| Version | Change | Agreement with hand labels |
|---|---|---|
| [v1](../prompts/judge/v1.yaml) | plain grounded boolean | 55% |
| [v2](../prompts/judge/v2.yaml) | + explicit honest-decline exception, claim-by-claim check | 60% |
| [v3](../prompts/judge/v3.yaml) | reordered: decide assert-vs-decline before grounding check | 60% |
| [v4](../prompts/judge/v4.yaml) | 3-way label (`GROUNDED`/`FABRICATED`/`HONEST_DECLINE`) decoded in Python instead of a self-reported boolean | 50% |

**Agreement never reached the 70% target.** Root cause, confirmed by inspecting raw judge output: on every one of the 5 ABSTAIN transcripts under v2/v3, the model's own `reason` text argued the answer should be grounded, while the `grounded` boolean field in the same JSON object still said `false`: a self-contradiction inside one generation, not a prompt-logic gap addressable by rewording instructions. v4's attempt to route around this by asking for a label instead of a boolean made agreement worse and introduced label/score inconsistency. `prompts.judge: v3` is shipped as the best available (tied-best score, more defensible decision procedure), but `judge()` should not be treated as a reliable sole gate: see [DECISIONS.md D16](DECISIONS.md) for the full writeup and the recommendation to cross-check with an algorithmic signal (e.g. the `fact_coverage` substring-match proxy already used above) until a larger judge model is tried.

## Techniques applied and their measured effect

| Technique | Where | Measured effect |
|---|---|---|
| Role prompting (plain instruction) | all v1 prompts | baseline for every role |
| Few-shot examples | text2sql v1/v2 | needed to reach workable accuracy at all (D13); not separately ablated here |
| Repeated/explicit negative instruction | text2sql v2 | 71.4% -> 100%, the single largest measured gain in this phase |
| Structured JSON output | router v2, synthesizer v3, judge v1-v4 | mixed to negative: hurt router accuracy, enabled synthesizer's abstain over-trigger, did not fix judge's boolean self-contradiction |
| Mandatory per-sentence citation tag | synthesizer v2 | requested format honored only 2% of the time; no measured benefit over not asking |
| Explicit abstain + anti-speculation clause | synthesizer v3 | over-corrected to near-total refusal; net harmful |
| Bounded chain-of-thought vs. none | generator model choice (D10/D13/D14) | the dominant lever all phase: `qwen3:4b`'s reasoning mode is fine for open-ended synthesis, unusable for any short/terminal/structured output regardless of prompt wording |

## No hardcoded prompts

`grep -rn "Kamu adalah\|Anda adalah\|You are a" src/` returns no matches outside SQL/string literals unrelated to LLM prompts (an embedding prefix constant and a SQL query, confirmed by inspection). Every LLM prompt lives in `prompts/<role>/vN.yaml`, loaded through `src/nusalens/llm/prompt_registry.py`.

## Defaults set

```yaml
prompts:
  router: v1
  text2sql: v2
  synthesizer: v1
  judge: v3
```
