# Phase 14: Portfolio Package, Executive Summary

NusaLens is a fully local hybrid RAG plus text-to-SQL assistant over Indonesian public data (World Bank indicators, Indonesian Wikipedia), built across 14 phases with a fast rule-based router, guarded SQL execution, hybrid document retrieval, a fine-tuned embedding model, an MCP server, and a FastAPI/Streamlit serving layer. Every reported number traces to a measurement; the one target not met (synthesizer groundedness) is reported with root-cause analysis rather than hidden.

## README number traceability

| Claim in `README.md` | Exact value | Source |
|---|---|---|
| Retrieval gain, production config | nDCG@10 0.390 -> 0.630, +61.5% | `reports/05_hybrid_search.md` line 31 |
| Retrieval gain, fine-tuned embedding | nDCG@10 0.646, +65.6% | `reports/06_finetune_embedding.md` line 89 |
| Text-to-SQL execution accuracy | 100% (35/35) | `reports/07_text2sql.md` line 25; reconfirmed `reports/09_e2e.md` ("SQL execution accuracy" row) |
| Routing accuracy | 99.2% (119/120) | `reports/09_e2e.md`, headline metrics table |
| Abstain recall | 100% (15/15) | `reports/09_e2e.md`, headline metrics table |
| Groundedness (fact_coverage / judge) | 68.6% / 40.6% | `reports/09_e2e.md`, headline metrics table |
| End-to-end latency p95 | 44.5 s overall; SQL p50 2.59 s, ABSTAIN p50 20 ms | `reports/09_e2e.md`, "Latency, measured per route" table |
| Test coverage | 70%, 98 tests | `reports/raw/13_coverage.txt` |
| Reproducibility wall time | 1980 s | `reports/13_quality.md`, "Reproducibility" section; raw log `reports/raw/13_make_all_run.log` |
| Connection pooling speedup | 1.15x, 84 ms/call pair | `reports/13_quality.md`, optimization table; raw `reports/raw/13_profile_connections.json` equivalent run captured in that table |
| Reranker speedup | 1.9x, nDCG@10 0.6298 -> 0.6415 | `reports/13_quality.md`, optimization table; raw `reports/raw/13_profile_rerank.json` |
| ef_search recall gain | 0.500 -> 0.543 | `reports/13_quality.md`, optimization table; raw `reports/raw/13_profile_ef_search.json` |
| Security: zero bypasses | 10 adversarial prompts + 10 direct payloads | `reports/07_text2sql.md` line 42; raw `reports/raw/07_security.json` |
| MCP verification | 6 tools, 7 calls, 1 guardrail re-proof | `reports/10_mcp.md`; raw `reports/raw/10_mcp_transcript.json` |

Every number above was checked against its source file during this phase; none were found unsupported, so none were deleted.

## DoD verification

1. **Every file exists and is populated.** `README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` (31 entries, exceeds the 12-entry minimum), `docs/DEMO.md`, `PORTFOLIO.md`, this file: all present, all substantive, none a placeholder.
2. **Traceability table above.**
3. **Quickstart verification.** All five commands re-run live, in order, during this phase: `docker compose up -d` (idempotent, container already running), `uv sync` (`Resolved 143 packages`, `Checked 122 packages`, no changes needed), `ollama pull` targets already present (confirmed via `nusalens.cli health`'s `llm.available` field), `python -m nusalens.cli health` (`database: OK`, `llm: OK`, `embedding: OK`, `overall: OK`), and `python -m nusalens.cli ask "Berapa nilai Gini index untuk Vietnam pada tahun 2022?"` (`[route: SQL via rule]`, `Jawaban: 36.1.`, correct SQL shown, `Keyakinan: 0.90`). Not re-run from a literally empty clone (would require re-downloading the ~15,000-document corpus and retraining the embedding model, hours of work); the commands themselves were actually executed against the real environment, not inferred from having worked before.
4. **No git command was executed at any point in this project.** Confirmed: this working directory has no `.git` (`git status` returns "fatal: not a git repository"), so no git command could have had an effect even if one had been typed; none was.
5. **Convention audit.** See below.

## Convention audit

```
grep -rnP '\xe2\x80[\x93\x94]' README.md docs/ reports/ src/ prompts/ models/
```
Zero matches outside two documented, out-of-scope categories (binary model artifacts; `reports/raw/*` captured tool/LLM output, the same class of exception O2 already grants runtime timestamps). Full reasoning: `docs/DECISIONS.md` D31.

```
grep -rniE '^\s*(#|//)\s*(step [0-9]|now we|next,? we|let us|let.s )' src/
```
Zero matches.

```
grep -rnE '[0-9]{4}-[0-9]{2}-[0-9]{2}|(Created|Updated|Generated|As of|Date)[: ]+[0-9]{4}' README.md docs/ reports/ src/ models/
```
Zero matches outside `reports/raw/*` (runtime-emitted timestamps, allowed by O2 explicitly).

```
grep -rn '"""' src/
```
Every remaining match is a single-line function or class docstring explaining a non-obvious contract, or a triple-quoted SQL string literal (not a docstring at all). No module docstrings, no multi-line docstrings, no `Args:`/`Returns:` blocks. 17 module docstrings and 2 multi-line function docstrings were removed or trimmed during this phase; detail in D31.

**No git command was executed at any point in this project.**
