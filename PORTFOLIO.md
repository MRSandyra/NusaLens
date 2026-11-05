# Portfolio: NusaLens

## CV bullet points

- Designed and shipped a hybrid RAG plus text-to-SQL assistant over Indonesian public data, reaching 99.2% routing accuracy and 100% abstain recall on a 120-item evaluation set, fully local with no paid API calls (`reports/09_e2e.md`).
- Fine-tuned a 118M-parameter multilingual embedding model on 5,436 synthetic query-passage pairs with hard-negative mining, improving retrieval nDCG@10 by 65.6% over the stock baseline through the full retrieval pipeline (`reports/06_finetune_embedding.md`).
- Built a layered SQL safety architecture (AST parsing, keyword denylist, cost-based query rejection, read-only database role) around an LLM-driven text-to-SQL agent, reaching 100% execution accuracy with zero bypasses across 20 adversarial security tests (`reports/07_text2sql.md`).
- Exposed the system as an MCP server (6 tools, 3 resources, 2 prompts) verified end-to-end against a real client over stdio, alongside a FastAPI and Streamlit serving layer (`reports/10_mcp.md`, `reports/11_serving.md`).
- Cut cross-encoder reranking latency 1.9x through measured profiling while improving ranking quality, at 70% test coverage across 98 tests and a fully reproducible build pipeline (`reports/13_quality.md`).

## Job requirement to proof mapping

| Requirement | Proof |
|---|---|
| RAG system design and evaluation | `reports/05_hybrid_search.md`, `reports/09_e2e.md` |
| Embedding fine-tuning | `reports/06_finetune_embedding.md`, `models/e5-nusalens-v1/` |
| LLM safety around structured data access (SQL) | `reports/07_text2sql.md`, `src/nusalens/agents/sql_guard.py` |
| Prompt engineering methodology, A/B tested | `reports/08_prompts.md` |
| Production pipeline design, concurrency | `reports/09_e2e.md`, `src/nusalens/pipeline.py` |
| Protocol/tool integration (MCP) | `reports/10_mcp.md` |
| API and UI delivery | `reports/11_serving.md` |
| Cost-benefit engineering judgment | `reports/12_qlora.md` |
| Testing, profiling, hardening discipline | `reports/13_quality.md` |
| Decision documentation and traceability | `docs/DECISIONS.md` (31 entries) |

## Draft application paragraph

Saya membangun NusaLens, sebuah asisten analitik data publik Indonesia yang menggabungkan RAG hybrid dan text-to-SQL di atas indikator Bank Dunia dan artikel Wikipedia Indonesia, seluruhnya berjalan lokal tanpa biaya API berbayar. Sistem ini mencapai akurasi eksekusi SQL 100%, recall abstain 100% pada metrik keamanan paling penting dalam proyek, serta melalui proses fine-tuning model embedding yang terukur meningkatkan kualitas retrieval sebesar 65,6%. Setiap angka yang saya laporkan dapat ditelusuri ke berkas pengukuran mentah, termasuk satu kegagalan yang belum terselesaikan yang saya dokumentasikan secara terbuka beserta analisis akar penyebabnya. Saya tertarik membawa pendekatan ini, mengukur sebelum mengklaim dan melaporkan kegagalan sejujur keberhasilan, ke tim Anda.

## Fifteen likely interview questions, with answers

**1. Why Reciprocal Rank Fusion instead of weighted score fusion?**
Dense (cosine similarity) and sparse (`ts_rank_cd`) scores live on different, uncalibrated scales; a weighted sum needs that calibration to mean anything, and recalibrating it is its own open problem. RRF only uses each system's rank position (`score(d) = sum of 1/(k + rank_i(d))`), so no cross-system score calibration is needed at all. Measured on the golden set (`reports/05_hybrid_search.md`): RRF alone (nDCG@10 0.564) does not beat sparse alone (0.584), since it is diluted by the weaker dense ranking; the full pipeline (RRF plus reranking) reaches 0.630. The simpler mechanism was chosen first, and its actual limitation (dilution by a weak component) was measured, not assumed.

**2. Why fine-tune the embedding model rather than the LLM?**
The embedding model is the cheaper, more measurable lever for this specific weakness: Phase 5's failure analysis showed the stock embedding underperforming BM25 on this golden set's lexically literal DOC template, a retrieval problem, not a generation problem. A 118M-parameter embedding model fine-tunes in 327 seconds at 9 GB peak VRAM on commodity hardware; fine-tuning the 4B generator would cost far more compute for a problem it does not have. QLoRA for the SQL generator was evaluated too and skipped (`reports/12_qlora.md`) once the prompt-only baseline hit 100% execution accuracy, leaving no headroom to fine-tune against.

**3. How is SQL injection through an LLM prevented?**
The LLM's output is never trusted as executable SQL; it passes through five independent layers before anything runs: `sqlglot` AST parsing (reject anything that is not exactly one `SELECT`), a keyword denylist (`INSERT|UPDATE|DELETE|DROP|ALTER|GRANT|REVOKE|TRUNCATE|CREATE|COPY|pg_`), automatic `LIMIT` injection, an `EXPLAIN` cost check against a budget, and execution exclusively through a dedicated read-only Postgres role with a 5 second statement timeout. Measured, not assumed: zero bypasses across 10 adversarial natural-language prompts run through the live agent and 10 direct destructive SQL payloads fed straight to the validator (`reports/07_text2sql.md`), re-proven again at the MCP protocol boundary (`reports/10_mcp.md`).

**4. Why no LangChain?**
The system needed five specific, well-understood pieces (a router, hybrid retrieval with RRF, a guarded SQL executor, a synthesizer, an MCP server), each of which is a few hundred lines of code against a well-documented library (`psycopg`, `sentence-transformers`, the official `mcp` SDK) rather than a novel abstraction problem. A framework earns its cost when it removes real complexity; here it would have added an abstraction layer between the code and exactly the parts that needed to be inspectable and testable (the SQL guard, the abstain gate), for a convenience (chaining calls) the project did not need.

**5. What is MCP, and when does it apply?**
The Model Context Protocol is a JSON-RPC protocol that lets a client discover and call a server's tools, resources, and prompts without the two sides agreeing on anything beyond the protocol itself; a client can point at this server or an unrelated one and get the same `list_tools`/`call_tool` surface. It applies when the goal is letting an external model-driven client (an IDE, a desktop assistant) use the system as a tool, not when building the system's own internal agent logic, which is plain Python here (`src/nusalens/pipeline.py`) and does not go through MCP at all. `reports/10_mcp.md` covers the tools-versus-resources-versus-prompts design decision in detail.

**6. How is RAG quality measured without human annotation?**
Two separate mechanisms, deliberately not relying on each other. First, a deterministic golden set (`eval/golden.jsonl`, 120 items, hash-frozen) built from the real corpus itself: DOC items reference real `gold_chunk_ids`, SQL items carry an executable `gold_sql` verified to actually return a result, so Recall@k/nDCG@k are computed against ground truth that required no manual labeling. Second, an LLM-as-judge for groundedness, explicitly calibrated against 20 hand-labeled transcripts rather than trusted by assumption; the judge only reached 60% agreement with the hand labels across four prompt iterations (`docs/DECISIONS.md` D16), which is reported as a limitation of that specific signal, not papered over, and a deterministic `fact_coverage` substring check is kept as a second, independent signal for exactly this reason.

**7. What would change at 100 times the data volume?**
The HNSW index and RRF fusion scale to millions of rows without redesign; `ef_search` would need retuning (`reports/13_quality.md` shows this corpus is small enough that it currently barely matters, which would stop being true). The bigger change is the router: its fast-rule layer is close to perfect on this project's templated golden set but is measured, not assumed, to generalize poorly to free-form phrasing (the LLM fallback alone scores 62.5%, `docs/DECISIONS.md` D17); 100 times the real-world traffic would surface far more of that fallback path, making router quality the actual bottleneck rather than retrieval scale. Synthesis latency (25 to 50 seconds per LLM call) would also need a queue and horizontal Ollama replicas rather than the single local instance used here.

**8. Why RRF plus a separate reranking step, instead of reranking alone?**
Reranking a cross-encoder over the full corpus is not viable at any scale (quadratic cost against the corpus); RRF cheaply narrows tens of thousands of candidates to a small pool (`k_rerank`, measured and tuned in `reports/13_quality.md`), and the reranker spends its cost only on that pool. The two stages solve different problems: RRF for cheap breadth, reranking for precision on a bounded set.

**9. Why Postgres with pgvector instead of a dedicated vector database?**
One engine serves relational SQL (the indicator data), sparse search (`tsvector`), and vector search (HNSW via pgvector) without a second system to keep in sync, a second connection pool, or a second set of operational concerns, and it is free. The documented fallback (`docs/DECISIONS.md` baseline table) was SQLite with `sqlite-vec` and FTS5, not a dedicated vector database, for the same reason: the project's constraint is "free and simple to run locally," not "scale beyond what one Postgres instance can hold."

**10. How is the abstain gate designed, and why does it matter more than other metrics?**
Abstain recall is explicitly named in the project's own specification as the most important safety metric, because a wrong confident answer is worse than a correct refusal. The gate fires at two points: before any LLM call, when the rule router recognizes a fictional entity or an out-of-range year (`docs/DECISIONS.md` D18, confirmed in ~20 to 36 milliseconds with zero LLM cost); and after retrieval or SQL execution, when the evidence itself is empty or the top rerank score falls below a measured threshold. Measured abstain recall is 100% (15/15) on the golden ABSTAIN set (`reports/09_e2e.md`).

**11. What is the one documented failure in this project, and why report it instead of fixing it first?**
The synthesizer sometimes denies a fact that is present verbatim in its own top-ranked retrieved chunk, found by directly inspecting two failing items against their actual context rather than trusting the aggregate metric (`docs/DECISIONS.md` D15, D19). It is reported, with root-cause analysis and two prior failed prompt iterations, rather than fixed first, because the project's own convention is that a missed target still satisfies its definition of done once two measured remediation attempts and an honest writeup exist; a hidden or silently patched failure would be a worse signal about engineering judgment than a found, explained one.

**12. Why does the judge model only reach 60% agreement with human labels, and what was tried?**
Four prompt iterations were tried, each targeting the specific failure the previous one's raw output revealed: v1's plain boolean, v2 adding an honest-decline exception, v3 reordering the decision logic, v4 replacing the boolean with a 3-way label decoded in Python. Agreement plateaued at 55 to 60% because the underlying issue is not prompt wording: the model's own `reason` text sometimes argues for one verdict while its boolean field states the opposite, a self-contradiction inside one generation that a 3B-parameter instruct model does not reliably avoid (`docs/DECISIONS.md` D16). This is reported as a model-capacity limitation, with the calibration transcripts kept so a larger judge model could be benchmarked against the same 20 items later.

**13. Why run SQL and retrieval concurrently for HYBRID questions, and what does that save?**
A HYBRID question needs both a numeric answer and a document summary, and the two evidence sources are independent: text-to-SQL generation and hybrid retrieval do not depend on each other's output. Running them with `asyncio.gather` over `asyncio.to_thread` instead of sequentially means the wall-clock cost is close to the slower of the two, not their sum; measured HYBRID latency (`reports/09_e2e.md`) is dominated by LLM synthesis either way, but the concurrency removes a real, avoidable serial cost from the SQL and retrieval stages.

**14. What does the testing and reproducibility discipline in this project actually catch, concretely?**
Running the real pipeline end to end for the Phase 13 reproducibility check, not just reading the code, found three real pre-existing bugs (a loader module calling a psycopg2-only method that does not exist in the psycopg v3 actually installed, a missing column population that silently zeroed out sparse search until a manual patch script, a script that re-created an index already identified and removed as a duplicate once before) and one orphaned CLI command that had never been wired into the main app despite being the tool that produced the real ingested corpus (`docs/DECISIONS.md` D24 to D27). None of these were found by code review; all were found by actually executing the thing being reviewed.

**15. What was not completed, and why?**
Listed honestly below, not folded into the results table.

## What was not completed, and why

- **Groundedness below target (68.6%/40.6% against an 80% target).** Root cause identified (synthesizer denies present facts, D15/D19), not fixed; the next concrete step is a milder abstain instruction, logged as future work rather than attempted under this project's time budget.
- **Fine-tuned embedding not wired into serving.** It measurably beats the stock embedding through the full pipeline (0.646 vs 0.630 nDCG@10) but switching the default was deliberately deferred until the groundedness issue above is fixed, so the retrieval gain is not spent on an still-ungrounded answer.
- **QLoRA fine-tuning of the SQL generator skipped.** A deliberate cost-benefit decision (`reports/12_qlora.md`), not a capability gap: the prompt-only baseline already measures 100% execution accuracy on the only SQL evaluation set this project has, leaving no headroom for a fine-tuned adapter to demonstrate against.
- **Satu Data Indonesia (S3) and the BPS web API not integrated.** S3 was scoped out for time; BPS was excluded entirely after confirming it sits behind bot protection, a finding recorded rather than silently worked around (`reports/01_data_inventory.md`).
- **A fresh full end-to-end latency run under Phase 13's new defaults (faster reranking) was not repeated.** The retrieval-level before/after numbers are independently measured and real (`reports/13_quality.md`); the system-level latency figures quoted elsewhere in this portfolio still reflect the prior defaults, stated as such rather than silently updated without a new measurement.

No git command was executed at any point in this project.
