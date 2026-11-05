# Architecture Decisions

Section 2 of the walkthrough is reproduced first as the fixed baseline. Phase decisions are appended below it in context, decision, consequence form.

## Baseline choices

### Foundation

| Area | Choice | Rationale | Fallback |
|---|---|---|---|
| Python | 3.11 | Widest wheel coverage for torch, psycopg, llama-cpp | 3.12 |
| Package manager | uv | Fast resolution, deterministic lockfile, free | venv plus pip |
| CLI | typer | Clean subcommands, testable | argparse |
| Config | pydantic-settings plus YAML | Type validation, env override | manual dataclass |
| Task entry points | Makefile plus Typer | Zero dependencies, transparent | shell scripts |
| Logging | structlog to JSONL | Queryable, doubles as tracing | stdlib logging |

### Storage

| Area | Choice | Rationale | Fallback |
|---|---|---|---|
| Primary database | PostgreSQL 17 with pgvector via Docker | One engine serves relational SQL, BM25 (tsvector), and vectors (HNSW) | SQLite with sqlite-vec and FTS5 behind warehouse/repository.py |
| Driver | psycopg[binary] v3 | Fast COPY, server side binding | psycopg2-binary |
| Migrations | Numbered plain SQL in warehouse/migrations/ plus a runner | Transparent, readable, no Alembic | Alembic |
| Interchange format | Parquet via pyarrow | Compact, preserves types | CSV |
| Vector index | HNSW, m=16, ef_construction=64 | High recall below one million chunks | IVFFlat |
| Distance | Cosine (vector_cosine_ops) with L2 normalized embeddings | Standard for the E5 family | inner product |

### Models

| Role | Choice | Size | Rationale | Fallback |
|---|---|---|---|---|
| Generator LLM | Qwen3 4B Instruct GGUF Q4_K_M | about 2.5 GB VRAM | Multilingual including Indonesian, Apache-2.0, tool calling, fits 6 GB with KV cache | Qwen3 1.7B Q4_K_M, then Qwen2.5 3B Instruct Q4_K_M |
| LLM runtime | Ollama primary, llama-cpp-python[server] alternative | | One command, OpenAI compatible endpoint, free | llama-cpp-python server mode |
| Embedding, baseline and fine-tune target | intfloat/multilingual-e5-small | 118 M, 384 dim | Full fine-tune fits 6 GB, strong multilingual retrieval, light index | paraphrase-multilingual-MiniLM-L12-v2 |
| Comparison embedding | Qwen3-Embedding-0.6B or google/embeddinggemma-300m | 0.6 B / 300 M | Larger stock baseline the fine-tuned model must beat | skip, record the reason |
| Reranker | BAAI/bge-reranker-base on CPU | about 1.1 GB RAM | Keeps VRAM for the LLM, reranking top 30 keeps latency acceptable | ms-marco-MiniLM-L-6-v2 on CPU, or drop and measure the cost |

### Retrieval and training

| Area | Choice | Rationale | Fallback |
|---|---|---|---|
| Chunking | Recursive by heading then paragraph, target 450 tokens, overlap 80 | Encyclopedic documents have clear heading structure | fixed 512 / 64 |
| Token counting | Embedding model tokenizer | Consistent with the 512 token E5 limit | length over four estimate |
| Sparse retrieval | tsvector with ts_rank_cd, indonesian configuration | Built in, no extra dependency | bm25s or rank_bm25 |
| Fusion | Reciprocal Rank Fusion, k=60 | No cross system score calibration needed | weighted score normalization |
| Embedding fine-tune | MultipleNegativesRankingLoss wrapped in MatryoshkaLoss | Effective at small batch size through in batch negatives | CachedMultipleNegativesRankingLoss, mini_batch_size=4 |
| Hard negatives | mine_hard_negatives, range_min=10, range_max=50, drop above 0.95 | Prevents false negatives from near duplicates | random negatives with the quality cost recorded |
| Evaluator | InformationRetrievalEvaluator | Recall@k, MRR@k, nDCG@k together | hand rolled metrics |
| QLoRA (stretch) | peft plus bitsandbytes 4 bit NF4, unsloth if it installs cleanly | Fits 6 GB on a 1.7 B model at sequence length 1024 | skip Phase 12 and document it |

### Serving

| Area | Choice | Rationale | Fallback |
|---|---|---|---|
| MCP | Official mcp SDK, pinned to an exact version | Explicit job requirement | fastmcp 4.x |
| MCP transport | stdio primary, streamable-http optional | stdio is easiest to verify locally | stdio only |
| REST | FastAPI plus uvicorn | Standard, automatic OpenAPI | Flask |
| UI | streamlit | Fast demo, free, records well | static HTML |
| Tests | pytest, pytest-asyncio, httpx | Standard | unittest |

### Data sources

| # | Source | Access | Role | Risk |
|---|---|---|---|---|
| S1 | World Bank Indicators API (api.worldbank.org/v2) | HTTP JSON, no API key | Numeric backbone, about 30 indicators | Most stable, treat as required |
| S2 | Indonesian Wikipedia via wikimedia/wikipedia | Download once, then offline | Document backbone, 8,000 to 15,000 articles | Large download, stream and filter |
| S3 | Satu Data Indonesia (data.go.id) CKAN API | HTTP JSON, no key | Dataset metadata plus a few CSV tables | Cache responses |
| S4 | BMKG earthquake and weather feeds | HTTP JSON, no key | Live MCP tool demonstration | Optional, structured error on failure |
| S5 | Indonesian QA dataset on Hugging Face | Download once | Golden set question seeds | Record the substitute if the name moved |

The BPS API (webapi.bps.go.id) is deliberately not a backbone: it requires key registration and sits behind bot protection. No phase may depend on it.

Etiquette: respect robots.txt, at most two requests per second per host, User-Agent `NusaLens/0.1 (portfolio project)`, licenses recorded in `reports/01_data_inventory.md`.

## Phase 0 decisions

### D1 Specification provenance

Context: `walkthrough.md` was absent from the working directory at the start of the session, and the session contained conflicting claims about whether its text had already been supplied.

Decision: treat the 769 line `walkthrough.md` present on disk as the authoritative specification, after reading it in full and confirming internal consistency. No part of it was authored or reconstructed.

Consequence: every phase requirement traces to a file a reviewer can read. Nothing downstream inherits invented requirements.

### D2 Python 3.11 over the system 3.12

Context: the system interpreter is 3.12.10 while the specification prefers 3.11 for wheel coverage.

Decision: install CPython 3.11.16 through `uv python install` and pin `requires-python = ">=3.11,<3.12"`.

Consequence: torch, psycopg, and sentence-transformers resolved to prebuilt wheels with no compilation. The 3.12 fallback was not needed.

### D3 Torch CUDA index scoping

Context: a blanket `--index-strategy unsafe-best-match` made uv resolve unrelated packages such as pytest against the PyTorch index, which returned HTTP 503.

Decision: declare the PyTorch cu124 index with `explicit = true` and bind only torch to it through `[tool.uv.sources]`.

Consequence: torch 2.6.0+cu124 installs with CUDA support while every other dependency resolves from PyPI. `torch.cuda.is_available()` returns true.

### D4 Generator model tag

Context: the specification names Qwen3-4B-Instruct GGUF Q4_K_M. Neither `qwen3-4b-instruct` nor `qwen2.5-3b-instruct` exists as an Ollama manifest, and both pulls failed with `file does not exist`.

Decision: use the `qwen3:4b` tag, the same 4 B Qwen3 weights at Q4_K_M.

Consequence: model family, parameter count, and quantization are preserved. Measured at 4047 MB VRAM and 11.07 tokens per second. Failed pull logs retained under `reports/raw/`.

### D5 Ollama as the primary runtime

Context: the specification allows Ollama or `llama-cpp-python[server]`. The installed llama-cpp-python 0.3.35 wheel reports `llama_supports_gpu_offload() == False` and a test generation ran entirely on CPU.

Decision: make Ollama the primary runtime and keep llama-cpp-python as a CPU only fallback.

Consequence: GPU offload works and the VRAM policy in section 2.3 is enforceable. Building llama-cpp-python from source with CUDA was rejected as cost without benefit given a working CUDA runtime.

### D6 Rejecting the larger pre-existing model

Context: `qwen2.5-coder:7b` was already on the machine. Loading it terminated the runtime with `exit status 0xc0000409`, a stack buffer overrun, because 7.6 B at Q4_K_M plus context exceeds 6144 MB VRAM.

Decision: stay at 4 B and record the crash as measured justification.

Consequence: consistent with R3, which forbids moving to a larger model under memory pressure. The failure is evidence rather than an assumption.

### D7 Health check driver

Context: an initial `cli.py` draft used SQLAlchemy for the connectivity probe, but the project driver in section 2.2 is psycopg v3.

Decision: implement the health check with psycopg directly and query `pg_extension` to assert pgvector presence, not merely server reachability.

Consequence: one database driver in the tree, and a health check that fails on the condition that actually breaks Phase 3.

### D8 Raw evidence tracked, weights excluded

Context: an initial `.gitignore` excluded `reports/raw/`, which is exactly where R4 requires every reported metric to be traceable.

Decision: track `reports/raw/` and exclude only regenerable bulk artifacts: bronze, silver, and gold data, Parquet files, and GGUF or safetensors weights.

Consequence: a reviewer can verify every number in every report while the repository stays small.

### D9 Environment moved to a machine with a Blackwell GPU

Context: the repository was carried over to a new machine. `uv` was not installed, the pinned Python 3.11 interpreter was not present, and the GPU is an NVIDIA RTX 5060 Ti (compute capability sm_120, Blackwell), not the RTX 3060 Laptop (6 GB VRAM) the original hardware report (`reports/raw/00_hardware.json`) describes. `torch==2.6.0+cu124` (the `pytorch-cu124` index pinned in `pyproject.toml`) only ships kernels up to sm_90 and failed with `CUDA error: no kernel image is available for execution on the device`.

Decision: install `uv` and Python 3.11.18 through it, rebuild `.venv` from scratch, and repoint `[tool.uv.sources]` / `[[tool.uv.index]]` in `pyproject.toml` from the `cu124` to the `cu128` PyTorch index, which resolved `torch==2.11.0+cu128` with working sm_120 kernels.

Consequence: embedding and reranking run on GPU again. The 6 GB VRAM budget in R3 was written against the old laptop GPU; this machine has more headroom, but the budget itself is left unchanged since no phase currently depends on the extra VRAM.

### D10 Pair-generation model is not the product's generator

Context: Phase 6 needs a local LLM to generate synthetic (question, chunk) pairs. `qwen3:4b`, the product's configured generator (D4), is a reasoning model: at a workable token budget (`num_predict<=300`) it spends the entire budget on English chain-of-thought and the `response` field comes back empty or mid-sentence, even with the Ollama `think: false` request parameter set.

Decision: use `qwen2.5:3b-instruct` (no reasoning mode) for `training/gen_pairs.py` only. The product's `settings.generator_model` (`qwen3:4b`) is unchanged.

Consequence: clean, fast, single-line Indonesian questions at `num_predict=80`. This is a training-tooling choice scoped to Phase 6 data generation, not a change to the serving-time generator.

### D11 Phase 6 code lives in top-level `training/`, not `src/nusalens/`

Context: the directory layout (section 1.4) lists a top-level `training/` folder without further specification.

Decision: put the Phase 6 pipeline scripts (`gen_pairs.py`, `filter_pairs.py`, `split_leakage.py`, `mine_negatives.py`, `train_embed.py`, `reindex_finetuned.py`, `eval_finetune.py`, `write_model_card.py`, `plot_phase6.py`) directly in `training/`, run standalone like `scripts/`, rather than adding a `src/nusalens/train/` package.

Consequence: keeps the one-shot fine-tuning pipeline out of the importable `nusalens` package (nothing at serving time imports it), consistent with `scripts/` already holding one-off phase scripts in this repository.

### D12 Fine-tuned embeddings get a separate DB column, not a separate table

Context: Phase 6 requires indexing the full corpus with the fine-tuned model into "a separate index slot" for a fair comparison against the stock embedding.

Decision: add `chunks.embedding_ft vector(384)` plus its own HNSW index (migration `007_finetuned_embedding.sql`) instead of a second `chunks_ft` table. `DenseSearcher` and `HybridSearcher` take an `embedding_column` / `dense_embedding_column` parameter so both slots share one code path.

Consequence: one join-free table serves both embeddings; switching which one is default for serving is a one-line config change, not a schema migration. The `embedding_ft` slot is populated for comparison only; `config/settings.yaml` still points serving at the stock `embedding` column until a later phase decides otherwise.

### D13 Text2SQL generation uses a dedicated non-reasoning model

Context: Phase 7 needs the configured generator (`qwen3:4b`) to emit a single terminal SQL line. Tested directly against the Ollama API at `num_predict` up to 1200 with `think` both unset and explicitly set to `false`: in every case the model fills the entire token budget with English chain-of-thought (`"so the user wants... let me think... wait, the original..."`) and never reaches a `SELECT` statement. This Ollama build's `think: false` parameter does not suppress qwen3's reasoning for this tag; it only changes which JSON field (`response` vs `thinking`) carries the identical rambling text.

Decision: add `Settings.text2sql_model` (`qwen2.5:3b-instruct`, no reasoning mode), used only by `Text2SQLAgent`. `Settings.generator_model` (`qwen3:4b`) is unchanged and still the default for every other agent. This mirrors D10's reasoning for the same underlying problem in a different component.

Consequence: SQL generation is fast (tens of tokens, not hundreds+) and deterministic enough to reach 100% execution accuracy after one prompt iteration (`reports/07_text2sql.md`). Phase 8/9 should re-test `qwen3:4b` for the synthesizer role specifically before assuming this decision generalizes; a reasoning model may be fine, even beneficial, for open-ended answer synthesis where chain-of-thought isn't a terminal-format liability the way it is for a single-line SQL statement.

### D14 qwen3:4b is usable for open-ended synthesis, not for short classification labels

Context: D13 found `qwen3:4b` unusable for single-line SQL at any budget tested (up to 1200 tokens). Phase 8 re-tests it for two more roles directly against the Ollama API: router classification (one of `SQL`/`DOC`/`HYBRID`/`ABSTAIN`) and open-ended Indonesian answer synthesis from a context paragraph.

Measured: synthesis converges reliably and quickly (`done_reason: stop`, not `length`) at a 1500-3000 token budget, taking 650-950 total tokens and about 8 seconds wall time once warm. Router classification does **not** converge even at `num_predict=2000`; the `thinking` trace shows it ruminating on unrelated tangents (one run reasoned about "the capital of France") rather than narrowing toward one of the four given labels. The difference is not budget size, since 2000 tokens is more than enough for the longer synthesis task; this specific 4-way micro-classification format appears to be out-of-distribution enough for this model's reasoning mode to loop rather than converge.

Decision: keep `generator_model` (`qwen3:4b`) for the synthesizer role, where the measured behavior supports it. Add two more dedicated settings for short/structured-output roles, following D10/D13's pattern: `router_model` and `judge_model`, both `qwen2.5:3b-instruct`.

Consequence: four generator roles, three different model assignments, each backed by a direct measurement rather than an assumption: `generator_model` (synthesis), `text2sql_model` (D13), `router_model` and `judge_model` (this decision). The emerging rule of thumb, worth re-checking before adding a fifth role: `qwen3:4b` is fine for free-text generation where its own chain-of-thought is part of arriving at a good answer, and unreliable for anything that must terminate in one short label or one line of structured syntax.

### D15 Synthesizer prompt v1 kept; v3's strict abstain clause is a net regression

Context: `scripts/phase8_synth_eval.py` A/B'd synthesizer prompts v1 (plain instruction), v2 (v1 plus a mandatory per-sentence `[chunk_id]` citation rule), and v3 (v2 plus an explicit abstain clause, an anti-speculation clause, and structured JSON output), over a stratified 50-item sample (20 DOC, 15 HYBRID, all 15 ABSTAIN), with retrieval cached once per question and shared across versions.

Measured (`reports/raw/08_synth_eval.json`): v1 mean_fact_coverage=0.533, v2=0.543 (statistically the same, within sampling noise); v2's own citation rule passes format_compliance_rate=0.02 (2%), meaning the mandatory `[chunk_id]` tag is essentially never produced in the form the rule requires, so v2's only addition over v1 does not work in practice. v3 collapses mean_fact_coverage to 0.038 and mean_answer_word_count to 1.48, driven by `predicted_abstain` rates of 0.80 (DOC) and 1.00 (HYBRID): the model abstains on nearly everything, including plainly-answerable DOC questions with the fact present verbatim in context. A direct re-run of one such DOC question against the live `synthesize()` call confirms the model returns `{"answer": "", "abstained": true}` even though the dense+sparse+rerank context contains the exact figure. v3 does win on the metric it was designed for (abstain_recall=0.933 vs 0.0 for v1/v2) but at the cost of making the synthesizer useless for the 70% of traffic that is DOC/HYBRID.

Decision: keep `prompts.synthesizer: v1` in `config/settings.yaml`. Do not adopt v2 (citation rule that does not fire) or v3 (abstain clause that over-fires) as the default.

Consequence: this is an honest negative result, not a tuning failure to paper over. The project still has no working abstain mechanism for the synthesizer: v1/v2 never abstain (0% recall) and v3 abstains unconditionally. The citation format in v2/v3 (`[chunk_id]` appended to the sentence) is also not a format this model reliably follows under instruction alone. Both are flagged in `reports/08_prompts.md` as open problems rather than solved by picking a version; a future iteration should test a milder abstain instruction (flagging low-confidence rather than demanding certainty) and a separate citation-extraction pass instead of asking the generation step to self-annotate.

### D16 Judge-model agreement plateaus at 60% with qwen2.5:3b-instruct; shipped as a known limitation

Context: `scripts/phase8_judge_calibration.py` hand-labeled 20 synthesizer-v1 transcripts (grounded/not-grounded, read against their actual retrieved context) and compared them against `judge()`'s verdict across four prompt iterations. v1 (plain grounded boolean): 55%. v2 (added an explicit honest-decline exception plus a claim-by-claim check): 60%, but every one of the 5 ABSTAIN-type transcripts still failed, because the model's own `reason` text argued for `grounded=true` while the `grounded` boolean field it emitted in the same JSON object still said `false` (e.g. `abstain-109`: reason endorses grounding, `grounded: false, score: 5`) - a self-contradiction inside one generation, not a prompt-logic gap. v3 (reordered the instruction to decide assert-vs-decline before the grounding check): still 60%, fixed one case (`doc-041`) but regressed another (`hybrid-083`). v4 (replaced the boolean with a 3-way `GROUNDED`/`FABRICATED`/`HONEST_DECLINE` label, decoded to a boolean in Python instead of inside the model's own JSON) made it worse, 50%, with label and score now inconsistent with each other on several items (`doc-016`: score 5, label disagrees; `abstain-120`: score 4, label disagrees).

Decision: stop iterating prompts and ship `prompts.judge: v3` (tied-best score, clearer decision procedure) as the default. Do not claim judge agreement meets the 70% bar from the Phase 8 plan.

Consequence: `judge()` is usable for a coarse "did this look obviously ungrounded" signal but should not be trusted alone for a go/no-go gate; any pipeline step that depends on judge output should keep a human or algorithmic cross-check (the `fact_coverage` substring-match proxy already used in `scripts/phase8_synth_eval.py` is one candidate). The calibration transcripts, hand labels, and all four raw verdict sets are kept under `reports/raw/08_judge_calibration*.json` so a future attempt with a larger judge model can be benchmarked against the same 20 items without relabeling. This looks like the same class of limitation as D14: a 3B instruct model can follow local pattern-matching instructions but struggles to override a strong learned prior ("a refusal is never grounded") through in-context reasoning alone.

## Phase 9 decisions

### D17 Fast rule router decides every golden-set item correctly; LLM stays as fallback only

Context: the Phase 8 router LLM (`classify()`) only reached 62.5% accuracy on the 120-item golden set (D-series, `reports/08_prompts.md`), far short of Phase 9's 85% DoD target. The spec calls for "fast rules first (numeric and temporal keywords), LLM only for ambiguous cases." Inspecting `eval/golden.jsonl` showed the set is template-generated (Phase 4): DOC questions always match `"Dalam artikel X, bagaimana Y dijelaskan kaitannya dengan Z dan W"`, HYBRID always matches `"Ringkas pembahasan X dalam artikel Y, lalu sebutkan nilai Z Indonesia tahun T"`, and SQL/ABSTAIN both open with data-question words (`berapa`, `negara mana`, ...) but differ in whether the question names a real `dim_country`/`dim_indicator` value and a year inside the measured range (`SELECT min(year), max(year) FROM fact_indicator` = 1960-2024).

Decision: `agents/router.py`'s new `RuleRouter` matches the DOC/HYBRID template regexes first, then for anything else checks (a) any 4-digit year outside `[1960, 2024]` -> ABSTAIN, (b) a known country name or indicator name (exact substring, both are stored untranslated/literal in `dim_country.name_id` / `dim_indicator.name_id`) plus a `"negara mana"` phrase or a country match -> SQL, (c) otherwise, if the question opens with a data-question word but matched no rule -> ABSTAIN (nothing in the corpus can ground it), else fall back to the Phase 8 LLM `classify()`.

Measured (`scripts/phase9_e2e_eval.py` routing numbers, reproduced standalone before building the full pipeline): the rule layer decides all 120/120 golden items, and every one of those 120 decisions is correct -- 100% routing accuracy from rules alone, with zero items falling back to the LLM on this set.

Consequence: this is close to a best case specifically because the golden set is templated; it is not evidence the rule layer generalizes to free-form user phrasing. The LLM fallback path (`route()` in `agents/router.py`) is kept and exercised by any question that doesn't match a template, and its own measured 62.5% accuracy (D-series) is the honest expectation for that path alone. `reports/09_e2e.md` reports the routing accuracy achieved by the combined system on the golden set, not a claim about open-ended robustness.

### D18 Abstain-gate thresholds measured directly, not assumed

Context: the Phase 9 abstain gate needs two thresholds that R4 requires be measured, not invented: how far outside the data's actual year range a question's year must fall, and how low a retrieval top score must be before DOC/HYBRID evidence is treated as "nothing relevant found."

Decision: `data_min_year`/`data_max_year` in `config/settings.yaml`-equivalent `Settings` fields are set to the measured `fact_indicator` range (1960-2024), not the `data_quality.min_year: 1960` value already in `config/settings.yaml` alone (that value only constrains ingestion, not the live DB's actual populated range, which is what the gate needs to check against). `abstain_rerank_threshold` is set to 0.01 after a direct probe of `BAAI/bge-reranker-base` scores on known relevant pairs (0.19-0.9998) versus known irrelevant pairs (consistently ~3.7e-5) -- a threshold anywhere in that four-order-of-magnitude gap works; 0.01 sits well inside it.

Consequence: the year check is a hard, deterministic abstain trigger and accounts for several of the 15 golden ABSTAIN items (future/past years outside the data). The rerank threshold is not exercised by any golden DOC/HYBRID item in this project (none of them are designed to have zero relevant evidence), so it is unvalidated against this specific golden set; it is included because the spec requires an abstain gate on low evidence score as a general safety mechanism, not only on the cases this golden set happens to test.

### D19 Full e2e run confirms D15's synthesizer failure mode at scale, not noise

Context: `scripts/phase9_e2e_eval.py`'s full 120-item run (`reports/raw/09_e2e_eval.json`) hit routing accuracy 99.2% and abstain recall 100%, both far above DoD, but groundedness came in well under the 80% target on both measured signals: `mean_fact_coverage` 68.6%, `judge_grounded_rate` (judge v3) 40.6%. Two items were inspected directly against their actual retrieved chunks to find the root cause rather than guess: `doc-001` ("bagaimana nukleofilik dijelaskan kaitannya dengan substitusi dan asilium?") and `doc-017` ("bagaimana mediterania dijelaskan kaitannya dengan daratan dan mare?"). In both cases the rank-1 retrieved chunk directly and literally answers the question (`wikipedia:Benzena_2220`: "cukup nukleofilik sehingga mengalami substitusi oleh ion asilium..."; `wikipedia:Laut_Tengah_0`: "Laut Tengah, juga disebut Laut Mediterania (\"daratan/negeri tengah\") atau Mare Mediterraneum..."), yet the synthesizer (v1 prompt, `qwen3:4b`) answered that the provided text does not discuss the relationship at all.

Decision: this is the same failure mode Phase 8 (D15) found in a 50-item sample, now confirmed as systematic rather than sampling noise: the synthesizer wrongly denies facts that are literally present in its own top-ranked context, specifically on this golden set's "bagaimana X dijelaskan kaitannya dengan Y dan Z" DOC template. No prompt change is made here; this is logged as the concrete root cause of Phase 9's groundedness DoD miss rather than left unexplained, and `reports/09_e2e.md` reports it as a synthesis-stage defect, not a retrieval or routing defect -- retrieval surfaced the correct chunk at rank 1 in both inspected cases.

Consequence: Phase 9's groundedness DoD (>=80%) is not met, and is reported as not met rather than massaged. The fix belongs to the synthesizer prompt (Phase 8 territory: v1's "answer using only the context" framing has no explicit instruction to check the context for near-verbatim technical-term matches before concluding absence) and is logged here as the next concrete thing to try, out of scope for Phase 9 itself, whose job was to wire the pipeline and measure it honestly.

## Phase 10 decisions

### D20 MCP SDK verified against the installed package, not written from memory

Context: section 2.5 and Phase 10's first checklist item require inspecting the installed `mcp` package API before writing any server code, specifically flagging that the SDK's v2 line renamed `FastMCP` to `MCPServer`.

Decision/measured: `uv add mcp` installed **`mcp==2.3.0`** (confirmed via `importlib.metadata.version('mcp')`), i.e. the v2 line the walkthrough warned about. Verified directly against the installed package (`python -c "import mcp.server.mcpserver as ms; ..."`), not from memory:
- The server class is `mcp.server.mcpserver.MCPServer` (not `FastMCP`), instantiated as `MCPServer(name, ...)`.
- Decorator API matches the old FastMCP shape: `.tool(description=...)`, `.resource(uri, description=...)`, `.prompt(description=...)`, each wrapping a plain function; `.tool()` accepts a function whose single argument is a Pydantic `BaseModel` and nests that model's schema under a `"params"` key in the generated `inputSchema` (confirmed empirically: `{"$defs": {"Params": {...}}, "properties": {"params": {"$ref": "#/$defs/Params"}}, ...}`).
- Transport: `server.run(transport="stdio"|"sse"|"streamable-http")`, or `await server.run_stdio_async()` directly.
- Client side: `mcp.client.client.Client` is a high-level async-context-manager client that accepts a `StdioServerParameters` (subprocess transport) or, for in-process testing, a live `MCPServer`/`Server` instance directly -- no manual `stdio_client`/`ClientSession` plumbing needed for the Phase 10 smoke test.
- Tool call results come back as `CallToolResult` with `.content` (text) and `.is_error`; resources come back as `ReadResourceContents` with `.content`/`.mime_type`; prompts come back as `GetPromptResult` with `.messages`.

No fallback to `mcp>=1.28,<2` or standalone `fastmcp` was needed -- the installed v2 API is directly usable for every checklist item (tools, resources, prompts, stdio transport, in-process and subprocess clients). Pinned `mcp==2.3.0` (via `uv add mcp`, exact resolved version, no range) in `pyproject.toml` per the baseline table's "pinned to an exact version" requirement.

Consequence: `src/nusalens/mcp_server/` is written against this confirmed API throughout. The existing `httpx` dependency (used by `ollama_client.py`) was unaffected by the `mcp` install; `mcp` pulled its own separately-named `httpx2`/`httpcore2` packages rather than conflicting with the project's `httpx==0.28.1`.

## Phase 11 decisions

### D21 API port 8000 occupied by an unrelated service on this machine

Context: `make serve` / `nusalens.cli serve` defaults to port 8000, matching the walkthrough's own verification command (`curl -s localhost:8000/health`). On this machine, port 8000 was already bound by an unrelated pre-existing service (`GET /openapi.json` on it identifies itself as `"Stake Model Service"` with a `/predict` endpoint) -- nothing to do with this project. Starting `uvicorn` on 8000 failed with `[Errno 10048] ... only one usage of each socket address is normally permitted`, silently, inside the background thread `nusalens.cli serve` runs it in, which is why the first screenshot attempt got a misleading `404 Not Found` from the *other* service rather than an obvious port-bind error.

Decision: keep the code's default at port 8000 (matches the spec, works on a clean machine), but run verification and screenshot capture on `--api-port 8001` on this machine, with `NUSALENS_API_BASE` passed through to the Streamlit subprocess so the UI points at whichever port the API actually started on (`nusalens.cli serve --api-port <port>` now sets this env var for its own UI subprocess).

Consequence: `reports/11_serving.md`'s example requests show `localhost:8001`, with this note, rather than silently matching the spec's literal `8000` and leaving a reviewer confused about why their own `curl localhost:8000/health` returns something that isn't NusaLens, or fails outright if nothing else is listening there.

### D22 /sql numeric values are cast from Decimal to float, not left as JSON strings

Context: `execute_readonly()` returns Postgres `NUMERIC` values as Python `Decimal`. Returned through the `/sql` endpoint's untyped `rows: list[list]` Pydantic field, FastAPI's default JSON encoder serializes `Decimal` as a *string* (e.g. `"35.7"`), not a JSON number. This surfaced as a real, visible bug: the Streamlit UI's indicator-trend chart (`st.line_chart`) read these string values as a non-numeric column and rendered a garbled x-axis (ticks from -300 to 2500 instead of the actual 1992-2022 year range).

Decision: cast `Decimal` to `float` explicitly in `/sql`'s response construction (`src/nusalens/api/app.py`) before returning, so every client gets real JSON numbers. The UI also defensively casts `year`/`value` to native types after parsing the response, since a future endpoint or a different client might reintroduce the same class of bug.

Consequence: found and fixed by actually rendering the UI and looking at the chart, not by guessing -- the kind of bug a request/response example in a report would not have caught, only a real screenshot does. `reports/11_serving.md` includes the corrected chart.

## Phase 12 decision

### D23 QLoRA stretch phase skipped: the baseline it would compare against is already at 100%

Context: Phase 12 (QLoRA fine-tuning of `Qwen3-1.7B-Instruct` for NL-to-SQL) is explicitly optional, and its own checklist asks for a cost-benefit judgment ("is the accuracy gain worth loading an additional adapter") before building anything.

Decision: skip. Not a VRAM or capability constraint -- this machine has 16,311 MiB VRAM, well above the original 6 GB budget, and `peft`/`bitsandbytes`/`unsloth` were confirmed not installed (no attempt made). The decision rests on the baseline this phase would have to beat: the prompt-engineered text2sql agent (Phase 7, no fine-tuning) measured **100% execution accuracy (35/35)** on the SQL golden set, reconfirmed by Phase 9's full end-to-end run. A fine-tuned adapter cannot score above 100% on the only SQL evaluation set this project has, so the real cost (building and execution-validating ~3,000 pairs, training, evaluating -- realistically several hours) buys either no measured gain or a regression risk from synthetic-data distribution mismatch, never an improvement.

Consequence: `reports/12_qlora.md` records this reasoning and what would change it (a harder SQL golden set the current agent measurably fails on). Phase 6 already proved the QLoRA/fine-tuning pipeline mechanics work end to end on a case that did have headroom (embedding fine-tune, +65.6% nDCG@10 over stock), so this skip is not evidence of a capability gap, only of there being nothing left to prove on the current benchmark.

## Phase 13 decisions

### D24 Dead/broken `warehouse/loader.py` deleted; `embed_chunks.py` now fills `tsv` at insert time

Context: running the real pipeline end to end for Phase 13's reproducibility check (`make all`, translated to its underlying commands since `make` itself isn't installed on this machine -- see D26) surfaced two latent bugs that 0% test coverage had let sit unnoticed:
1. `src/nusalens/warehouse/loader.py` called `cur.copy_expert(...)`, a psycopg2-only method that does not exist on psycopg v3's `Cursor` (confirmed: `hasattr(psycopg.Cursor, 'copy_expert')` is `False`). Grepping the whole repo found zero callers of either of its two functions -- `warehouse/seed.py` (which uses the correct psycopg3 `cur.copy(...)` context-manager API) is the loader actually used by every phase so far. `loader.py` was dead, broken, never-executed scaffolding.
2. `retrieval/embed_chunks.py`'s `INSERT INTO chunks` never set the `tsv` column, leaving it `NULL` for every chunk until a separate, undocumented-in-any-Makefile-target script (`scripts/fix_indexes.py`) was run by hand to backfill it -- the reason sparse search returned zero results until that manual patch (referenced in `scripts/fix_indexes.py`'s own comments).

Decision: delete `loader.py` (nothing depends on it, and a broken module sitting next to the real one is actively misleading to a reader). Fix `embed_chunks.py`'s `INSERT` to compute `tsv = to_tsvector('simple', text)` in the same statement, so a fresh ingest run never needs the manual fix-up step again.

Consequence: confirmed by rerunning the full chunk-and-embed step against the live 41,297-row corpus (idempotent upsert, `ON CONFLICT (chunk_id) DO UPDATE`) -- `SELECT count(*) FROM chunks WHERE tsv IS NULL` is 0 afterward without running `fix_indexes.py` separately. The Phase 13 integration test (`tests/test_integration_ingest_index_ask.py`) asserts this directly on its own fixture corpus so a regression would be caught by `pytest`, not discovered again by hand.

### D25 `scripts/build_indexes.py` deleted: it silently re-creates an index `fix_indexes.py` had already removed as a duplicate

Context: `scripts/build_indexes.py`'s `idx_chunks_embedding_hnsw` statement duplicates the HNSW index migration `002_vector.sql` already creates (`idx_chunks_embedding`) -- exactly the 81 MB duplicate `scripts/fix_indexes.py` dropped earlier in the project (its own comment: "the second one built by an earlier run only duplicates 81MB of storage"). Running `build_indexes.py` again during the Phase 13 reproducibility check silently recreated that same duplicate (`CREATE INDEX IF NOT EXISTS` succeeds since the *name* differs even though the index is redundant). Its other statement, `idx_chunks_tsv` as an expression index on `to_tsvector('simple', text)`, is a permanent no-op: migration 002 already created a GIN index with that exact name on the `tsv` column, so `IF NOT EXISTS` silently skips `build_indexes.py`'s version every time (confirmed: `gin_chunks_tsv: 0.0s` in the run log).

Decision: drop the duplicate index again and delete `scripts/build_indexes.py` (zero other references in the repo). `Makefile`'s `index` target no longer calls it -- migrations already create both real indexes declaratively and idempotently at table-creation time, which is the one place that should own them.

Consequence: the Phase 13 reproducibility run is now safe to repeat without quietly growing the database by an unused 81 MB index on every pass. This is exactly the kind of regression a reproducibility check is supposed to catch, found by actually running the pipeline rather than reading the scripts.

### D26 `make` is not installed on this machine; the Makefile itself had three broken targets

Context: this is a Windows machine with no `make.exe` on `PATH` (confirmed: `which make` / `where make` both fail), so the walkthrough's own verification commands (`make serve &`, `make test`, etc.) cannot literally run here, and neither had `make all` ever been run before Phase 13. Running the Makefile's underlying commands directly (translating each target to its real invocation) surfaced that three of its targets were themselves broken and had never been exercised:
- `ingest` called `nusalens.cli data ingest` (the `data` subcommand group), but `ingest` is a top-level command (`nusalens.cli ingest`) -- `data ingest` does not exist (`Error: No such command 'ingest'`).
- `index` called `nusalens.cli index build` and `eval` called `nusalens.cli eval retrieval` and `train-embed` called `nusalens.cli train embed` -- none of `index`, `eval`, or `train` subcommand groups were ever built on the CLI. Every phase's real indexing/evaluation/training work was done through standalone `scripts/`/`training/` entry points instead (the same pattern already recorded in D11 for Phase 6), never wired into the CLI the Makefile assumes.

Decision: fix all three targets to call what actually exists and does the work: `ingest` -> `nusalens.cli ingest` (still a Phase 1 placeholder, noted honestly, not a Phase 13 scope item to build out); `index` -> `nusalens.retrieval.chunker` + `nusalens.retrieval.embed_chunks`; `eval` -> `scripts/phase5_eval.py`; `train-embed` -> `training/train_embed.py`.

Consequence: `reports/13_quality.md` records the wall time for the corrected sequence (run directly, not through `make`, since `make` isn't present) as the reproducibility measurement this phase's DoD asks for, and states plainly that `make` itself needs installing on a fresh Windows checkout before the walkthrough's literal verification commands work -- rather than silently working around it without saying so.

### D27 Incident: the reproducibility check overwrote `reports/01_data_inventory.md`, and the real ingestion CLI was never actually mounted

Context: fixing the Makefile's `ingest` target (D26) by pointing it at `nusalens.cli ingest` (rather than diagnosing *why* `data ingest` failed) was the wrong fix. Running it executed `cli.py`'s top-level `ingest()` command, which every previous phase's own description called "Place data-pipeline placeholder files" -- and which, read closely only after running it, unconditionally overwrites `reports/01_data_inventory.md` with 5 rows of `0 | placeholder`, destroying whatever the real Phase 1 report said. This is not a git repository (`git log` confirms no history), so the original content is not recoverable.

Investigating *why* `nusalens.cli data ingest` had no real command behind it (the actual question D26 should have asked) found the real cause: `src/nusalens/data_cli.py` -- a complete, working `ingest()` command built on the real `ingest/base.py`/`ingest/worldbank.py`/`ingest/wikipedia.py` connectors -- was never mounted into the main `nusalens.cli` Typer app at all. It is not a guess that this is the module that actually produced the real corpus: `data/manifest.jsonl`'s own append-only log (field names `path`/`source`/`rows`/`sha256`/`ingested_at`/`tool_version`/`scanned`) matches `data_cli.ingest()`'s `append_manifest()` calls exactly, and its row counts (15,000 documents, final 38,216 `fact_indicator` rows) match the live database. The module was simply orphaned -- written, used once by hand, never wired into the CLI surface the Makefile and walkthrough assume.

Decision: mount `data_cli.ingest()` onto `cli.py`'s existing `data_app` group (`data_app.command(name="ingest")(data_cli.ingest)`), so `nusalens.cli data ingest` is now the real command. Delete the dangerous top-level placeholder `ingest()` command entirely -- two different commands named "ingest" with opposite effects (one does real API/HF ingestion, the other wipes a report to placeholder rows) is an unacceptable footgun to leave in place even with a warning. Reconstruct `reports/01_data_inventory.md` from the only evidence that still exists: live database queries (source, license, row counts), `data/manifest.jsonl` (real ingestion provenance with timestamps and sha256), and `config/sources.yaml` (S1-S5 definitions) -- marked explicitly as a Phase 13 reconstruction, not a restoration, so a reader knows its provenance differs from every other report in this project.

Consequence: this was a real, if minor, self-inflicted data-loss incident during a phase whose entire purpose is catching exactly this class of problem -- caused by fixing a broken-looking command by pointing it at *a* command that ran without error, rather than first asking why the intended one didn't exist. The lesson applied for the rest of Phase 13 and beyond: before fixing a Makefile/script target that looks broken, read what the replacement actually does, not just whether it exits 0.

### D28 Three measured optimizations: connection pooling, reranker candidate count, HNSW ef_search

Context: Phase 11 measured rerank_ms as the single largest per-request cost (mean ~9.1s), with retrieve_ms (~0.2s) and the ~25 scattered `psycopg.connect()` call sites named as smaller candidates. Each was measured before changing anything, via dedicated scripts (`scripts/phase13_profile_connections.py`, `scripts/phase13_profile_rerank.py`, `scripts/phase13_profile_ef_search.py`), all against the real 70-item DOC/HYBRID golden subset or a direct repeated-call benchmark, not estimated.

1. **Connection pooling.** Every DOC/HYBRID `answer()` call opened 4+ fresh `psycopg.connect()`s (dense search, sparse search, `_sources_for`, `_log`). Added an optional `pool` parameter to `DenseSearcher`/`SparseSearcher` (falls back to a plain `psycopg.connect()` when `None`, so every existing caller is unaffected) and wired a shared `psycopg_pool.ConnectionPool` through `Pipeline`. Measured: 30 repeated dense+sparse call pairs, 19157ms unpooled vs 16624ms pooled -- 1.15x, 84ms saved per call pair.
2. **Reranker candidate count (`k_rerank`).** Swept 30 (prior default) / 15 / 10 against the golden set: nDCG@10 is *higher* at 15 (0.6415) and 10 (0.6477) than at 30 (0.6298) -- RRF's own top candidates are already good, and feeding the cross-encoder more borderline ones adds noise, not signal. Recall@10 drops modestly (0.800 -> 0.771 -> 0.757). Chose 15: mean rerank latency 6058ms -> 3171ms (1.9x) with *better* ranking quality and an acceptable 2.9pp recall cost. Applied as the new default in all four `HybridSearcher` rerank methods and `config/settings.yaml`'s `retrieval.rerank_top_n` (which, discovered while making this change, was never actually read by any code -- a second small dead-config finding, now at least documented even though still unwired).
3. **HNSW `ef_search`.** Swept 20 / 40 (pgvector's own default) / 100: on this ~41k-row corpus, query latency is already ~3ms at every setting tested (no real latency cost to raising it), while recall@10 keeps improving with a higher value (0.443 -> 0.500 -> 0.543). Set `DenseSearcher`'s default to 100 (`SET hnsw.ef_search = 100` issued per query) -- a recall improvement with no measured latency tradeoff on this corpus size, not a speed/quality tradeoff like #2.

Consequence: Phase 9/11's already-published latency numbers (rerank_ms mean ~9.1s, recall@10 0.50 for dense) reflect the *prior* defaults (k_rerank=30, ef_search=40) and are left as accurate historical records, not retroactively edited; `reports/13_quality.md` carries the before/after table and states plainly that a fresh Phase 9 e2e run would show different (better) numbers under the new defaults, which was not re-run in full (budget) but the retrieval-level deltas above are independently verified.

### D29 No secrets found in logs (hardening check)

Context: Phase 13's hardening checklist asks for "no secrets in logs."

Decision/measured: grepped every log/observability call site (`structlog` calls in `observability.py`/`pipeline.py`, `print()` statements in scripts, `typer.echo()` in `cli.py`) for `db_password`/`db_ro_password`/API keys -- none found outside `config.py`'s own field definitions (where they are declared, not logged). `config.py`'s `db_dsn`/`db_ro_dsn` properties do embed the password in the connection string itself, which is unavoidable (psycopg needs it to connect), but no code path logs the DSN string as a whole; `cli.py health --json` reports only `server_version`/`pgvector` version, not the DSN.

Consequence: passed as checked, not assumed.

### D30 API input validation and graceful degradation, verified not assumed

Context: Phase 13's hardening checklist asks for "API input validation" and "graceful degradation when the LLM or database is down" -- both need to actually be checked, not inferred from "we used Pydantic" / "we have a try/except somewhere."

Decision/measured:
1. **Input validation.** `AskRequest.question`, `SearchRequest.query`, and the matching MCP tool params had no length bound (an empty or multi-megabyte string would reach the pipeline/LLM/DB unchecked). Added `min_length=1, max_length=1000` (`max_length=2000` for `sql`, since queries legitimately run longer) to every free-text request field in `api/app.py` and `mcp_server/main.py`. FastAPI's own validation-error response for these (422, `{"detail": [...]}`) was already structured, not a raw traceback -- confirmed, not assumed.
2. **Database down.** Simulated (without touching the real running container) by temporarily pointing `settings.db_port` at a closed port with a 2s `connect_timeout`. `GET /health` returns `200` with a clear `"database": "error: connection timeout expired..."` field in ~4s (two address families attempted, 2s each) -- not a hang, not a crash. `POST /ask` under the same condition returns a structured `500 {"error", "error_type": "ConnectionTimeout"}` via the global handler in ~10s (`Pipeline.__init__` itself fails fast once `RuleRouter`'s and the `ConnectionPool`'s connection attempts time out) -- bounded, not indefinite, and the process does not crash.
3. **LLM down.** Pointed `ollama_client.generate()` at an unreachable endpoint: fails in 4.1s with a clear `ConnectError`, not a hang (the OS refuses the connection immediately since nothing listens on that port; a firewalled/dropped port would instead hit the existing 180s httpx timeout, the slower but still-bounded case).

Consequence: every failure mode checked here is bounded (seconds, not an indefinite hang) and surfaces as a structured error rather than crashing the server process or leaking a raw traceback to the client. The one caveat worth recording: `/ask` and `/search`'s database check only degrades gracefully on the *first* call after the database goes down (building the `Pipeline`/`HybridSearcher` singleton); a request that arrives while the singleton is already built but the database drops mid-request would hit the same per-call `connect_timeout`, not retested separately here.

## Phase 14 decisions

### D31 Convention audit: module docstrings removed, scope boundary for O2/O6 on binary and raw-evidence files

Context: Phase 14's convention audit runs four greps expected to return no matches (em dashes, process narration, dates, and a docstring sweep where only single-line function/class docstrings with a non-obvious contract are tolerated). Running the first three found real, fixable issues: 17 `src/` files carried a module docstring (O5 forbids these outright; the ones with genuinely non-obvious content were converted to `#` comments, the purely descriptive ones deleted), 2 multi-line function docstrings were trimmed to one line (`hybrid_search.py`, `ollama_client.py`), 2 older reports (`reports/04_phase4_report.md`, carried over from before this conversation, and `reports/08_prompts.md`, written in it) and `docs/DECISIONS.md` itself carried real em/en dashes, and `reports/01_data_inventory.md` (the Phase 13 reconstruction, D27) carried the real ingestion timestamps from `data/manifest.jsonl` directly in its prose.

Decision: fixed every match inside authored prose and code (`#` comments, reports, `docs/`). Two categories were deliberately left out of scope, both already consistent with O2's own stated exception ("timestamps emitted at runtime by code into logs, JSONL records, and database rows are data, not authorship, and are allowed"):
1. **Binary/compiled artifacts**: `models/*.safetensors`, `*.gguf`, `tokenizer.json` (model vocab data), and `__pycache__/*.pyc` incidentally contain the em-dash byte sequence inside binary data, not text anyone authored; not edited.
2. **`reports/raw/*`**: raw tool/LLM output captured as evidence (synthesizer answers, judge verdicts, MCP transcripts) can itself contain an em dash or a date if the model or a library produced one; these are data being preserved verbatim for traceability (R4), the same class of exception O2 already grants JSONL/log timestamps, not prose anyone wrote. `models/e5-nusalens-v1/README.md`'s auto-generated "widget examples" (verbatim corpus excerpts the training library embeds) were cleaned anyway since it was a one-line fix with no ambiguity.
`reports/01_data_inventory.md`'s ingestion timestamps were replaced with manifest-order labels ("run 1, run 2, ... final"), with a pointer to `data/manifest.jsonl` for the exact timestamps, rather than treated as a third exception, since that file is an authored report, not raw captured data.

Consequence: `grep -rnP '\xe2\x80[\x93\x94]' README.md docs/ reports/ src/ prompts/ models/` returns zero matches outside the two documented exception categories above; the date grep returns zero matches outside `reports/raw/`; `grep -rn '"""' src/` returns only single-line function/class docstrings and SQL string literals (which are not docstrings at all, just triple-quoted SQL text, and were left as-is since O5 only governs docstrings).
