Phase 4 golden set evaluation harness. eval/golden.jsonl, eval/golden.sha256.

Construction method

qwen3:4b on Ollama exposes a `thinking` capability that cannot be disabled with
`think: false` on the installed Ollama build (reports/raw/04_llm_latency_probe.txt):
a single 60-token generation took 113.3 seconds and the response text was
still inline chain-of-thought, not a usable question. Drafting 120 items this
way was infeasible inside the tool's execution limits. Questions are instead
generated deterministically from real gold-layer rows (scripts/build_golden.py),
tethered to concrete chunk_id and executed SQL, then filtered.

DOC and HYBRID question terms are selected by BM25 IDF computed over the full
41,297-chunk corpus (min_idf 3.0) so only corpus-rare, topical terms are used;
an initial version used a hand stopword list and let function words like
"cukup" and "sehingga" through, producing incoherent questions, caught by
manual inspection and fixed before freezing (see round 1 in
reports/raw/04_golden_build.txt history superseded by the final run).

Pipeline

1. DOC pool: 200+ token chunks, one per source document, with 3 IDF-ranked terms
   templated into a question referencing the article title.
2. SQL pool: indicator_code x country_iso3 pairs with at least 10 non-null
   years, five query templates (latest value, cross-country max, period
   average, two-year trend, year coverage count) cycled across pairs.
3. HYBRID pool: same construction as DOC, paired with an Indonesia-only SQL
   lookup on a cycled indicator code.
4. ABSTAIN: 15 hand-written unanswerable questions (fictional entities, years
   outside 1960-2024 coverage, false-premise subjects), no gold_chunk_ids or
   gold_sql, single set given the exact 15-item quota.

Round-trip filter (DOC, HYBRID)

Each question is scored against the corpus with a real BM25 (k1=1.5, b=0.75)
built over all 41,297 chunks; the question is kept only if its own source
chunk_id appears in the top 50 ranked results. This rejects the question when
retrieval could plausibly fail to ground it.

SQL verification (SQL, HYBRID)

Every generated SQL statement is executed against the live Postgres warehouse
(fact_indicator, dim_indicator) with psycopg. An item is rejected if the query
raises or returns only NULL. Returned values are appended to
gold_answer_facts as formatted strings, giving the item a real expected
answer fragment, not an invented one.

Results, reports/raw/04_golden_build.txt

    pools:    doc=135  sql=105  hybrid=75  abstain=15
    sql verified kept=105 rejected=0
    hybrid verified kept=50 rejected=0
    round-trip: doc_rt_pass=45 doc_rt_fail=0 hybrid_rt_pass=50 hybrid_rt_fail=11
    final distribution: DOC=45 SQL=35 HYBRID=25 ABSTAIN=15  total=120

Independent verification, reports/raw/04_golden_verify.txt (separate script,
re-reads the frozen file rather than trusting the builder's own counts)

    rows: 120
    distribution: DOC=45 SQL=35 HYBRID=25 ABSTAIN=15
    unique qid: True
    unique question: True
    DOC/HYBRID with gold_chunk_ids: 70/70
    SQL/HYBRID with gold_sql: 60/60
    ABSTAIN with no chunk and no sql: 15/15
    items carrying gold_answer_facts: 105
    FROZEN HASH MATCHES: True

Freeze

eval/golden.jsonl sha256: 16da4790acd57953cece89809b805eea44e9b0defd6cd650338ee89f9f2f05bf
recorded in eval/golden.sha256. Any future edit to golden.jsonl must be
treated as a new version with a new recorded hash; scripts/verify_golden.py
recomputes and compares against the recorded value.

DoD

    target 120 items: met, 120/120
    distribution 45/35/25/15: met exactly
    DOC/HYBRID grounded in a real chunk_id: met, 70/70
    SQL/HYBRID grounded in an executed, non-null SQL result: met, 60/60
    frozen with sha256: met, hash verified independently

Known limitation

DOC and HYBRID question phrasing is template-based (three fixed sentence
frames), not linguistically varied, because the LLM path was not viable at
this VRAM/latency budget. Terms are real and IDF-filtered, and every item is
independently round-trip or SQL verified, but a human reviewer should still
skim eval/golden.jsonl before using it to gate model changes.
