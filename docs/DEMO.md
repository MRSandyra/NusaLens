# Five-minute demo script

For a technical interview. Start the stack first (`docker compose up -d`, `make serve` or `python -m nusalens.cli serve --api-port 8001`), then follow this order.

## 1. Architecture (30 seconds)

Show `README.md`'s Mermaid diagram. One sentence: a fast rule-based router sends a question to SQL, document retrieval, or both at once, and a local LLM synthesizes the answer, never fabricating when the evidence is missing.

## 2. SQL route (1 minute)

Type into the UI: **"Berapa nilai Gini index untuk Vietnam pada tahun 2022?"**

Point out, in order:
- The route badge says SQL, answered in about 2.6 seconds (p50 from `reports/09_e2e.md`).
- Expand "Tampilkan SQL": the exact generated, validated, executed statement.
- The trend chart below it, built from a second query over the full year range.
- Confidence 0.90.

## 3. DOC route, with a deliberately adversarial question (1 minute)

Type: **"Dalam artikel Paus Pius IX, bagaimana paus dijelaskan kaitannya dengan pius dan giovanni?"**

Point out:
- Citations under "Sumber" are clickable, each a real Wikipedia URL.
- This is the item from the Phase 8 judge-calibration set; mention in one sentence that a different question of this same shape is the one documented failure case in `reports/09_e2e.md` (D15/D19): the synthesizer sometimes denies a fact sitting verbatim in its own top-ranked chunk. Naming a real, found, documented bug here is a credibility signal, not a liability.

## 4. HYBRID route (1 minute)

Type: **"Ringkas pembahasan energi dalam artikel Energi gelombang, lalu sebutkan nilai Maternal mortality ratio (modeled estimate, per 100,000 live births) Indonesia tahun 2023."**

Point out:
- SQL and document retrieval ran concurrently (`asyncio.gather`), not sequentially; show the latency breakdown panel.
- The numeric half of the answer is independently checkable against the SQL panel.

## 5. ABSTAIN, the safety story (30 seconds)

Type: **"Berapa jumlah penduduk Gondor menurut data Bank Dunia?"**

Point out:
- Answered in about 20 milliseconds (`reports/09_e2e.md`): no LLM call at all, caught by the rule router before anything expensive runs.
- Abstain recall is 100% (15/15) on the golden ABSTAIN set, the project's own stated most important safety metric.

## 6. MCP server, the primary differentiator (1 minute)

Switch to a terminal:
```bash
python -m nusalens.cli mcp
```
In another terminal, run the smoke client transcript (or show the saved one):
```bash
cat reports/raw/10_mcp_transcript.json
```
Point out:
- 6 tools, 3 resources, 2 prompts, verified against a real client over stdio, not just unit tests.
- The guardrail re-proof: a `DROP TABLE` sent through `run_readonly_sql` comes back `{"ok": false, "error": "keyword denylist match"}`, not a crash.

## If asked for one more thing

`GET /metrics` on the API: live query count, mean latency per stage, abstain rate, route distribution, computed from the real `query_log` table, not a mock.
