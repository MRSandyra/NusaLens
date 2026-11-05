# Phase 1: Data Inventory

**Reconstructed in Phase 13.** The original version of this report was accidentally overwritten by `nusalens.cli ingest` (the top-level placeholder command from early Phase 1 scaffolding, which unconditionally rewrites this exact file with empty placeholder rows) when Phase 13's reproducibility check ran the project's Makefile targets literally. See [DECISIONS.md D27](DECISIONS.md) for the full incident. This reconstruction uses only real, currently-verifiable evidence: live queries against the populated database, `data/manifest.jsonl` (the real ingestion tool's own provenance log), and `config/sources.yaml`. It is not a byte-for-byet restoration of the original Phase 1 report, which is unrecoverable (this is not a git repository -- `git log` confirms no history to recover from).

## Sources

| # | Source | Access | License | Rows ingested | Role | Status |
|---|---|---|---|---|---|---|
| S1 | World Bank Indicators API (`api.worldbank.org/v2`) | HTTP JSON, no key | CC BY 4.0 | 38,216 `fact_indicator` rows across 53 indicators x 16 countries (`dim_indicator.source = 'World Bank'`) | Numeric backbone | Required, stable |
| S2 | Indonesian Wikipedia (`wikimedia/wikipedia`, config `20231101.id`) | HF dataset, offline after download | CC BY-SA 4.0 | 15,000 documents (`documents.source = 'S2_wikipedia_id'`), scanned 22,290 candidates before the 300-char/filter cutoff (`data/manifest.jsonl`) | Document backbone | Complete |
| S3 | Satu Data Indonesia (`data.go.id` CKAN API) | HTTP JSON, no key | Varies per dataset | 0 -- not exercised in this build | Optional dataset metadata | Not used; no phase depends on it |
| S4 | BMKG earthquake/weather feeds (`data.bmkg.go.id`) | HTTP JSON, no key | Open data BMKG | N/A -- live, not indexed | Dynamic MCP tool demo (Phase 10 `get_latest_earthquake`) | Used live, confirmed working (`reports/10_mcp.md`) |
| S5 | TyDiQA Indonesian gold-passage subset (`khalidalt/tydiqa-goldp`) | HF dataset | Apache-2.0 | Used as phrasing reference for golden-set construction (Phase 4) | Question-phrasing seed | Used |

**Deliberately excluded:** `webapi.bps.go.id` requires key registration and sits behind bot protection (HTTP 403), making it unreliable for an automated pipeline (`config/sources.yaml`'s `excluded.bps`). No phase depends on it.

## S1 ingestion provenance (from `data/manifest.jsonl`)

The World Bank ingest ran multiple times as indicator/country coverage expanded; the manifest's own append-only log shows the progression, final run authoritative. Exact timestamps live in `data/manifest.jsonl` itself (a runtime-emitted provenance log, not an authored report) rather than repeated here:

| Run (in manifest order) | Rows | sha256 (truncated) |
|---|---|---|
| 1 | 6,686 | `dafe12cf...` |
| 2 | 6,686 | `6f216f45...` |
| 3 | 8,504 | `6f216f45...` |
| 4 (final) | **38,216** | `1c12bdab...` |

S2 ran once: 15,000 documents, 22,290 scanned, sha256 `3fdae6c1...`.

## Etiquette and rate limiting

`config/sources.yaml`: `User-Agent: "NusaLens/0.1 (portfolio project)"`, default rate limit 2 requests/second per host, default cache TTL 168 hours. Enforced in `ingest/base.py`'s `BaseConnector`.

## CLI wiring note (Phase 13 fix, D27)

The real ingestion connectors (`ingest/base.py`, `ingest/worldbank.py`, `ingest/wikipedia.py`) that actually produced the data above are driven by `src/nusalens/data_cli.py`'s `ingest()` command -- which was never mounted into the main `nusalens.cli` app until Phase 13. Before this fix, `python -m nusalens.cli data ingest` had no real command behind it (`Error: No such command 'ingest'`), and the only reachable `ingest` command was the top-level placeholder that caused this file's overwrite. Both are now fixed: `nusalens.cli data ingest` runs the real connectors; the top-level placeholder command has been removed (D27).
