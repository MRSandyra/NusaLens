# Phase 10: MCP Server

## SDK verification (mandatory first step)

`mcp==2.3.0` was installed and inspected directly before any server code was written, per section 2.5's version warning (v2 renamed `FastMCP` to `MCPServer`). Full findings in [DECISIONS.md D20](../docs/DECISIONS.md): server class is `mcp.server.mcpserver.MCPServer`, decorator API (`.tool()`, `.resource()`, `.prompt()`) matches the old FastMCP shape, transport is `server.run(transport="stdio"|"sse"|"streamable-http")`, and the client side (`mcp.client.client.Client`) can drive either a real subprocess (`StdioServerParameters`) or an in-process server instance. No fallback to `mcp>=1.28,<2` or `fastmcp` 4.x was needed. Pinned to the exact resolved version (`mcp==2.3.0`) in `pyproject.toml`.

## What MCP is, and why tools/resources/prompts are separate

MCP (Model Context Protocol) is a JSON-RPC protocol that lets a model-driven client discover and call a server's capabilities without the server and the model needing to agree on anything beyond the protocol itself. A client can point at any MCP server -- this one, or an unrelated one -- and get a uniform `list_tools`/`call_tool`/`list_resources`/`read_resource`/`list_prompts`/`get_prompt` surface.

The three capability types exist because they answer different questions:
- **Tools** are things the *model* decides to invoke, mid-conversation, based on its own judgment about what the question needs (`search_documents`, `query_indicators`, `run_readonly_sql`, `list_indicators`, `get_latest_earthquake`, `answer_question`). They can compute, call out to the database or the live BMKG feed, and have real cost (retrieval, SQL, an LLM call for `answer_question`).
- **Resources** are passive data the *client* (not the model) can read and attach to context ahead of time -- a schema card, a source list, row counts -- without the model needing to decide to fetch them. `nusalens://schema`, `nusalens://sources`, `nusalens://stats`.
- **Prompts** are reusable message templates the *user* (via the client's UI) explicitly selects, not something the model reaches for on its own -- `analyze_indicator_trend`, `compare_countries` turn a few named fields into a ready-made instruction that then drives the model's own tool use.

Mixing these up would be a design mistake: putting `nusalens://stats` behind a tool call would make the model spend a turn deciding to fetch something a client could have attached for free; putting `run_readonly_sql` behind a resource would hide a real side-effecting, cost-bearing operation behind something that looks like static data.

## Tool schema design

Every tool takes a single Pydantic `BaseModel` parameter (except the two that take none: `list_indicators`, `get_latest_earthquake`). Confirmed empirically (D20) that the installed SDK nests the model's fields under a `"params"` key in the generated JSON schema rather than flattening them -- this is why `mcp_smoke_client.py` and `test_mcp_tools.py` call tools as `{"params": {...}}`, not with flat kwargs.

Two deliberate schema-level decisions:
- **`query_indicators` is structured, not free SQL.** It takes `country`/`indicator`/`year_from`/`year_to`/`limit` as typed fields, resolves `country`/`indicator` against the real `dim_country`/`dim_indicator` name lists (exact match, case-insensitive) before building its own fixed-template SQL, and returns a structured `UnknownEntity`-style error if either name doesn't resolve -- never executes a string the caller supplied. `run_readonly_sql` exists alongside it specifically for the cases this shape can't express (aggregates, rankings, year-over-year comparisons), and it is the one tool that does accept a SQL string -- behind the full Phase 7 guard stack (`validate_sql`/`execute_readonly`), re-proven at the protocol boundary rather than assumed to still hold (see DoD #3 below).
- **Every tool returns `{"ok": bool, ...}`**, with `{"ok": false, "error": str, "error_type": str}` on any failure (bad entity name, rejected SQL, a network error from the live BMKG call, or any other exception) instead of letting an exception surface as a raw traceback over the protocol. Checked directly: a `DROP TABLE` sent through `run_readonly_sql` comes back as a normal (non-protocol-error) tool result whose JSON body says `"ok": false, "error": "keyword denylist match"`.

## How tool descriptions affect model selection

A model picks a tool by matching the question against each tool's `description` string -- it never sees the implementation. A weak or overlapping description is a real failure mode here specifically because three tools can all plausibly answer a numeric question: `query_indicators`, `run_readonly_sql`, and `answer_question`. Each description was written to name what it is *not* for, not just what it is:
- `search_documents`'s description ends with "not for numeric World Bank indicator values, which query_indicators or run_readonly_sql answer directly" -- without this, a model could reach for hybrid search on a GDP question because "search" sounds generic enough to fit.
- `query_indicators`'s description says "Not free-form SQL -- call list_indicators first... Use run_readonly_sql instead for anything this structured shape cannot express" -- without the second clause, a model faced with a ranking or aggregate question has no signal to escalate to the SQL tool instead of retrying `query_indicators` with parameters it doesn't have.
- `answer_question`'s description says "use the other tools individually when the intermediate evidence... is what's needed instead" -- without this, a model that wants to show its own retrieved chunks or raw SQL rows (rather than a finished prose answer) has no cue to avoid the end-to-end tool.

This matters in proportion to how similar two tools' surface area looks; `list_indicators` and `get_latest_earthquake` needed no such hedging because nothing else in the tool set could be mistaken for them.

## Error handling

Confirmed structured, not raw tracebacks, in `tests/test_mcp_tools.py` and `reports/raw/10_mcp_transcript.json`:
- Unknown country/indicator in `query_indicators` -> `{"ok": false, "error_type": "ValueError", ...}`.
- A write statement in `run_readonly_sql` -> `{"ok": false, "error": "keyword denylist match", ...}`, with the rejected SQL echoed back for the caller's own logging.
- Any other exception (e.g. the live BMKG call failing) is caught in every tool body and returned the same way.

## Real-client verification

`scripts/mcp_smoke_client.py` launches `python -m nusalens.mcp_server.main --transport stdio` as an actual subprocess (not an in-process test double) via `mcp.client.client.Client` + `StdioServerParameters`, performs the handshake, lists tools/resources/prompts, and calls all 6 tools plus one extra guardrail re-proof call (7 calls total). Full transcript: `reports/raw/10_mcp_transcript.json`.

| Call | Result |
|---|---|
| `list_indicators` | `ok: true`, 53 indicators |
| `search_documents("Apa itu rutenium?")` | `ok: true`, 3 cited chunks |
| `query_indicators(Vietnam, Gini index, 2020-2022)` | `ok: true`, 2 rows |
| `run_readonly_sql("SELECT ... Population, total ... 2022")` | `ok: true`, 1 row |
| `run_readonly_sql("DROP TABLE documents")` | **`ok: false`, `error: "keyword denylist match"`** -- guardrail re-proven through MCP |
| `get_latest_earthquake` | `ok: true`, live BMKG data |
| `answer_question("Berapa jumlah penduduk Gondor...")` | `ok: true`, `route: "ABSTAIN"`, correctly declines |

All three resources (`nusalens://schema`, `nusalens://sources`, `nusalens://stats`) and both prompts (`analyze_indicator_trend`, `compare_countries`) were also read/rendered in the same transcript.

## DoD

1. The server starts over stdio without error. **Met** -- the smoke client's subprocess launch and handshake succeeded with exit code 0.
2. `reports/raw/10_mcp_transcript.json` holds real output: handshake, tool list, and responses from at least four distinct tool calls. **Met** -- 6 distinct tools, 7 calls total.
3. `run_readonly_sql` rejects write statements when called through MCP, re-proving the guardrail at the protocol boundary. **Met**, see table above.
4. This report. **Met.**

## Artifacts

- `src/nusalens/mcp_server/main.py` -- server, 6 tools, 3 resources, 2 prompts.
- `scripts/mcp_smoke_client.py` -- real stdio-subprocess client.
- `tests/test_mcp_tools.py` -- 6 passing unit tests (schema completeness, guardrail re-proof, unknown-entity error, resource JSON validity).
- `docs/mcp_client_config.md` -- desktop client config block.
