"""Phase 10: real MCP client smoke test over a real stdio subprocess (not in-process, not a unit
test double). Launches `python -m nusalens.mcp_server.main --transport stdio`, performs the
handshake, lists tools/resources/prompts, and calls one of each required category plus the SQL
guardrail re-proof. Prints the full transcript as JSON to stdout.
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from nusalens.config import PROJECT_ROOT


def _tool_result_to_obj(result) -> dict:
    text = result.content[0].text if result.content else None
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = text
    return {"is_error": result.is_error, "parsed": parsed}


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nusalens.mcp_server.main", "--transport", "stdio"],
    )
    transcript: dict = {"calls": []}

    async with Client(params, raise_exceptions=False) as client:
        transcript["server_info"] = {
            "name": client.server_info.name if client.server_info else None,
            "version": client.server_info.version if client.server_info else None,
        }
        transcript["instructions"] = client.instructions

        tools = await client.list_tools()
        transcript["tools"] = [{"name": t.name, "description": t.description} for t in tools.tools]

        resources = await client.list_resources()
        transcript["resources"] = [{"uri": str(r.uri), "description": r.description} for r in resources.resources]

        prompts = await client.list_prompts()
        transcript["prompts"] = [{"name": p.name, "description": p.description} for p in prompts.prompts]

        calls = [
            ("list_indicators", {}),
            ("search_documents", {"params": {"query": "Apa itu rutenium?", "k": 3}}),
            ("query_indicators", {"params": {"country": "Vietnam", "indicator": "Gini index",
                                              "year_from": 2020, "year_to": 2022}}),
            ("run_readonly_sql", {"params": {"sql": "SELECT value FROM v_indicator_series "
                                                     "WHERE country_name = 'Indonesia' AND "
                                                     "indicator_name = 'Population, total' AND "
                                                     "year = 2022"}}),
            ("run_readonly_sql", {"params": {"sql": "DROP TABLE documents"}}),  # guardrail re-proof
            ("get_latest_earthquake", {}),
            ("answer_question", {"params": {"question": "Berapa jumlah penduduk Gondor menurut data Bank Dunia?"}}),
        ]
        for name, args in calls:
            result = await client.call_tool(name, args)
            transcript["calls"].append({"tool": name, "arguments": args, "result": _tool_result_to_obj(result)})

        resource_reads = {}
        for uri in ("nusalens://schema", "nusalens://sources", "nusalens://stats"):
            content = await client.read_resource(uri)
            text = content.contents[0].text if content.contents else None
            try:
                resource_reads[uri] = json.loads(text) if text else None
            except json.JSONDecodeError:
                resource_reads[uri] = text
        transcript["resource_reads"] = resource_reads

        prompt_result = await client.get_prompt(
            "analyze_indicator_trend", {"indicator": "GDP per capita (current US$)", "country": "Indonesia"}
        )
        transcript["prompt_render"] = prompt_result.messages[0].content.text

    out_path = PROJECT_ROOT / "reports" / "raw" / "10_mcp_transcript.json"
    out_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
