import json

import pytest

from nusalens.mcp_server.main import server


@pytest.mark.asyncio
async def test_list_tools_resources_prompts():
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "search_documents", "query_indicators", "run_readonly_sql",
        "list_indicators", "get_latest_earthquake", "answer_question",
    }
    assert all(t.description for t in tools), "every tool needs a non-empty description"

    resources = await server.list_resources()
    assert {str(r.uri) for r in resources} == {"nusalens://schema", "nusalens://sources", "nusalens://stats"}

    prompts = await server.list_prompts()
    assert {p.name for p in prompts} == {"analyze_indicator_trend", "compare_countries"}


@pytest.mark.asyncio
async def test_run_readonly_sql_rejects_write_statements_through_mcp():
    for payload in [
        "DROP TABLE documents",
        "DELETE FROM fact_indicator WHERE iso3 = 'IDN'",
        "SELECT * FROM pg_shadow",
    ]:
        result = await server.call_tool("run_readonly_sql", {"params": {"sql": payload}})
        body = json.loads(result.content[0].text)
        assert body["ok"] is False, f"guardrail let through: {payload}"
        assert "error" in body and "error_type" in body


@pytest.mark.asyncio
async def test_run_readonly_sql_accepts_a_safe_select():
    result = await server.call_tool(
        "run_readonly_sql",
        {"params": {"sql": "SELECT value FROM v_indicator_series WHERE country_name = 'Indonesia' "
                            "AND indicator_name = 'Population, total' AND year = 2022"}},
    )
    body = json.loads(result.content[0].text)
    assert body["ok"] is True
    assert body["rows"]


@pytest.mark.asyncio
async def test_query_indicators_rejects_unknown_entity_with_structured_error():
    result = await server.call_tool("query_indicators", {"params": {"country": "Wakanda"}})
    body = json.loads(result.content[0].text)
    assert body["ok"] is False
    assert body["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_list_indicators_returns_catalog():
    result = await server.call_tool("list_indicators", {})
    body = json.loads(result.content[0].text)
    assert body["ok"] is True
    assert body["n"] > 0
    assert {"code", "name_id", "unit"}.issubset(body["indicators"][0].keys())


@pytest.mark.asyncio
async def test_resources_return_valid_json():
    for uri in ("nusalens://schema", "nusalens://sources", "nusalens://stats"):
        contents = await server.read_resource(uri)
        json.loads(contents[0].content)  # must not raise


@pytest.mark.asyncio
async def test_stats_resource_reports_real_table_counts():
    contents = await server.read_resource("nusalens://stats")
    body = json.loads(contents[0].content)
    assert body["documents"] > 0
    assert body["chunks"] > 0
    assert body["dim_country"] == 16


@pytest.mark.asyncio
async def test_sources_resource_lists_wikipedia_and_world_bank():
    contents = await server.read_resource("nusalens://sources")
    body = json.loads(contents[0].content)
    sources = {d["source"] for d in body["documents"]}
    assert "S2_wikipedia_id" in sources
    indicator_sources = {d["source"] for d in body["indicators"]}
    assert "World Bank" in indicator_sources
    assert any(l["name"].startswith("BMKG") for l in body["live"])


@pytest.mark.asyncio
async def test_analyze_indicator_trend_prompt_renders():
    result = await server.get_prompt(
        "analyze_indicator_trend", {"indicator": "Gini index", "country": "Vietnam"}
    )
    text = result.messages[0].content.text
    assert "Gini index" in text
    assert "Vietnam" in text


@pytest.mark.asyncio
async def test_compare_countries_prompt_renders():
    result = await server.get_prompt(
        "compare_countries", {"indicator": "Gini index", "countries": "Vietnam, Indonesia"}
    )
    text = result.messages[0].content.text
    assert "Gini index" in text
    assert "Vietnam, Indonesia" in text


@pytest.mark.asyncio
async def test_get_latest_earthquake_returns_live_structured_data():
    result = await server.call_tool("get_latest_earthquake", {})
    body = json.loads(result.content[0].text)
    assert body["ok"] is True
    assert "Magnitude" in body["earthquake"]


@pytest.mark.asyncio
async def test_answer_question_tool_abstains_on_fictional_entity():
    result = await server.call_tool(
        "answer_question", {"params": {"question": "Berapa jumlah penduduk Gondor menurut data Bank Dunia?"}}
    )
    body = json.loads(result.content[0].text)
    assert body["ok"] is True
    assert body["route"] == "ABSTAIN"
    assert body["abstained"] is True
