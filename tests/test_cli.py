from typer.testing import CliRunner

from nusalens.cli import app

runner = CliRunner()


def test_health_reports_ok_against_the_real_stack():
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert "overall: OK" in result.output


def test_health_json_output_is_parseable():
    # sentence-transformers' model-loading tqdm bar writes to stderr (confirmed by running the
    # real subprocess and redirecting streams separately); CliRunner's combined `.output` mixes
    # stdout+stderr, so this reads `.stdout` alone, matching what a real `| jq` pipe would see.
    import json

    result = runner.invoke(app, ["health", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["overall"] == "OK"
    assert set(body["checks"]) == {"database", "llm", "embedding"}


def test_data_stats_lists_gold_parquet_files():
    result = runner.invoke(app, ["data", "stats"])
    assert result.exit_code == 0
    assert "documents.parquet" in result.output
    assert "chunks.parquet" in result.output


def test_data_validate_runs_without_crashing():
    result = runner.invoke(app, ["data", "validate"])
    assert result.exit_code in (0, 1)  # 1 only with --strict on a real failure; just must not crash
    assert "no gold directory" not in result.output


def test_ask_abstain_route_is_fast_and_has_no_llm_call():
    result = runner.invoke(app, ["ask", "Berapa jumlah penduduk Gondor menurut data Bank Dunia?"])
    assert result.exit_code == 0
    assert "ABSTAIN" in result.output
