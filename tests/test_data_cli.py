from typer.testing import CliRunner

from nusalens.data_cli import app

runner = CliRunner()


def test_stats_lists_real_gold_parquet_files():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "documents.parquet" in result.output
    assert "indicators.parquet" in result.output
