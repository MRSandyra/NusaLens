from typer.testing import CliRunner
import pytest

from nusalens.data_cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


def test_stats_lists_real_gold_parquet_files():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "documents.parquet" in result.output
    assert "indicators.parquet" in result.output
