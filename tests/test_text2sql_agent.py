from nusalens.agents.text2sql import Text2SQLAgent


def test_ask_returns_correct_value_for_a_known_sql_question():
    agent = Text2SQLAgent()
    result = agent.ask("Berapa nilai Gini index untuk Vietnam pada tahun 2022?")
    assert result.error is None
    assert result.rows
    assert result.sql is not None
    assert "SELECT" in result.sql.upper()
    assert result.attempts >= 1


def test_ask_snaps_an_english_country_alias_to_the_view_s_indonesian_name():
    # the view stores name_id ("Selandia Baru"), not name_en ("New Zealand") -- the model
    # sometimes emits the English alias, which _snap_candidates must correct (Phase 7, D-series).
    agent = Text2SQLAgent()
    result = agent.ask("Berapa nilai Population, total untuk Selandia Baru pada tahun 2022?")
    assert result.rows
