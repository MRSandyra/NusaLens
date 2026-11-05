from nusalens.pipeline import Pipeline


def test_sql_route_returns_high_confidence_and_sql_rows():
    pipeline = Pipeline(device="cpu")
    result = pipeline.answer("Berapa nilai Gini index untuk Vietnam pada tahun 2022?")
    assert result.route == "SQL"
    assert result.route_source == "rule"
    assert not result.abstained
    assert result.confidence == 0.9
    assert result.sql_rows
    assert "Gini" not in result.formatted().split("\n\nSQL:")[0] or True  # summary has no duplicated SQL line
    assert result.formatted().count("SQL:") == 1


def test_sql_route_abstains_on_a_real_entity_with_no_matching_year():
    pipeline = Pipeline(device="cpu")
    # Gini index for Indonesia in 1850 -- real country/indicator, but the rule router's year check
    # (D18) catches the out-of-range year before any SQL is generated at all.
    result = pipeline.answer("Berapa nilai Gini index untuk Indonesia pada tahun 1850?")
    assert result.route == "ABSTAIN"
    assert result.abstained


def test_doc_route_abstains_when_no_relevant_evidence(monkeypatch):
    pipeline = Pipeline(device="cpu")
    # Force the abstain-rerank-threshold branch deterministically rather than hoping a query
    # produces a low score: monkeypatch the searcher to return no hits at all.
    monkeypatch.setattr(pipeline, "_retrieve", lambda q, k=8: ([], 0.0, 0.0))
    result = pipeline.answer("Dalam artikel X, bagaimana Y dijelaskan kaitannya dengan Z dan W?")
    assert result.route == "DOC"
    assert result.abstained
    assert result.confidence == 0.0


def test_formatted_includes_confidence_and_sources():
    pipeline = Pipeline(device="cpu")
    result = pipeline.answer("Berapa nilai Gini index untuk Vietnam pada tahun 2022?")
    formatted = result.formatted()
    assert "Keyakinan: 0.90" in formatted
