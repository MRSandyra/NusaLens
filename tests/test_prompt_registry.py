import pytest

from nusalens.llm.prompt_registry import active_version, list_versions, load_prompt, render


def test_active_version_reads_settings_yaml():
    # config/settings.yaml pins each role's production version; these are the Phase 8/9 decisions
    # (D15/D17), not invented here -- a changed default should be caught by this test.
    assert active_version("router") == "v1"
    assert active_version("text2sql") == "v2"
    assert active_version("synthesizer") == "v1"
    assert active_version("judge") == "v3"


def test_active_version_defaults_to_v1_for_unknown_role():
    assert active_version("no_such_role") == "v1"


def test_list_versions_finds_every_file():
    versions = list_versions("synthesizer")
    assert versions == ["v1", "v2", "v3", "v4"]


def test_list_versions_empty_for_unknown_role():
    assert list_versions("no_such_role") == []


def test_load_prompt_returns_required_keys():
    prompt = load_prompt("router", "v1")
    assert prompt["id"] == "router"
    assert "template" in prompt
    assert "variables" in prompt


def test_load_prompt_missing_version_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("router", "v99")


def test_load_prompt_uses_active_version_when_unspecified():
    assert load_prompt("router")["version"] == load_prompt("router", active_version("router"))["version"]


def test_render_fills_template_variables():
    rendered = render("router", "v1", question="Berapa PDB Indonesia?")
    assert "Berapa PDB Indonesia?" in rendered["rendered"]


def test_render_missing_variable_raises_key_error():
    with pytest.raises(KeyError):
        render("router", "v1")
