# No prompt text lives in Python; every template is loaded from prompts/<role>/vN.yaml.
from pathlib import Path

import yaml

from nusalens.config import PROJECT_ROOT, settings


def _settings_prompts() -> dict:
    path = PROJECT_ROOT / "config" / "settings.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("prompts", {})


def active_version(role: str) -> str:
    cfg = _settings_prompts()
    return cfg.get(role, "v1")


def load_prompt(role: str, version: str | None = None) -> dict:
    v = version or active_version(role)
    path = PROJECT_ROOT / "prompts" / role / f"{v}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no prompt file for role={role} version={v} at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render(role: str, version: str | None = None, **kwargs) -> dict:
    """Returns the loaded prompt dict with 'template' (and 'repair_template' if present) formatted."""
    prompt = dict(load_prompt(role, version))
    prompt["rendered"] = prompt["template"].format(**kwargs)
    return prompt


def list_versions(role: str) -> list[str]:
    role_dir = PROJECT_ROOT / "prompts" / role
    if not role_dir.exists():
        return []
    return sorted(p.stem for p in role_dir.glob("v*.yaml"))
