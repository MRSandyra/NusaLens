import re

import yaml

from nusalens.config import settings

_ACRONYM_TOKEN = re.compile(r"\b[A-Za-z]{2,6}\b")


def load_acronyms() -> dict[str, str]:
    path = settings.config_dir / "acronyms.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def expand_query(query: str, acronyms: dict[str, str] | None = None) -> str:
    """Append the expansion in parentheses after the first occurrence of each known acronym."""
    acronyms = load_acronyms() if acronyms is None else acronyms
    if not acronyms:
        return query
    upper_map = {k.upper(): v for k, v in acronyms.items()}
    seen: set[str] = set()

    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        key = tok.upper()
        if key in upper_map and key not in seen:
            seen.add(key)
            return f"{tok} ({upper_map[key]})"
        return tok

    return _ACRONYM_TOKEN.sub(_sub, query)
