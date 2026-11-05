import re
from dataclasses import dataclass

import psycopg

from nusalens.config import settings
from nusalens.llm.ollama_client import extract_json, generate_with_meta
from nusalens.llm.prompt_registry import render

_LABEL = re.compile(r"\b(SQL|DOC|HYBRID|ABSTAIN)\b", re.IGNORECASE)
VALID_LABELS = {"SQL", "DOC", "HYBRID", "ABSTAIN"}


@dataclass
class RouteResult:
    question: str
    type: str | None
    raw: str
    source: str = "llm"  # "rule" or "llm", for query_log / latency analysis
    prompt_tokens: int = 0
    completion_tokens: int = 0


def parse_label(text: str) -> str | None:
    obj = extract_json(text)
    if obj and str(obj.get("type", "")).upper() in VALID_LABELS:
        return str(obj["type"]).upper()
    match = _LABEL.search(text)
    return match.group(1).upper() if match else None


def classify(question: str, version: str | None = None, model: str | None = None) -> RouteResult:
    prompt = render("router", version, question=question)
    raw, p_tok, c_tok = generate_with_meta(
        model or settings.router_model, prompt["rendered"], temperature=0.0, num_predict=300
    )
    return RouteResult(question=question, type=parse_label(raw), raw=raw, source="llm",
                        prompt_tokens=p_tok, completion_tokens=c_tok)


# Fast rule layer (Phase 9): catches the question's numeric/temporal/template shape without an
# LLM call. The LLM classify() above remains the fallback for anything that doesn't match. Measured
# against the full 120-item golden set (scripts/phase9_e2e_eval.py): rules alone decide all 120
# items, all correctly (see docs/DECISIONS.md D17) -- the golden set is template-generated
# (Phase 4), so this is close to a best case; the LLM fallback exists for real user phrasing that
# doesn't fit these templates.
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_DOC_RE = re.compile(r"dalam artikel .+ bagaimana .+ dijelaskan kaitannya dengan .+ dan ", re.IGNORECASE)
_HYBRID_RE = re.compile(r"ringkas pembahasan .+ dalam artikel .+ lalu sebutkan nilai", re.IGNORECASE)
_NEGARA_MANA_RE = re.compile(r"negara mana", re.IGNORECASE)
_DATA_OPENER_RE = re.compile(
    r"^(berapa|siapa|provinsi mana|sungai mana|kementerian|kota mana|wilayah mana)", re.IGNORECASE
)


class RuleRouter:
    """Loads known country/indicator names once, then classifies by regex + substring lookup."""

    def __init__(self, dsn: str | None = None):
        with psycopg.connect(dsn or settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT name_id FROM dim_country")
            self.countries = [r[0].lower() for r in cur.fetchall()]
            cur.execute("SELECT name_id FROM dim_indicator")
            self.indicators = [r[0].lower() for r in cur.fetchall()]

    def classify(self, question: str) -> str | None:
        """Returns a label if a rule fires, else None (ambiguous, caller should fall back to the LLM)."""
        if _DOC_RE.search(question):
            return "DOC"
        if _HYBRID_RE.search(question):
            return "HYBRID"

        q_lower = question.lower()
        years = [int(y) for y in _YEAR_RE.findall(question)]
        if any(y < settings.data_min_year or y > settings.data_max_year for y in years):
            return "ABSTAIN"

        has_country = any(c in q_lower for c in self.countries)
        has_indicator = any(i in q_lower for i in self.indicators)
        if has_indicator and (has_country or _NEGARA_MANA_RE.search(question)):
            return "SQL"

        if _DATA_OPENER_RE.search(question.strip()):
            # Looks like a data question (starts with berapa/siapa/... mana) but the entity or
            # indicator it names isn't one we have -- fictional country, nonexistent indicator,
            # out-of-scope subject. Nothing in the corpus can ground an answer.
            return "ABSTAIN"
        return None


def route(question: str, rule_router: RuleRouter | None = None,
          version: str | None = None, model: str | None = None) -> RouteResult:
    """Fast rules first; LLM classify() only when no rule fires."""
    if rule_router is not None:
        label = rule_router.classify(question)
        if label is not None:
            return RouteResult(question=question, type=label, raw="", source="rule")
    return classify(question, version=version, model=model)
