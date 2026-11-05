import re
from dataclasses import dataclass, field

from nusalens.config import settings
from nusalens.llm.ollama_client import extract_json, generate_with_meta
from nusalens.llm.prompt_registry import active_version, render

_CITATION = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class SynthesisResult:
    question: str
    answer: str
    abstained: bool
    citations: list[str] = field(default_factory=list)
    raw: str = ""
    json_parsed: bool | None = None  # None for versions that don't ask for JSON (v1, v2)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def synthesize(question: str, chunks: list[dict], version: str | None = None,
               model: str | None = None, num_predict: int = 6000) -> SynthesisResult:
    context = _format_context(chunks) if chunks else "(tidak ada konteks yang relevan ditemukan)"
    prompt = render("synthesizer", version, question=question, context=context)
    raw, p_tok, c_tok = generate_with_meta(
        model or settings.generator_model, prompt["rendered"], temperature=0.0, num_predict=num_predict
    )

    requires_json = (version or active_version("synthesizer")) == "v3"
    obj = extract_json(raw)
    json_parsed = None
    if requires_json:
        json_parsed = obj is not None and "answer" in obj

    if obj is not None and "answer" in obj:
        answer = str(obj.get("answer", "")).strip()
        abstained = bool(obj.get("abstained", False)) or not answer
    else:
        answer = raw.strip()
        abstained = False

    citations = _CITATION.findall(answer)
    return SynthesisResult(question=question, answer=answer, abstained=abstained, citations=citations,
                            raw=raw, json_parsed=json_parsed, prompt_tokens=p_tok, completion_tokens=c_tok)
