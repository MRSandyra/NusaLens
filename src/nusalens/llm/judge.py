from dataclasses import dataclass

from nusalens.config import settings
from nusalens.llm.ollama_client import extract_json, generate
from nusalens.llm.prompt_registry import active_version, load_prompt, render


@dataclass
class JudgeVerdict:
    grounded: bool
    score: int
    reason: str
    raw: str


def judge(question: str, context: str, answer: str, version: str | None = None,
          model: str | None = None) -> JudgeVerdict:
    selected_version = version or active_version("judge")
    if selected_version == "v5":
        prompt = load_prompt("judge", selected_version)
        analysis_prompt = prompt["template"].format(question=question, context=context, answer=answer)
        analysis_raw = generate(
            model or settings.judge_model, analysis_prompt, temperature=0.0, num_predict=400
        )
        analysis_obj = extract_json(analysis_raw) or {}
        analysis = str(analysis_obj.get("analysis", analysis_raw)).strip()
        try:
            score = int(analysis_obj.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0

        decision_prompt = prompt["decision_template"].format(analysis=analysis)
        decision_raw = generate(
            model or settings.judge_model, decision_prompt, temperature=0.0, num_predict=80
        )
        decision_obj = extract_json(decision_raw) or {}
        return JudgeVerdict(
            grounded=decision_obj.get("grounded") is True,
            score=score,
            reason=analysis,
            raw=f"analysis={analysis_raw}\ndecision={decision_raw}",
        )

    prompt = render("judge", version, question=question, context=context, answer=answer)
    raw = generate(model or settings.judge_model, prompt["rendered"], temperature=0.0, num_predict=300)
    obj = extract_json(raw) or {}
    if "label" in obj:
        grounded = obj.get("label") in ("GROUNDED", "HONEST_DECLINE")
    else:
        grounded = bool(obj.get("grounded", False))
    return JudgeVerdict(
        grounded=grounded,
        score=int(obj.get("score", 0) or 0),
        reason=str(obj.get("reason", "")),
        raw=raw,
    )
