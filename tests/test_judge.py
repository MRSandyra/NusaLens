from nusalens.llm import judge as judge_module
from nusalens.llm.judge import judge


def test_judge_grounded_true_for_a_verbatim_supported_claim():
    context = "[c1] Rutenium adalah sebuah unsur kimia dengan lambang Ru dan nomor atom 44."
    verdict = judge("Apa lambang kimia rutenium?", context, "Lambang kimia rutenium adalah Ru.", version="v1")
    assert verdict.grounded is True
    assert 1 <= verdict.score <= 5


def test_judge_grounded_false_for_a_fabricated_claim():
    context = "[c1] Rutenium adalah sebuah unsur kimia dengan lambang Ru dan nomor atom 44."
    verdict = judge("Siapa presiden pertama Indonesia?", context, "Presiden pertama Indonesia adalah Soekarno.", version="v1")
    assert verdict.grounded is False


def test_judge_v4_label_maps_honest_decline_to_grounded_true():
    context = "[c1] Rutenium adalah sebuah unsur kimia dengan lambang Ru dan nomor atom 44."
    verdict = judge(
        "Berapa jumlah penduduk Gondor?", context,
        "Tidak ada data karena Gondor adalah negara fiksi.", version="v4",
    )
    assert verdict.grounded is True


def test_judge_v5_makes_decision_after_separate_evidence_analysis(monkeypatch):
    responses = iter([
        '{"analysis": "Jawaban mengarang nama presiden yang tidak ada di konteks.", "score": 1}',
        '{"grounded": false}',
    ])
    monkeypatch.setattr(judge_module, "generate", lambda *args, **kwargs: next(responses))

    verdict = judge("Siapa presiden pertama Indonesia?", "[c1] Rutenium berlambang Ru.",
                    "Presiden pertama Indonesia adalah Soekarno.", version="v5")

    assert verdict.grounded is False
    assert verdict.score == 1
    assert "mengarang" in verdict.reason
