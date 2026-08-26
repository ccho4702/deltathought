from deltaomni.report import render_report


def test_dark_report_contains_qa_contract_and_metrics() -> None:
    summary = {
        "run_id": "run",
        "initial_validation_losses": {
            "total": 2.0,
            "reconstruction": 1.0,
            "trigger": 1.0,
            "caption": 1.0,
            "length": 1.0,
        },
        "final_validation_losses": {
            "total": 1.0,
            "reconstruction": 0.5,
            "trigger": 0.5,
            "caption": 0.5,
            "length": 0.5,
        },
        "interleaving": "<FULL_A> <DELTA_A>",
    }
    verification = {
        "passed": True,
        "qa_contract": {"question": "A new question?"},
        "metrics": {
            "caption_exact": 1.0,
            "trigger": {"f1": 1.0},
            "final_full_only_qa_accuracy": 0.5,
            "caption_history_qa_accuracy": 0.9,
            "full_plus_caption_qa_accuracy": 1.0,
        },
        "checks": {"one": True},
    }
    provenance = {"approved": ["model"], "blocked": ["dataset"]}

    rendered = render_report(summary, verification, provenance)

    assert "--bg:#0b0f14" in rendered
    assert "A new question?" in rendered
    assert "Full + caption QA" in rendered

