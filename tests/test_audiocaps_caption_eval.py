from deltaomni.audiocaps_caption_eval import _checks


def test_caption_test_gate_requires_nll_and_generation_control_gaps() -> None:
    metrics = {
        "nll": {"normal": 1.0, "zero": 1.2, "shuffled": 1.3},
        "generation": {
            "normal": {"word_f1": 0.5},
            "zero": {"word_f1": 0.4},
            "shuffled": {"word_f1": 0.3},
        },
    }

    assert all(_checks(metrics, 0.01).values())
    metrics["generation"]["shuffled"]["word_f1"] = 0.5
    assert not _checks(metrics, 0.01)["word_f1_beats_shuffled"]
