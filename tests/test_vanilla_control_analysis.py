from deltaomni.vanilla_control_analysis import analyze


def test_analysis_requires_matched_controls_and_reports_paired_flips() -> None:
    rows = []
    outcomes = {
        ("source-a", "q1"): [1, 0, 1, 0],
        ("source-a", "q2"): [0, 0, 1, 1],
        ("source-b", "q1"): [1, 1, 1, 1],
    }
    controls = ("multimodal", "text_only", "video_only", "audio_only")
    for (source, question), values in outcomes.items():
        for control, correct in zip(controls, values, strict=True):
            rows.append(
                {
                    "task": "nextqa_multiple_choice",
                    "source_id": source,
                    "question_id": question,
                    "control": control,
                    "correct": correct,
                }
            )

    result = analyze(rows, seed=1, bootstrap_samples=100)

    assert result["controls"]["video_only"]["accuracy"] == 1.0
    comparison = result["comparisons"]["video_only_minus_multimodal"]
    assert comparison["left_only_correct"] == 1
    assert comparison["right_only_correct"] == 0
