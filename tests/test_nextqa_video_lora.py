from pathlib import Path

import torch

from deltaomni.nextqa_video_lora import (
    InterfaceConfig,
    VideoDeltaAdapter,
    VideoQADataset,
    load_config,
)


def test_scaled_configs_use_one_seed_and_variable_horizons() -> None:
    smoke = load_config(Path("configs/nextqa_video_lora_smoke.yaml"))
    full = load_config(Path("configs/nextqa_video_lora.yaml"))

    assert smoke.seed == full.seed == 42
    assert smoke.interface.max_delta_updates == full.interface.max_delta_updates == 29
    assert smoke.training.max_steps == 20
    assert full.training.max_steps == 1000
    assert full.evaluation.validation_examples == 1024


def test_video_delta_adapter_accepts_variable_horizons() -> None:
    adapter = VideoDeltaAdapter(
        InterfaceConfig(
            delta_width=4,
            hidden_width=8,
            max_delta_updates=11,
            max_prompt_tokens=16,
            max_new_tokens=4,
            system_prompt="fixture",
        )
    )
    first = torch.randn(64, 8)

    anchors, short = adapter(first, torch.randn(4, 1, 4))
    _, long = adapter(first, torch.randn(10, 1, 4))

    assert anchors.shape == (64, 8)
    assert short.shape == (4, 8)
    assert long.shape == (10, 8)


def test_cross_source_control_matches_horizon(tmp_path: Path) -> None:
    records = []
    for index in range(3):
        path = tmp_path / f"source-{index}.pt"
        deltas = 4 if index < 2 else 7
        torch.save(
            {
                "source_id": f"source-{index}",
                "video_first": torch.zeros(64, 8, dtype=torch.float16),
                "video_deltas": torch.full((deltas, 1, 4), float(index), dtype=torch.float16),
                "qa": [
                    {
                        "question_id": "q1",
                        "question_type": "TC",
                        "question": "What happened?",
                        "choices": ["a", "b", "c", "d", "e"],
                        "answer_index": 0,
                    }
                ],
            },
            path,
        )
        records.append({"cache_path": str(path)})
    data = VideoQADataset({"splits": {"validation": records}}, "validation", 8)

    donors = data.cross_source_donors(2)

    for index, donor in enumerate(donors.tolist()):
        assert data.item(index)["source_id"] != data.item(donor)["source_id"]
        assert (
            data.item(index, delta_index=donor)["video_deltas"].shape
            == data.item(index)["video_deltas"].shape
        )
