from dataclasses import replace
from pathlib import Path

import torch

from deltaomni.config import load_config
from deltaomni.train_sanity import train


def test_interrupted_run_resumes_to_identical_model(tmp_path: Path) -> None:
    base = load_config(Path("configs/sanity.yaml"))
    training = replace(
        base.training,
        examples=8,
        validation_examples=4,
        batch_size=4,
        max_steps=6,
        checkpoint_interval_steps=2,
        run_root=tmp_path / "outputs",
        log_root=tmp_path / "logs",
    )
    config = replace(base, training=training)

    interrupted = train(config, run_id="resumed", stop_after_step=3)
    resumed = train(config, run_id="resumed")
    train(config, run_id="uninterrupted")

    assert interrupted["status"] == "interrupted"
    assert resumed["status"] in {"complete", "failed_sanity"}
    resumed_state = torch.load(
        training.run_root / "resumed/checkpoints/step-000006.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    uninterrupted_state = torch.load(
        training.run_root / "uninterrupted/checkpoints/step-000006.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    assert resumed_state.keys() == uninterrupted_state.keys()
    assert all(
        torch.equal(resumed_state[name], uninterrupted_state[name]) for name in resumed_state
    )
