import json
from pathlib import Path

import torch

from deltaomni.streaming_sequence import (
    CommitHead,
    SectionRef,
    StreamingSequence,
    build_sequences,
    commit_loss,
)


def _section(index: int, updates: int) -> SectionRef:
    return SectionRef(
        source_id=f"source-{index}",
        source_group_id=f"group-{index}",
        cache_path=Path(f"/cache/{index}.pt"),
        delta_updates=updates,
        captions=5,
    )


def test_streaming_timeline_repeats_commit_then_full_refresh() -> None:
    sequence = StreamingSequence(
        sequence_id="train:0",
        split="train",
        sections=(_section(0, 2), _section(1, 3), _section(2, 1)),
    )

    commit, refresh, elapsed, section = sequence.timeline()

    assert commit.tolist() == [0, 1, 0, 0, 1, 1]
    assert refresh.tolist() == [True, False, True, False, False, True]
    assert elapsed.tolist() == [1, 2, 1, 2, 3, 1]
    assert section.tolist() == [0, 0, 1, 1, 1, 2]


def test_commit_head_resets_previous_section_state_and_receives_gradients() -> None:
    torch.manual_seed(0)
    head = CommitHead(delta_width=4, hidden_width=8)
    deltas = torch.randn(1, 5, 4)
    changed = deltas.clone()
    changed[:, :2] += 100
    elapsed = torch.tensor([[1.0, 2.0, 1.0, 2.0, 3.0]])
    refresh = torch.tensor([[True, False, True, False, False]])
    valid = torch.ones_like(refresh)

    logits = head(deltas, elapsed, refresh, valid)
    changed_logits = head(changed, elapsed, refresh, valid)
    loss = commit_loss(
        logits,
        torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0]]),
        valid,
        positive_weight=2.0,
    )
    loss.backward()

    assert torch.allclose(logits[:, 2:], changed_logits[:, 2:])
    assert head.output.weight.grad is not None
    assert torch.isfinite(loss)


def test_sequence_builder_requires_multiple_sections_and_tracks_remainder(
    tmp_path: Path,
) -> None:
    records = [
        {
            "source_id": f"source-{index}",
            "source_group_id": f"group-{index}",
            "cache_path": f"/cache/{index}.pt",
            "delta_updates": 10 + index,
            "captions": 5,
        }
        for index in range(7)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"train": records}}))

    sequences, discarded = build_sequences(manifest, sections_per_sequence=3, seed=42)

    assert len(sequences["train"]) == 2
    assert all(len(sequence.sections) == 3 for sequence in sequences["train"])
    assert discarded == {"train": 1}
