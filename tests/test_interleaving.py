from dataclasses import replace
from pathlib import Path

import torch

from deltaomni.config import load_config
from deltaomni.interleaving import StreamingDeltaEngine, render_interleaving
from deltaomni.model import DeltaCodecModel
from deltaomni.types import EventKind, Modality


def _embedding(value: float, tokens: int, dimension: int) -> torch.Tensor:
    return torch.full((1, tokens, dimension), float(value))


def test_video_commit_does_not_reset_audio_and_refreshes_video_full_embedding() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    stream_config = replace(
        config.stream,
        trigger_threshold=1.1,
        load_threshold=1_000.0,
        max_section_steps=10,
    )
    model = DeltaCodecModel(config.model, config.modalities)
    engine = StreamingDeltaEngine(model, stream_config)
    initial = {
        Modality.AUDIO: _embedding(0.0, config.model.embedding_tokens, config.model.embedding_dim),
        Modality.VIDEO: _embedding(1.0, config.model.embedding_tokens, config.model.embedding_dim),
    }
    initial_events = engine.initialize(0.0, initial)
    current = {
        Modality.AUDIO: _embedding(0.5, config.model.embedding_tokens, config.model.embedding_dim),
        Modality.VIDEO: _embedding(1.5, config.model.embedding_tokens, config.model.embedding_dim),
    }

    events = engine.step(1.0, current, force_commits={Modality.VIDEO})

    assert [(event.kind, event.modality) for event in initial_events] == [
        (EventKind.FULL, Modality.AUDIO),
        (EventKind.FULL, Modality.VIDEO),
    ]
    assert [(event.kind, event.modality) for event in events] == [
        (EventKind.DELTA, Modality.AUDIO),
        (EventKind.DELTA, Modality.VIDEO),
        (EventKind.CAPTION, Modality.VIDEO),
        (EventKind.FULL, Modality.VIDEO),
    ]
    assert engine.states[Modality.AUDIO].section_steps == 1
    assert engine.states[Modality.VIDEO].section_steps == 0
    assert torch.equal(engine.states[Modality.AUDIO].anchor, initial[Modality.AUDIO])
    assert torch.equal(engine.states[Modality.AUDIO].previous, current[Modality.AUDIO])
    assert torch.equal(engine.states[Modality.VIDEO].anchor, current[Modality.VIDEO])

    rendered = render_interleaving(initial_events + events)
    assert rendered.startswith("<FULL_A> <FULL_V> <DELTA_A> <DELTA_V> <CAPTION_D_V>")
    assert rendered.endswith("</CAPTION_D_V> <FULL_V>")


def test_modalities_can_commit_at_the_same_timestamp() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    stream_config = replace(config.stream, trigger_threshold=1.1, load_threshold=1_000.0)
    model = DeltaCodecModel(config.model, config.modalities)
    engine = StreamingDeltaEngine(model, stream_config)
    initial = {
        modality: _embedding(index, config.model.embedding_tokens, config.model.embedding_dim)
        for index, modality in enumerate(config.modalities)
    }
    engine.initialize(0.0, initial)
    current = {modality: embedding + 0.5 for modality, embedding in initial.items()}

    events = engine.step(1.0, current, force_commits=set(config.modalities))

    captions = [event for event in events if event.kind is EventKind.CAPTION]
    assert [event.modality for event in captions] == list(config.modalities)
    assert all(event.timestamp_seconds == 1.0 for event in captions)
    assert all(state.section_steps == 0 for state in engine.states.values())
