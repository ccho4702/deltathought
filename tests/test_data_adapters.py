import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from deltaomni.data.audioset_strong import build_episode as build_audio_episode
from deltaomni.data.audioset_strong import inspect_tsv, parse_tsv
from deltaomni.data.nextqa import build_episode as build_qa_episode
from deltaomni.data.schema import TOP_LEVEL_FIELDS, CanonicalEpisode, MediaAsset, QAAnnotation
from deltaomni.data.something_something import build_episode as build_video_episode
from deltaomni.provenance import audit
from deltaomni.types import Modality

SHA256 = "a" * 64
SCHEMA = json.loads(
    Path("schemas/deltaomni.episode.v2.schema.json").read_text(encoding="utf-8")
)
VALIDATOR = Draft202012Validator(SCHEMA)


def _media(name: str, duration: float = 4.0) -> MediaAsset:
    return MediaAsset(Path(f"/immutable/{name}"), SHA256, duration)


def _assert_v2_shape(episode: CanonicalEpisode) -> None:
    record = episode.to_dict()
    assert set(record) == TOP_LEVEL_FIELDS
    assert set(record["media"]) == {"image", "video", "audio"}
    assert set(record["captions"]) == {"image", "video", "audio", "joint"}
    assert set(record["text"]) == {"transcript", "subtitle", "ocr"}
    VALIDATOR.validate(record)


def test_something_something_adapter_uses_clip_end_as_provisional_commit() -> None:
    provenance = audit(Path("configs/provenance.yaml"))

    episode = build_video_episode(
        {"id": "123", "label": "Moving a cup from left to right"},
        _media("123.webm"),
        split="train",
        dataset_revision="official-v2",
        chunk_seconds=1.0,
        provenance_report=provenance,
    )

    assert episode.media.video is not None
    assert episode.media.image is episode.media.audio is None
    assert [item.end_seconds for item in episode.temporal_blocks or ()] == [1, 2, 3, 4]
    assert episode.captions.video is not None
    assert episode.captions.video[0].annotation_origin == "human_verified_action_label"
    assert episode.captions.video[0].timing_origin == "clip_end_only"
    assert episode.qa is None
    _assert_v2_shape(episode)


def test_audioset_strong_adapter_preserves_human_event_times(tmp_path: Path) -> None:
    annotation = tmp_path / "strong.tsv"
    annotation.write_text(
        "segment_id\tstart_time_seconds\tend_time_seconds\tlabel\n"
        "clip\t0.500\t1.250\t/m/dog\nclip\t2.000\t3.500\t/m/speech\n",
        encoding="utf-8",
    )
    parsed = parse_tsv(annotation)

    episode = build_audio_episode(
        "clip",
        parsed["clip"],
        {"/m/dog": "Dog", "/m/speech": "Speech"},
        _media("clip.wav"),
        split="train",
        dataset_revision="2021-05",
        chunk_seconds=0.5,
        provenance_report=audit(Path("configs/provenance.yaml")),
    )

    assert episode.media.audio is not None
    assert episode.captions.audio is not None
    assert [section.commit_seconds for section in episode.captions.audio] == [1.25, 3.5]
    assert episode.captions.audio[0].text == "Sound event: Dog."
    assert episode.captions.audio[0].annotation_origin == "deterministic_label_verbalization"
    assert episode.events is not None and episode.events[0].modality is Modality.AUDIO
    _assert_v2_shape(episode)


def test_audioset_inspection_quarantines_zero_duration_event(tmp_path: Path) -> None:
    annotation = tmp_path / "strong.tsv"
    annotation.write_text(
        "segment_id\tstart_time_seconds\tend_time_seconds\tlabel\n"
        "clip\t1.000\t1.000\t/m/zero\nclip\t1.000\t2.000\t/m/valid\n",
        encoding="utf-8",
    )

    inspected = inspect_tsv(annotation)

    assert inspected.data_rows == 2
    assert len(inspected.invalid_events) == 1
    assert list(inspected.events) == ["clip"]
    with pytest.raises(ValueError, match="row 2"):
        parse_tsv(annotation)


def test_nextqa_is_independent_final_qa_without_caption_targets() -> None:
    row = {
        "qid": "q1",
        "question": "Why does the person move the cup?",
        "answer": "2",
        "a0": "To clean it",
        "a1": "To hide it",
        "a2": "To place it on the table",
        "a3": "To break it",
        "a4": "To paint it",
    }

    episode = build_qa_episode(
        "video-1",
        [row],
        _media("video-1.mp4"),
        split="validation",
        dataset_revision="official",
        chunk_seconds=1.0,
        provenance_report=audit(Path("configs/provenance.yaml")),
    )

    assert episode.captions.video is None
    assert episode.qa is not None
    assert episode.qa[0].answer == "To place it on the table"
    assert episode.qa[0].independent_from_captions
    _assert_v2_shape(episode)


def test_downstream_schema_rejects_caption_derived_qa() -> None:
    qa = QAAnnotation(
        "q",
        "What happened?",
        "A dog barked",
        None,
        None,
        None,
        (Modality.VIDEO,),
        None,
        "constructed",
        False,
    )
    episode = build_video_episode(
        {"id": "123", "label": "Moving a cup"},
        _media("123.webm"),
        split="train",
        dataset_revision="official-v2",
        chunk_seconds=1.0,
        provenance_report=audit(Path("configs/provenance.yaml")),
    )
    invalid = replace(episode, qa=(qa,))

    with pytest.raises(ValueError, match="independently"):
        invalid.validate_for_independent_qa()
