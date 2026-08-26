import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from deltaomni.data.schema import (
    SCHEMA_VERSION,
    TOP_LEVEL_FIELDS,
    CanonicalEpisode,
    CaptionAnnotation,
    CaptionBundle,
    MediaAsset,
    MediaBundle,
    ProvenanceRecord,
    QAAnnotation,
    TextBundle,
    read_jsonl,
    temporal_grid,
    write_jsonl,
)
from deltaomni.types import Modality

SHA256 = "b" * 64


def _episode() -> CanonicalEpisode:
    duration = 3.0
    return CanonicalEpisode(
        episode_id="fixture:validation:one",
        dataset="fixture",
        dataset_revision="v1",
        split="validation",
        source_id="one",
        source_group_id="fixture-source:one",
        media=MediaBundle(
            image=None,
            video=MediaAsset(
                Path("/immutable/one.mp4"),
                SHA256,
                duration,
                mime_type="video/mp4",
                width=640,
                height=480,
                fps=30.0,
            ),
            audio=None,
        ),
        duration_seconds=duration,
        temporal_blocks=temporal_grid(duration, 2.0),
        captions=CaptionBundle(
            image=None,
            video=(
                CaptionAnnotation(
                    "caption-1",
                    "video",
                    "A person moves a cup.",
                    0.0,
                    3.0,
                    3.0,
                    "en",
                    "human",
                    "human_temporal",
                    True,
                ),
            ),
            audio=None,
            joint=None,
        ),
        text=TextBundle(transcript=None, subtitle=None, ocr=None),
        events=None,
        qa=(
            QAAnnotation(
                "q1",
                "What moved?",
                "A cup",
                ("A plate", "A cup"),
                1,
                "object",
                (Modality.VIDEO,),
                ((0.0, 3.0),),
                "human",
                True,
            ),
        ),
        provenance=ProvenanceRecord("fixture"),
        metadata={"note": "round trip"},
    )


def test_v2_round_trip_preserves_null_and_empty_semantics(tmp_path: Path) -> None:
    episode = _episode()
    episode.validate()
    path = tmp_path / "episodes.jsonl"

    write_jsonl(path, [episode])
    loaded = read_jsonl(path)

    assert loaded == [episode]
    record = json.loads(path.read_text().strip())
    assert record["schema"] == SCHEMA_VERSION
    assert record["media"]["image"] is None
    assert record["captions"]["audio"] is None
    assert record["events"] is None


def test_serialized_v2_episode_satisfies_published_json_schema() -> None:
    schema = json.loads(
        Path("schemas/deltaomni.episode.v2.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)

    Draft202012Validator(schema).validate(_episode().to_dict())


def test_every_v2_record_has_exactly_the_same_top_level_fields() -> None:
    record = _episode().to_dict()

    assert set(record) == TOP_LEVEL_FIELDS
    assert set(record["media"]) == {"image", "video", "audio"}
    assert set(record["captions"]) == {"image", "video", "audio", "joint"}
    assert set(record["text"]) == {"transcript", "subtitle", "ocr"}


def test_v2_rejects_unknown_fields_and_inconsistent_choices() -> None:
    record = _episode().to_dict()
    record["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unknown"):
        CanonicalEpisode.from_dict(record)

    episode = _episode()
    assert episode.qa is not None
    invalid_qa = QAAnnotation(**{**episode.qa[0].__dict__, "answer_index": 0})
    invalid = CanonicalEpisode(**{**episode.__dict__, "qa": (invalid_qa,)})
    with pytest.raises(ValueError, match="does not match"):
        invalid.validate()


def test_null_means_unavailable_while_empty_means_annotated_without_items() -> None:
    episode = _episode()
    empty_events = CanonicalEpisode(**{**episode.__dict__, "events": ()})

    empty_events.validate()
    assert episode.to_dict()["events"] is None
    assert empty_events.to_dict()["events"] == []
