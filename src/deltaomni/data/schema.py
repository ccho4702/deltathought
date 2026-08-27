from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deltaomni.types import Modality

SCHEMA_VERSION = "deltaomni.episode.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "episode_id",
        "dataset",
        "dataset_revision",
        "split",
        "source_id",
        "source_group_id",
        "media",
        "duration_seconds",
        "temporal_blocks",
        "captions",
        "text",
        "events",
        "qa",
        "provenance",
        "metadata",
    }
)


@dataclass(frozen=True)
class MediaAsset:
    path: Path
    sha256: str
    duration_seconds: float | None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: int | None = None
    channels: int | None = None


@dataclass(frozen=True)
class MediaBundle:
    image: MediaAsset | None
    video: MediaAsset | None
    audio: MediaAsset | None


@dataclass(frozen=True)
class TemporalBlock:
    block_index: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class CaptionAnnotation:
    caption_id: str
    scope: str
    text: str
    start_seconds: float | None
    end_seconds: float | None
    commit_seconds: float | None
    language: str | None
    annotation_origin: str
    timing_origin: str | None
    independent_from_qa: bool | None


@dataclass(frozen=True)
class CaptionBundle:
    image: tuple[CaptionAnnotation, ...] | None
    video: tuple[CaptionAnnotation, ...] | None
    audio: tuple[CaptionAnnotation, ...] | None
    joint: tuple[CaptionAnnotation, ...] | None


@dataclass(frozen=True)
class EventAnnotation:
    event_id: str
    modality: Modality
    start_seconds: float
    end_seconds: float
    label: str
    label_id: str | None
    annotation_origin: str


@dataclass(frozen=True)
class TimedTextAnnotation:
    text_id: str
    kind: str
    text: str
    start_seconds: float | None
    end_seconds: float | None
    language: str | None
    annotation_origin: str


@dataclass(frozen=True)
class TextBundle:
    transcript: tuple[TimedTextAnnotation, ...] | None
    subtitle: tuple[TimedTextAnnotation, ...] | None
    ocr: tuple[TimedTextAnnotation, ...] | None


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    text: str


@dataclass(frozen=True)
class QAAnnotation:
    question_id: str
    question: str
    answer: str
    choices: tuple[str, ...] | None
    answer_index: int | None
    question_type: str | None
    required_modalities: tuple[Modality, ...] | None
    evidence_spans: tuple[tuple[float, float], ...] | None
    annotation_origin: str
    independent_from_captions: bool | None
    acceptable_answers: tuple[str, ...] | None = None
    dialogue_history: tuple[DialogueTurn, ...] | None = None
    turn_index: int | None = None


@dataclass(frozen=True)
class ProvenanceRecord:
    resource_name: str
    source_url: str | None = None
    license_record: Path | None = None
    annotation_path: Path | None = None
    annotation_sha256: str | None = None
    preprocessing_config_sha256: str | None = None
    code_revision: str | None = None
    processed_at_utc: str | None = None


@dataclass(frozen=True)
class CanonicalEpisode:
    episode_id: str
    dataset: str
    dataset_revision: str
    split: str
    source_id: str
    source_group_id: str
    media: MediaBundle
    duration_seconds: float | None
    temporal_blocks: tuple[TemporalBlock, ...] | None
    captions: CaptionBundle
    text: TextBundle
    events: tuple[EventAnnotation, ...] | None
    qa: tuple[QAAnnotation, ...] | None
    provenance: ProvenanceRecord
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        identifiers = {
            "episode_id": self.episode_id,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "source_id": self.source_id,
            "source_group_id": self.source_group_id,
        }
        for name, value in identifiers.items():
            if not value.strip():
                raise ValueError(f"Empty canonical identifier: {name}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")

        assets = {
            Modality.IMAGE: self.media.image,
            Modality.VIDEO: self.media.video,
            Modality.AUDIO: self.media.audio,
        }
        if not any(asset is not None for asset in assets.values()):
            raise ValueError("Canonical episode requires at least one media asset")
        temporal_assets = []
        for modality, asset in assets.items():
            if asset is None:
                continue
            if not str(asset.path) or not _SHA256.fullmatch(asset.sha256):
                raise ValueError(f"Invalid {modality.value} path or SHA-256")
            if asset.duration_seconds is not None and asset.duration_seconds <= 0:
                raise ValueError(f"Invalid {modality.value} duration")
            if modality is not Modality.IMAGE:
                if asset.duration_seconds is None:
                    raise ValueError(f"Temporal {modality.value} asset requires duration")
                temporal_assets.append(asset)
            for name in ("width", "height", "sample_rate", "channels"):
                value = getattr(asset, name)
                if value is not None and value <= 0:
                    raise ValueError(f"Invalid {modality.value} {name}")
            if asset.fps is not None and asset.fps <= 0:
                raise ValueError(f"Invalid {modality.value} fps")

        if temporal_assets:
            if self.duration_seconds is None or self.duration_seconds <= 0:
                raise ValueError("Temporal episode requires positive duration_seconds")
            if any(
                asset.duration_seconds is not None
                and asset.duration_seconds > self.duration_seconds + 1e-6
                for asset in temporal_assets
            ):
                raise ValueError("Media duration exceeds episode duration")
            self._validate_blocks()
        elif self.duration_seconds is not None or self.temporal_blocks is not None:
            raise ValueError("Image-only episode must use null duration and temporal_blocks")

        caption_groups = {
            "image": self.captions.image,
            "video": self.captions.video,
            "audio": self.captions.audio,
            "joint": self.captions.joint,
        }
        for scope, captions in caption_groups.items():
            if captions is None:
                continue
            for caption in captions:
                self._validate_caption(caption, scope)
        text_groups = {
            "transcript": self.text.transcript,
            "subtitle": self.text.subtitle,
            "ocr": self.text.ocr,
        }
        for kind, text_items in text_groups.items():
            if text_items is None:
                continue
            for annotation in text_items:
                self._validate_text(annotation, kind)
        if self.events is not None:
            for event in self.events:
                self._validate_event(event)
        if self.qa is not None:
            for qa in self.qa:
                self._validate_qa(qa)

    def _validate_blocks(self) -> None:
        if not self.temporal_blocks:
            raise ValueError("Temporal episode requires non-empty temporal_blocks")
        previous_end = 0.0
        for expected_index, block in enumerate(self.temporal_blocks):
            if block.block_index != expected_index:
                raise ValueError("Temporal block indices must be consecutive")
            if abs(block.start_seconds - previous_end) > 1e-6:
                raise ValueError("Temporal blocks must be contiguous and start at zero")
            if block.end_seconds <= block.start_seconds:
                raise ValueError("Temporal block must have positive duration")
            previous_end = block.end_seconds
        if self.duration_seconds is None or abs(previous_end - self.duration_seconds) > 1e-6:
            raise ValueError("Temporal blocks must cover the complete episode")

    def _validate_caption(self, caption: CaptionAnnotation, expected_scope: str) -> None:
        if caption.scope != expected_scope:
            raise ValueError(f"Caption scope mismatch: {caption.caption_id}")
        if not caption.caption_id.strip() or not caption.text.strip():
            raise ValueError("Caption ID and text must be non-empty")
        has_start = caption.start_seconds is not None
        has_end = caption.end_seconds is not None
        if has_start != has_end:
            raise ValueError(f"Caption span must provide both bounds: {caption.caption_id}")
        if has_start and has_end:
            assert caption.start_seconds is not None and caption.end_seconds is not None
            self._validate_span(caption.start_seconds, caption.end_seconds, caption.caption_id)
            if caption.commit_seconds is not None and not (
                caption.end_seconds
                <= caption.commit_seconds
                <= (self.duration_seconds or caption.commit_seconds)
            ):
                raise ValueError(f"Invalid caption commit: {caption.caption_id}")
        elif caption.commit_seconds is not None:
            raise ValueError(f"Caption without span cannot have commit: {caption.caption_id}")

    def _validate_event(self, event: EventAnnotation) -> None:
        if not event.event_id.strip() or not event.label.strip():
            raise ValueError("Event ID and label must be non-empty")
        self._validate_span(event.start_seconds, event.end_seconds, event.event_id)

    def _validate_text(self, annotation: TimedTextAnnotation, expected_kind: str) -> None:
        if annotation.kind != expected_kind:
            raise ValueError(f"Timed text kind mismatch: {annotation.text_id}")
        if not annotation.text_id.strip() or not annotation.text.strip():
            raise ValueError("Timed text ID and text must be non-empty")
        has_start = annotation.start_seconds is not None
        has_end = annotation.end_seconds is not None
        if has_start != has_end:
            raise ValueError(f"Timed text must provide both bounds: {annotation.text_id}")
        if has_start and has_end:
            assert annotation.start_seconds is not None and annotation.end_seconds is not None
            self._validate_span(
                annotation.start_seconds,
                annotation.end_seconds,
                annotation.text_id,
            )

    def _validate_span(self, start: float, end: float, identifier: str) -> None:
        if self.duration_seconds is None or not 0 <= start < end <= self.duration_seconds:
            raise ValueError(f"Invalid temporal span: {identifier}")

    def _validate_qa(self, qa: QAAnnotation) -> None:
        if not qa.question_id.strip() or not qa.question.strip() or not qa.answer.strip():
            raise ValueError("QA ID, question, and answer must be non-empty")
        if qa.choices is None:
            if qa.answer_index is not None:
                raise ValueError(f"Open QA cannot have answer_index: {qa.question_id}")
        else:
            if not qa.choices or any(not choice.strip() for choice in qa.choices):
                raise ValueError(f"QA choices must be non-empty: {qa.question_id}")
            if qa.answer_index is None or not 0 <= qa.answer_index < len(qa.choices):
                raise ValueError(f"Invalid answer_index: {qa.question_id}")
            if qa.answer != qa.choices[qa.answer_index]:
                raise ValueError(f"Answer does not match indexed choice: {qa.question_id}")
        if qa.evidence_spans is not None:
            for index, (start, end) in enumerate(qa.evidence_spans):
                self._validate_span(start, end, f"{qa.question_id}:evidence:{index}")
        if qa.acceptable_answers is not None:
            if not qa.acceptable_answers or any(
                not answer.strip() for answer in qa.acceptable_answers
            ):
                raise ValueError(f"Invalid acceptable answers: {qa.question_id}")
            if qa.answer not in qa.acceptable_answers:
                raise ValueError(f"Primary answer is not acceptable: {qa.question_id}")
        if qa.dialogue_history is not None:
            for turn in qa.dialogue_history:
                if turn.role not in {"system", "user", "assistant"} or not turn.text.strip():
                    raise ValueError(f"Invalid dialogue history: {qa.question_id}")
        if qa.turn_index is not None and qa.turn_index < 0:
            raise ValueError(f"Invalid dialogue turn index: {qa.question_id}")

    def validate_for_caption_training(self) -> None:
        self.validate()
        groups = (
            self.captions.image,
            self.captions.video,
            self.captions.audio,
            self.captions.joint,
        )
        if not any(group for group in groups if group is not None):
            raise ValueError("Caption training requires at least one caption")

    def validate_for_qa_training(self) -> None:
        self.validate()
        if not self.qa:
            raise ValueError("QA training requires at least one QA annotation")

    def validate_for_independent_qa(self) -> None:
        self.validate_for_qa_training()
        assert self.qa is not None
        if not all(qa.independent_from_captions is True for qa in self.qa):
            raise ValueError("Independent QA requires independently authored annotations")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, **_json_value(asdict(self))}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanonicalEpisode:
        if value.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported canonical schema: {value.get('schema')}")
        if set(value) != TOP_LEVEL_FIELDS:
            raise ValueError("Canonical episode has missing or unknown top-level fields")

        def asset(raw: dict[str, Any] | None) -> MediaAsset | None:
            if raw is None:
                return None
            return MediaAsset(
                path=Path(raw["path"]),
                **{
                    key: raw.get(key)
                    for key in (
                        "sha256", "duration_seconds", "mime_type", "width", "height", "fps",
                        "sample_rate", "channels",
                    )
                },
            )

        raw_media = value["media"]
        raw_captions = value["captions"]
        raw_provenance = dict(value["provenance"])
        for key in ("license_record", "annotation_path"):
            if raw_provenance.get(key) is not None:
                raw_provenance[key] = Path(raw_provenance[key])
        episode = cls(
            episode_id=value["episode_id"],
            dataset=value["dataset"],
            dataset_revision=value["dataset_revision"],
            split=value["split"],
            source_id=value["source_id"],
            source_group_id=value["source_group_id"],
            media=MediaBundle(
                image=asset(raw_media["image"]),
                video=asset(raw_media["video"]),
                audio=asset(raw_media["audio"]),
            ),
            duration_seconds=value["duration_seconds"],
            temporal_blocks=(
                None if value["temporal_blocks"] is None
                else tuple(TemporalBlock(**raw) for raw in value["temporal_blocks"])
            ),
            captions=CaptionBundle(
                **{
                    scope: None if raw_captions[scope] is None
                    else tuple(CaptionAnnotation(**raw) for raw in raw_captions[scope])
                    for scope in ("image", "video", "audio", "joint")
                }
            ),
            text=TextBundle(
                **{
                    kind: None if value["text"][kind] is None
                    else tuple(
                        TimedTextAnnotation(**raw) for raw in value["text"][kind]
                    )
                    for kind in ("transcript", "subtitle", "ocr")
                }
            ),
            events=(
                None if value["events"] is None
                else tuple(
                    EventAnnotation(**{**raw, "modality": Modality(raw["modality"])})
                    for raw in value["events"]
                )
            ),
            qa=(
                None if value["qa"] is None
                else tuple(_qa_from_dict(raw) for raw in value["qa"])
            ),
            provenance=ProvenanceRecord(**raw_provenance),
            metadata=dict(value["metadata"]),
        )
        episode.validate()
        return episode


def _qa_from_dict(raw: dict[str, Any]) -> QAAnnotation:
    return QAAnnotation(
        **{
            **raw,
            "choices": None if raw["choices"] is None else tuple(raw["choices"]),
            "required_modalities": (
                None if raw["required_modalities"] is None
                else tuple(Modality(item) for item in raw["required_modalities"])
            ),
            "evidence_spans": (
                None if raw["evidence_spans"] is None
                else tuple(tuple(span) for span in raw["evidence_spans"])
            ),
            "acceptable_answers": (
                None
                if raw["acceptable_answers"] is None
                else tuple(raw["acceptable_answers"])
            ),
            "dialogue_history": (
                None
                if raw["dialogue_history"] is None
                else tuple(DialogueTurn(**turn) for turn in raw["dialogue_history"])
            ),
        }
    )


def temporal_grid(duration_seconds: float, chunk_seconds: float) -> tuple[TemporalBlock, ...]:
    if duration_seconds <= 0 or chunk_seconds <= 0:
        raise ValueError("duration_seconds and chunk_seconds must be positive")
    blocks = []
    start = 0.0
    index = 0
    while start < duration_seconds:
        end = min(round(start + chunk_seconds, 6), duration_seconds)
        blocks.append(TemporalBlock(index, start, end))
        start = end
        index += 1
    return tuple(blocks)


def write_jsonl(path: Path, episodes: list[CanonicalEpisode]) -> None:
    seen = set()
    records = []
    for episode in episodes:
        episode.validate()
        if episode.episode_id in seen:
            raise ValueError(f"Duplicate episode_id: {episode.episode_id}")
        seen.add(episode.episode_id)
        records.append(json.dumps(episode.to_dict(), sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(records) + ("\n" if records else ""))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[CanonicalEpisode]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[CanonicalEpisode]:
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            episode = CanonicalEpisode.from_dict(json.loads(line))
            if episode.episode_id in seen:
                raise ValueError(
                    f"Duplicate episode_id at line {line_number}: {episode.episode_id}"
                )
            seen.add(episode.episode_id)
            yield episode


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Modality):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value
