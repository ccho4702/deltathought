from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from deltaomni.data.schema import (
    CanonicalEpisode,
    CaptionAnnotation,
    CaptionBundle,
    EventAnnotation,
    MediaAsset,
    MediaBundle,
    ProvenanceRecord,
    TextBundle,
    temporal_grid,
)
from deltaomni.provenance import require_approved
from deltaomni.types import Modality


@dataclass(frozen=True)
class StrongEvent:
    clip_id: str
    start_seconds: float
    end_seconds: float
    class_id: str


@dataclass(frozen=True)
class InvalidStrongEvent:
    line_number: int
    event: StrongEvent
    reason: str


@dataclass(frozen=True)
class StrongEventFile:
    events: dict[str, tuple[StrongEvent, ...]]
    invalid_events: tuple[InvalidStrongEvent, ...]
    data_rows: int


def inspect_tsv(path: Path) -> StrongEventFile:
    grouped: dict[str, list[StrongEvent]] = defaultdict(list)
    invalid_events = []
    data_rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields[0] == "segment_id":
            continue
        data_rows += 1
        if len(fields) < 4:
            raise ValueError(f"Malformed AudioSet Strong row {line_number}")
        event = StrongEvent(fields[0], float(fields[1]), float(fields[2]), fields[3])
        if not 0 <= event.start_seconds < event.end_seconds:
            invalid_events.append(
                InvalidStrongEvent(line_number, event, "start must be strictly before end")
            )
            continue
        grouped[event.clip_id].append(event)
    return StrongEventFile(
        events={
            clip_id: tuple(
                sorted(events, key=lambda event: (event.end_seconds, event.start_seconds))
            )
            for clip_id, events in grouped.items()
        },
        invalid_events=tuple(invalid_events),
        data_rows=data_rows,
    )


def parse_tsv(path: Path) -> dict[str, tuple[StrongEvent, ...]]:
    inspected = inspect_tsv(path)
    if inspected.invalid_events:
        first = inspected.invalid_events[0]
        raise ValueError(f"Invalid AudioSet Strong bounds at row {first.line_number}")
    return inspected.events


def build_episode(
    clip_id: str,
    events: tuple[StrongEvent, ...],
    labels: dict[str, str],
    media: MediaAsset,
    *,
    split: str,
    dataset_revision: str,
    chunk_seconds: float,
    provenance_report: dict[str, object],
    timeline_duration_seconds: float | None = None,
) -> CanonicalEpisode:
    require_approved(provenance_report, ["audioset_strong_annotations"])
    if media.duration_seconds is None:
        raise ValueError("AudioSet audio requires duration")
    duration = timeline_duration_seconds or media.duration_seconds
    if duration < media.duration_seconds:
        raise ValueError("AudioSet timeline cannot be shorter than media")
    captions = tuple(
        CaptionAnnotation(
            caption_id=f"{clip_id}:{index}",
            scope="audio",
            text=f"Sound event: {labels[event.class_id]}.",
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
            commit_seconds=event.end_seconds,
            language="en",
            annotation_origin="deterministic_label_verbalization",
            timing_origin="human_strong",
            independent_from_qa=None,
        )
        for index, event in enumerate(events)
    )
    event_annotations = tuple(
        EventAnnotation(
            event_id=f"{clip_id}:{index}",
            modality=Modality.AUDIO,
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
            label=labels[event.class_id],
            label_id=event.class_id,
            annotation_origin="human_strong",
        )
        for index, event in enumerate(events)
    )
    episode = CanonicalEpisode(
        episode_id=f"audioset-strong:{split}:{clip_id}",
        dataset="audioset_strong",
        dataset_revision=dataset_revision,
        split=split,
        source_id=clip_id,
        source_group_id=f"youtube:{clip_id.rsplit('_', 1)[0]}",
        media=MediaBundle(image=None, video=None, audio=media),
        duration_seconds=duration,
        temporal_blocks=temporal_grid(duration, chunk_seconds),
        captions=CaptionBundle(image=None, video=None, audio=captions, joint=None),
        text=TextBundle(transcript=None, subtitle=None, ocr=None),
        events=event_annotations,
        qa=None,
        provenance=ProvenanceRecord(resource_name="audioset_strong_annotations"),
    )
    episode.validate()
    return episode
