from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deltaomni.types import Modality


@dataclass(frozen=True)
class MediaAsset:
    path: Path
    sha256: str
    duration_seconds: float


@dataclass(frozen=True)
class Observation:
    timestamp_seconds: float


@dataclass(frozen=True)
class CaptionSection:
    section_id: str
    start_seconds: float
    end_seconds: float
    commit_seconds: float
    caption: str
    caption_origin: str
    timing_origin: str


@dataclass(frozen=True)
class FinalQA:
    question_id: str
    question: str
    choices: tuple[str, ...]
    answer: str
    annotation_origin: str
    independent_from_captions: bool


@dataclass(frozen=True)
class CanonicalEpisode:
    episode_id: str
    dataset: str
    dataset_revision: str
    split: str
    source_id: str
    modality: Modality
    media: MediaAsset
    observations: tuple[Observation, ...]
    sections: tuple[CaptionSection, ...]
    final_qa: tuple[FinalQA, ...]

    def validate(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.media.duration_seconds <= 0 or len(self.media.sha256) != 64:
            raise ValueError("Media duration and SHA-256 must be valid")
        times = [observation.timestamp_seconds for observation in self.observations]
        if not times or times[0] != 0.0 or times != sorted(set(times)):
            raise ValueError("Observations must be unique, ordered, and start at zero")
        if times[-1] > self.media.duration_seconds:
            raise ValueError("Observation exceeds media duration")
        for section in self.sections:
            valid_bounds = (
                0
                <= section.start_seconds
                < section.end_seconds
                <= self.media.duration_seconds
            )
            if not valid_bounds:
                raise ValueError(f"Invalid section bounds: {section.section_id}")
            if not (section.end_seconds <= section.commit_seconds <= self.media.duration_seconds):
                raise ValueError(f"Invalid commit time: {section.section_id}")
            if not section.caption.strip():
                raise ValueError(f"Empty section caption: {section.section_id}")
        for qa in self.final_qa:
            if not qa.question.strip() or not qa.answer.strip():
                raise ValueError(f"Invalid final QA: {qa.question_id}")
            if qa.choices and qa.answer not in qa.choices:
                raise ValueError(f"Answer is absent from choices: {qa.question_id}")

    def validate_for_downstream(self) -> None:
        self.validate()
        if not self.final_qa:
            raise ValueError("Downstream evaluation requires final QA")
        if not all(qa.independent_from_captions for qa in self.final_qa):
            raise ValueError("Final QA must be authored independently from caption targets")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["modality"] = self.modality.value
        value["media"]["path"] = str(self.media.path)
        value["schema"] = "deltaomni.episode.v1"
        return value


def observation_grid(duration_seconds: float, chunk_seconds: float) -> tuple[Observation, ...]:
    if duration_seconds <= 0 or chunk_seconds <= 0:
        raise ValueError("duration_seconds and chunk_seconds must be positive")
    times = [0.0]
    current = chunk_seconds
    while current < duration_seconds:
        times.append(round(current, 6))
        current += chunk_seconds
    if times[-1] != duration_seconds:
        times.append(duration_seconds)
    return tuple(Observation(timestamp) for timestamp in times)
