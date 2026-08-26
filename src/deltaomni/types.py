from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"

    @property
    def short_name(self) -> str:
        return {self.AUDIO: "A", self.VIDEO: "V", self.IMAGE: "I"}[self]


class EventKind(StrEnum):
    FULL = "full"
    DELTA = "delta"
    CAPTION = "caption"


@dataclass(frozen=True)
class StreamEvent:
    kind: EventKind
    modality: Modality
    timestamp_seconds: float
    section_start_seconds: float
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def open_token(self) -> str:
        suffix = self.modality.short_name
        if self.kind is EventKind.FULL:
            return f"<FULL_{suffix}>"
        if self.kind is EventKind.DELTA:
            return f"<DELTA_{suffix}>"
        return f"<CAPTION_D_{suffix}>"

    @property
    def close_token(self) -> str:
        return self.open_token.replace("<", "</", 1)

    def render(self) -> str:
        if self.kind is EventKind.CAPTION:
            return f"{self.open_token}{self.text or ''}{self.close_token}"
        return self.open_token

