from __future__ import annotations

from typing import Any

from deltaomni.data.schema import (
    CanonicalEpisode,
    CaptionSection,
    MediaAsset,
    observation_grid,
)
from deltaomni.provenance import require_approved
from deltaomni.types import Modality


def build_episode(
    annotation: dict[str, Any],
    media: MediaAsset,
    *,
    split: str,
    dataset_revision: str,
    chunk_seconds: float,
    provenance_report: dict[str, Any],
) -> CanonicalEpisode:
    require_approved(provenance_report, ["something_something_v2"])
    source_id = str(annotation["id"])
    caption = str(annotation.get("label") or annotation.get("template") or "").strip()
    if not caption:
        raise ValueError(f"Something-Something record {source_id} has no action label")
    episode = CanonicalEpisode(
        episode_id=f"something-something-v2:{split}:{source_id}",
        dataset="something_something_v2",
        dataset_revision=dataset_revision,
        split=split,
        source_id=source_id,
        modality=Modality.VIDEO,
        media=media,
        observations=observation_grid(media.duration_seconds, chunk_seconds),
        sections=(
            CaptionSection(
                section_id=f"{source_id}:clip",
                start_seconds=0.0,
                end_seconds=media.duration_seconds,
                commit_seconds=media.duration_seconds,
                caption=caption,
                caption_origin="human_verified_action_label",
                timing_origin="clip_end_only",
            ),
        ),
        final_qa=(),
    )
    episode.validate()
    return episode

