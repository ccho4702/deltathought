from __future__ import annotations

from typing import Any

from deltaomni.data.schema import (
    CanonicalEpisode,
    CaptionAnnotation,
    CaptionBundle,
    MediaAsset,
    MediaBundle,
    ProvenanceRecord,
    TextBundle,
    temporal_grid,
)
from deltaomni.provenance import require_approved


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
    if media.duration_seconds is None:
        raise ValueError("Something-Something video requires duration")
    duration = media.duration_seconds
    episode = CanonicalEpisode(
        episode_id=f"something-something-v2:{split}:{source_id}",
        dataset="something_something_v2",
        dataset_revision=dataset_revision,
        split=split,
        source_id=source_id,
        source_group_id=f"something_something_v2:{source_id}",
        media=MediaBundle(image=None, video=media, audio=None),
        duration_seconds=duration,
        temporal_blocks=temporal_grid(duration, chunk_seconds),
        captions=CaptionBundle(
            image=None,
            video=(
                CaptionAnnotation(
                    caption_id=f"{source_id}:clip",
                    scope="video",
                    text=caption,
                    start_seconds=0.0,
                    end_seconds=duration,
                    commit_seconds=duration,
                    language="en",
                    annotation_origin="human_verified_action_label",
                    timing_origin="clip_end_only",
                    independent_from_qa=None,
                ),
            ),
            audio=None,
            joint=None,
        ),
        text=TextBundle(transcript=None, subtitle=None, ocr=None),
        events=None,
        qa=None,
        provenance=ProvenanceRecord(resource_name="something_something_v2"),
        metadata={"template": annotation.get("template")},
    )
    episode.validate()
    return episode
