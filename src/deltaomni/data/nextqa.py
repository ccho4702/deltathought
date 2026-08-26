from __future__ import annotations

from typing import Any

from deltaomni.data.schema import (
    CanonicalEpisode,
    CaptionBundle,
    MediaAsset,
    MediaBundle,
    ProvenanceRecord,
    QAAnnotation,
    TextBundle,
    temporal_grid,
)
from deltaomni.provenance import require_approved
from deltaomni.types import Modality


def build_episode(
    source_id: str,
    qa_rows: list[dict[str, Any]],
    media: MediaAsset,
    *,
    split: str,
    dataset_revision: str,
    chunk_seconds: float,
    provenance_report: dict[str, Any],
) -> CanonicalEpisode:
    require_approved(provenance_report, ["nextqa_annotations"])
    if media.duration_seconds is None:
        raise ValueError("NExT-QA video requires duration")
    duration = media.duration_seconds
    final_qa = []
    for row in qa_rows:
        choices = tuple(str(row[key]) for key in ("a0", "a1", "a2", "a3", "a4"))
        answer_field = row["answer"]
        answer = choices[int(answer_field)] if str(answer_field).isdigit() else str(answer_field)
        final_qa.append(
            QAAnnotation(
                question_id=str(row["qid"]),
                question=str(row["question"]),
                answer=answer,
                choices=choices,
                answer_index=int(answer_field) if str(answer_field).isdigit() else None,
                question_type=(str(row["type"]) if row.get("type") else None),
                required_modalities=(Modality.VIDEO,),
                evidence_spans=None,
                annotation_origin="human_nextqa",
                independent_from_captions=True,
            )
        )
    episode = CanonicalEpisode(
        episode_id=f"nextqa:{split}:{source_id}",
        dataset="nextqa",
        dataset_revision=dataset_revision,
        split=split,
        source_id=source_id,
        source_group_id=f"nextqa:{source_id}",
        media=MediaBundle(image=None, video=media, audio=None),
        duration_seconds=duration,
        temporal_blocks=temporal_grid(duration, chunk_seconds),
        captions=CaptionBundle(image=None, video=None, audio=None, joint=None),
        text=TextBundle(transcript=None, subtitle=None, ocr=None),
        events=None,
        qa=tuple(final_qa),
        provenance=ProvenanceRecord(resource_name="nextqa_annotations"),
    )
    episode.validate_for_independent_qa()
    return episode
