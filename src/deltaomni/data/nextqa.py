from __future__ import annotations

from typing import Any

from deltaomni.data.schema import CanonicalEpisode, FinalQA, MediaAsset, observation_grid
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
    final_qa = []
    for row in qa_rows:
        choices = tuple(str(row[key]) for key in ("a0", "a1", "a2", "a3", "a4"))
        answer_field = row["answer"]
        answer = choices[int(answer_field)] if str(answer_field).isdigit() else str(answer_field)
        final_qa.append(
            FinalQA(
                question_id=str(row["qid"]),
                question=str(row["question"]),
                choices=choices,
                answer=answer,
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
        modality=Modality.VIDEO,
        media=media,
        observations=observation_grid(media.duration_seconds, chunk_seconds),
        sections=(),
        final_qa=tuple(final_qa),
    )
    episode.validate_for_downstream()
    return episode

