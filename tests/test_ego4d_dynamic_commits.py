from dataclasses import replace
from pathlib import Path

from deltaomni.data.schema import (
    CanonicalEpisode,
    CaptionAnnotation,
    CaptionBundle,
    MediaBundle,
    ProvenanceRecord,
    TextBundle,
)
from deltaomni.ego4d_dynamic_commits import build_windows, load_config


def _caption(index: int, start: float, commit: float) -> CaptionAnnotation:
    return CaptionAnnotation(
        caption_id=f"caption-{index}",
        scope="video",
        text=f"action {index}",
        start_seconds=start,
        end_seconds=commit,
        commit_seconds=commit,
        language="en",
        annotation_origin="fixture",
        timing_origin="fixture",
        independent_from_qa=True,
    )


def _episode() -> CanonicalEpisode:
    return CanonicalEpisode(
        episode_id="episode",
        dataset="fixture",
        dataset_revision="v1",
        split="train",
        source_id="video",
        source_group_id="group",
        media=MediaBundle(image=None, video=None, audio=None),
        duration_seconds=40,
        temporal_blocks=(),
        captions=CaptionBundle(
            image=None,
            video=(
                _caption(0, 0.4, 3.2),
                _caption(1, 4.0, 7.1),
                _caption(2, 20.0, 30.0),
                _caption(3, 30.0, 32.0),
            ),
            audio=None,
            joint=None,
        ),
        text=TextBundle(transcript=None, subtitle=None, ocr=None),
        events=None,
        qa=None,
        provenance=ProvenanceRecord(
            resource_name="fixture",
            source_url="https://example.com",
            annotation_path=Path("fixture.json"),
            annotation_sha256="0" * 64,
            preprocessing_config_sha256="1" * 64,
            code_revision="revision",
            processed_at_utc="2026-08-28T00:00:00+00:00",
        ),
        metadata={},
    )


def test_dynamic_windows_use_variable_delta_counts_and_refresh_long_gaps() -> None:
    config = load_config(Path("configs/ego4d_dynamic_commits.yaml"))
    config = replace(config, maximum_commit_gap_seconds=10)
    windows = build_windows(_episode(), config)

    assert len(windows) == 2
    assert [commit.delta_updates for commit in windows[0].commits] == [3, 4]
    assert [commit.delta_updates for commit in windows[1].commits] == [9, 2]
    assert windows[0].anchor_block == 0
    assert windows[1].anchor_block == 20
    assert all(len(window.commits) == 2 for window in windows)


def test_dynamic_commit_config_is_not_fixed_to_nine_updates() -> None:
    config = load_config(Path("configs/ego4d_dynamic_commits.yaml"))

    assert config.maximum_window_seconds == 120
    assert config.maximum_commit_gap_seconds == 90
    assert config.maximum_commits_per_window == 8
