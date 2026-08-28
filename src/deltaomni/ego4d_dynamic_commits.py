from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from deltaomni.data.schema import CanonicalEpisode, CaptionAnnotation


@dataclass(frozen=True)
class DynamicCommitConfig:
    block_seconds: float
    maximum_window_seconds: float
    maximum_commit_gap_seconds: float
    maximum_commits_per_window: int
    minimum_commits_per_window: int


@dataclass(frozen=True)
class DynamicCommit:
    caption_id: str
    text: str
    start_seconds: float
    end_seconds: float
    commit_seconds: float
    previous_full_block: int
    current_full_block: int

    @property
    def delta_updates(self) -> int:
        return self.current_full_block - self.previous_full_block


@dataclass(frozen=True)
class CommitWindow:
    window_id: str
    source_id: str
    source_group_id: str
    anchor_block: int
    final_block: int
    commits: tuple[DynamicCommit, ...]
    truncated_precontext: bool

    @property
    def delta_updates(self) -> int:
        return self.final_block - self.anchor_block

    def validate(self, config: DynamicCommitConfig) -> None:
        maximum_updates = max(
            0, math.ceil(config.maximum_window_seconds / config.block_seconds) - 1
        )
        commit_count = len(self.commits)
        if not (
            config.minimum_commits_per_window
            <= commit_count
            <= config.maximum_commits_per_window
        ):
            raise ValueError(f"Invalid dynamic commit count: {self.window_id}")
        if self.anchor_block < 0 or self.anchor_block > self.final_block:
            raise ValueError(f"Invalid dynamic commit window bounds: {self.window_id}")
        if self.delta_updates > maximum_updates:
            raise ValueError(f"Dynamic commit window exceeds its token budget: {self.window_id}")
        previous = self.anchor_block
        for commit in self.commits:
            if commit.previous_full_block != previous:
                raise ValueError(f"Dynamic commit sequence contains a gap: {self.window_id}")
            if commit.current_full_block < commit.previous_full_block:
                raise ValueError(f"Dynamic commit sequence runs backward: {self.window_id}")
            previous = commit.current_full_block
        if previous != self.final_block:
            raise ValueError(f"Dynamic commit sequence end mismatch: {self.window_id}")


def load_config(path: Path) -> DynamicCommitConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = DynamicCommitConfig(
        block_seconds=float(raw["block_seconds"]),
        maximum_window_seconds=float(raw["maximum_window_seconds"]),
        maximum_commit_gap_seconds=float(raw["maximum_commit_gap_seconds"]),
        maximum_commits_per_window=int(raw["maximum_commits_per_window"]),
        minimum_commits_per_window=int(raw["minimum_commits_per_window"]),
    )
    if min(
        config.block_seconds,
        config.maximum_window_seconds,
        config.maximum_commit_gap_seconds,
        config.maximum_commits_per_window,
        config.minimum_commits_per_window,
    ) <= 0:
        raise ValueError("Dynamic commit controls must be positive")
    if config.minimum_commits_per_window > config.maximum_commits_per_window:
        raise ValueError("Dynamic commit minimum exceeds its maximum")
    if config.maximum_commit_gap_seconds > config.maximum_window_seconds:
        raise ValueError("Dynamic commit gap cannot exceed the whole window")
    return config


def _start_block(caption: CaptionAnnotation, block_seconds: float) -> int:
    return max(0, math.floor(caption.start_seconds / block_seconds))


def _current_block(caption: CaptionAnnotation, block_seconds: float) -> int:
    return max(0, math.ceil(caption.commit_seconds / block_seconds) - 1)


def build_windows(
    episode: CanonicalEpisode,
    config: DynamicCommitConfig,
) -> tuple[CommitWindow, ...]:
    captions = sorted(
        episode.captions.video or (),
        key=lambda caption: (
            caption.commit_seconds,
            caption.start_seconds,
            caption.caption_id,
        ),
    )
    if not captions:
        return ()
    maximum_updates = max(
        0, math.ceil(config.maximum_window_seconds / config.block_seconds) - 1
    )
    maximum_gap_blocks = math.ceil(config.maximum_commit_gap_seconds / config.block_seconds)
    candidates: list[tuple[int, bool, list[CaptionAnnotation]]] = []
    anchor = _start_block(captions[0], config.block_seconds)
    end = _current_block(captions[0], config.block_seconds)
    truncated = False
    if end - anchor > maximum_updates:
        anchor = end - maximum_updates
        truncated = True
    current = [captions[0]]
    previous_end = end

    for caption in captions[1:]:
        caption_end = _current_block(caption, config.block_seconds)
        gap = caption_end - previous_end
        fits = (
            caption_end - anchor <= maximum_updates
            and gap <= maximum_gap_blocks
            and len(current) < config.maximum_commits_per_window
        )
        if fits:
            current.append(caption)
            previous_end = max(previous_end, caption_end)
            continue
        candidates.append((anchor, truncated, current))
        anchor = _start_block(caption, config.block_seconds)
        truncated = False
        if caption_end - anchor > maximum_updates:
            anchor = caption_end - maximum_updates
            truncated = True
        current = [caption]
        previous_end = caption_end
    candidates.append((anchor, truncated, current))

    windows = []
    for candidate_index, (anchor, truncated, values) in enumerate(candidates):
        if len(values) < config.minimum_commits_per_window:
            continue
        previous = anchor
        commits = []
        for caption in values:
            end = max(previous, _current_block(caption, config.block_seconds))
            commits.append(
                DynamicCommit(
                    caption_id=caption.caption_id,
                    text=caption.text,
                    start_seconds=caption.start_seconds,
                    end_seconds=caption.end_seconds,
                    commit_seconds=caption.commit_seconds,
                    previous_full_block=previous,
                    current_full_block=end,
                )
            )
            previous = end
        window = CommitWindow(
            window_id=f"{episode.split}:{episode.source_id}:{candidate_index:04d}",
            source_id=episode.source_id,
            source_group_id=episode.source_group_id,
            anchor_block=anchor,
            final_block=previous,
            commits=tuple(commits),
            truncated_precontext=truncated,
        )
        window.validate(config)
        windows.append(window)
    return tuple(windows)
