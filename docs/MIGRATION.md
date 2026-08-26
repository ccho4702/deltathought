# Server migration handoff

Prepared: 2026-08-26

## Source repository

- GitHub: `git@github.com:ccho4702/deltathought.git`
- Branch: `main`
- Base experiment commit: `10c09d6de6a12d548610cc5f6d244f48f58d8bc5`
- Layout-aware delta milestone: `1a79f7b2fb91798603246086c872879cc7cdb9bf`
- Migration handoff tag: `migration-20260826-v2`

Clone and recreate the locked environment:

```bash
git clone git@github.com:ccho4702/deltathought.git
cd deltathought
uv sync --frozen --group dev
uv run ruff check src tests
uv run pytest -q
```

## Raw data

- DeltaOmni annotation root: `/mnt/nfs_shared_data/dataset/deltaomni`
- Existing shared SSV2: `/mnt/nfs_shared_data/dataset/ssv2`
- Existing shared NExT-QA: `/mnt/nfs_shared_data/dataset/NExT-QA`
- Existing shared AudioSet media: `/mnt/nfs_shared_data/dataset/omniembed/audioset`

Raw files are not stored in Git and were never modified by DeltaOmni.

## Selected generated-state backup

Location:

```text
/mnt/nfs_shared_data/project_backups/deltathought/
  10c09d6de6a12d548610cc5f6d244f48f58d8bc5/
```

Contents:

- `checkpoints/ssv2_reconstruction_step300.pt`
  - SHA-256 `288c7e49c703e985cf7871189edec359096c1c4c26449f5f754d8f1673f2bbdd`
- `checkpoints/audioset_timing_step100.pt`
  - SHA-256 `b62133e29f874e2e523886499637ff3727c03d3cabd186c9e938d51c4d24b1af`
- `checkpoints/ssv2_semantic_step200.pt`
  - SHA-256 `8d81c49dd4548f82d700fb6a121041bb16474bbf550ceefca494d5b553fc83c7`
- `checkpoints/ssv2_semantic_resampler.pt`
  - SHA-256 `90babd9936dc57396fbb3659c270bdbca4f2c96f0062c5fc1b1a03da426e1763`
- `results_and_reports.tar.gz`
  - SHA-256 `08a68419f94e4364e8a86f0b4f79d216accabd1e12cf4604a570adf6a52d8077`
- `SHA256SUMS`

Verify after migration:

```bash
cd /mnt/nfs_shared_data/project_backups/deltathought/10c09d6de6a12d548610cc5f6d244f48f58d8bc5
sha256sum -c SHA256SUMS
```

## Not backed up

- `.venv/` — recreated with `uv sync --frozen --group dev`
- `inputs/models/` — official pinned revisions in `configs/backbones.yaml`
- `intermediates/cache/` — regenerated from shared raw media
- redundant/intermediate checkpoints — selected final checkpoints only were retained

## Restore selected results

The `results_and_reports.tar.gz` archive contains the small result JSON/HTML files and logs without
model checkpoints. Extract it at the new repository root if historical local reports are needed.
Embedding caches are optional; rerunning the bounded preparation stages is preferred.

## Layout-aware delta milestone backup

The verified v4 video-delta artifacts for commit
`1a79f7b2fb91798603246086c872879cc7cdb9bf` are stored separately at:

```text
/mnt/nfs_shared_data/project_backups/deltathought/
  1a79f7b2fb91798603246086c872879cc7cdb9bf/
```

Retained checkpoints:

- Balanced 17-token eight-frame codec: `checkpoints/video_delta_balanced_17tokens.pt`
  - SHA-256 `ea25055b46d93bb1895c4f6043de40a489bca6a0037159113b7870676f3fbc38`
- Fidelity 65-token eight-frame codec: `checkpoints/video_delta_fidelity_65tokens.pt`
  - SHA-256 `49c778a5ee683bfc01602a7f3320b59b5e52e5dd9130460703f67c279d6edb41`

The same directory contains the balanced, fidelity, per-step temporal, and four-frame multi-seed
JSON reports plus `SHA256SUMS`. The complete backup is 120 MB. Verify it with:

```bash
cd /mnt/nfs_shared_data/project_backups/deltathought/1a79f7b2fb91798603246086c872879cc7cdb9bf
sha256sum -c SHA256SUMS
```

The 483 MB eight-frame DINO embedding cache is intentionally omitted because it is reproducible
from the immutable shared SSV2 raw media and pinned model revision.

Synthetic checkpoints now record `PairDeltaEncoder.ALGORITHM_VERSION`. Automatic verification and
resume reject unversioned or incompatible checkpoints rather than partially loading them. After
restoring this milestone, create a fresh sanity run before verification if only older checkpoints
are present.

## Ongoing management rule

After migration, continue using GitHub as the source of truth for all code and reproducibility
material. Push every verified milestone with updated status/results. Continue using NAS for immutable
raw data and selected checksummed large-artifact backups only. Follow `AGENTS.md`; do not allow a
local server copy to become the only authoritative source.
