# Server migration handoff

Prepared: 2026-08-26

## Source repository

- GitHub: `git@github.com:ccho4702/deltathought.git`
- Branch: `main`
- Base experiment commit: `10c09d6de6a12d548610cc5f6d244f48f58d8bc5`
- Migration handoff tag: `migration-20260826`

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
