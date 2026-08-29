# Server migration handoff

Prepared: 2026-08-29

## Source repository

- GitHub: `git@github.com:ccho4702/deltathought.git`
- Branch: `main`
- Base experiment commit: `10c09d6de6a12d548610cc5f6d244f48f58d8bc5`
- Layout-aware delta milestone: `1a79f7b2fb91798603246086c872879cc7cdb9bf`
- Native-Omni video S1 result: `71f5116`
- Native-Omni audio S1 result: `73f9969`
- Native-Omni AudioCaps Caption LoRA result: `ec8cf1d`
- One-second native-Omni audio/video S1 result: `2cb7048`
- Matched MSR-VTT raw/delta/continuous-KV results: `387fd72`
- Ego4D dynamic cache and matched full/delta trainers: `c60d3a6`
- Frozen LongVideoBench analysis contract: `ac3941e`
- Resumable LongVideoBench evaluator and first seven-arm run: `61d14b1`
- Untouched-Qwen and stronger delta-control implementation: `8d688ec`
- Extended eleven-arm LongVideoBench result: `f57ba1c`
- Ego4D video-only result runs: `ego4d-goalstep-full-caption-800step-main` and
  `ego4d-goalstep-delta-caption-800step-main`
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

- DeltaThought authoritative raw root: `/mnt/nfs_shared_data/dataset/deltathought/raw`
- Legacy read-only annotation root: `/mnt/nfs_shared_data/dataset/deltaomni`
- Existing shared SSV2: `/mnt/nfs_shared_data/dataset/ssv2`
- Existing shared NExT-QA: `/mnt/nfs_shared_data/dataset/NExT-QA`
- Existing shared AudioSet media: `/mnt/nfs_shared_data/dataset/omniembed/audioset`
- Existing shared VGGSound media: `/mnt/nfs_shared_data/dataset/omniembed/vggsound`
- Existing shared Ego4D media/annotations: `/mnt/nfs_shared_data/dataset/ego4d_540`
- Existing shared LongVideoBench split-tar release: `/mnt/nfs_shared_data/dataset/LongVideoBench`

Raw files are not stored in Git and were never modified by DeltaOmni.

Ego4D and LongVideoBench are used read-only from existing shared official copies under
`configs/ego4d_media_policy.yaml` and `configs/longvideobench_media_policy.yaml`. Never back up or
migrate access credentials with the project.

## Current video-only long-video milestone

The retained Ego4D training runs are:

- `outputs/real_pilots/ego4d_goalstep_full_caption/ego4d-goalstep-full-caption-800step-main/`
  - checkpoint `checkpoints/step-000800.pt`
  - SHA-256 `d117aeccaf0fafaee37663fba44cfbce05866ec4c05a0ba776099ef494f45b04`
- `outputs/real_pilots/ego4d_goalstep_caption/ego4d-goalstep-delta-caption-800step-main/`
  - checkpoint `checkpoints/step-000800.pt`
  - SHA-256 `713a022832c794fa2eea6845539f848c947111e054c4d47dc2060ef59d1360b2`

The regenerable LongVideoBench native-token cache is
`intermediates/cache/omni_longvideobench/manifest.json`: 753 videos, 3,468 windows, 358,853
one-second blocks, 1,337 QA, and approximately 3.75 GB. The complete compact comparison is tracked
at `outputs/reports/longvideobench_video_qa_analysis_extended.json`; per-arm compact reports retain
their evaluation code revisions. Raw predictions, cache tensors, logs, and checkpoints remain
ignored by Git.

This is not yet a complete server-migration backup. Before moving servers, copy the two
non-regenerable Ego4D checkpoints and compact run metadata/logs to a new
`/mnt/nfs_shared_data/project_backups/deltathought/<base-commit>/` directory, create and verify
`SHA256SUMS`, update this document with that exact path, and create the annotated migration tag.

AudioCaps annotations are pinned at
`/mnt/nfs_shared_data/dataset/deltathought/raw/audiocaps` revision
`d004db3ea1b01cf4fd0347dd8d27db90cadc8809`. Original split CSV SHA-256 values are:

- train: `c0c5223db682b3bf724ce7e7ce58d5b36929f74572e8526a7211f92d2eef7c8e`
- validation: `dab1c96641d5f3053ddb99dca3949450da9a75737bda53e11cc0aa8b102be0c3`
- test: `b91c4b7ded2f4f6e7db5b9c4983dc1e1dca3d556f505b61e3fd65cac7e1c638a`

Clotho v2.1 is retained under `/mnt/nfs_shared_data/dataset/deltathought/raw/clotho_v2.1` from
official Zenodo record `4783391`. The original archives and verified MD5 values are:

- development: `c8b05bc7acdb13895bb3c6a29608667e`
- validation: `7dba730be08bada48bd15dc4e668df59`
- evaluation: `4569624ccadf96223f19cb59fe4f849f`

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

## Native Qwen2.5-Omni S1 checkpoints

The final one-token native-Omni DeltaTok checkpoints are retained in separate checksummed backups:

- Video: `/mnt/nfs_shared_data/project_backups/deltathought/71f5116/`
  - `checkpoints/step-002000.pt`
  - SHA-256 `0933e2d29b2cc0b8bbe20959250ef52feb16b87a5b0c5464996f042c6672b0ae`
- Audio: `/mnt/nfs_shared_data/project_backups/deltathought/73f9969/`
  - `checkpoints/step-001000.pt`
  - SHA-256 `8c64d86088180e045b75720d1e0270ced413043683914fb17dbbd60328c15796`

Each directory contains its own `SHA256SUMS` and short provenance README. Verify with:

```bash
cd /mnt/nfs_shared_data/project_backups/deltathought/71f5116 && sha256sum -c SHA256SUMS
cd /mnt/nfs_shared_data/project_backups/deltathought/73f9969 && sha256sum -c SHA256SUMS
```

The corresponding untouched-test reports are tracked in Git under `outputs/reports/`. Native Omni
embedding caches are intentionally omitted from NAS backup because they are checksummed but
regenerable from pinned Qwen2.5-Omni and immutable VGGSound media.

The selected AudioCaps Caption LoRA plus delta-interface checkpoint is retained at:

- `/mnt/nfs_shared_data/project_backups/deltathought/ec8cf1d/checkpoints/step-000200.pt`
- SHA-256 `c00e3e4d6466becccbed2616f50b7fa210bafb4e68227b71266ae68d0d9cf872`

Verify with:

```bash
cd /mnt/nfs_shared_data/project_backups/deltathought/ec8cf1d && sha256sum -c SHA256SUMS
```

The final one-second DeltaTok checkpoints are retained together at
`/mnt/nfs_shared_data/project_backups/deltathought/2cb7048/`:

- Audio `checkpoints/audio-step-001000.pt`
  - SHA-256 `f0706270d06b66821b18e2ed40513917caea1e6852de757ae0f66330985d7b38`
- Video `checkpoints/video-step-002000.pt`
  - SHA-256 `2d46816bc45b4dda9b4ef27c4372b5b956cff4b34a0b7a999d94bd295918a4e5`

Verify with:

```bash
cd /mnt/nfs_shared_data/project_backups/deltathought/2cb7048 && sha256sum -c SHA256SUMS
```

Synthetic checkpoints now record `PairDeltaEncoder.ALGORITHM_VERSION`. Automatic verification and
resume reject unversioned or incompatible checkpoints rather than partially loading them. After
restoring this milestone, create a fresh sanity run before verification if only older checkpoints
are present.

## Ongoing management rule

After migration, continue using GitHub as the source of truth for all code and reproducibility
material. Push every verified milestone with updated status/results. Continue using NAS for immutable
raw data and selected checksummed large-artifact backups only. Follow `AGENTS.md`; do not allow a
local server copy to become the only authoritative source.
