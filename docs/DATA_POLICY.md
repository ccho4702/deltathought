# Data and model provenance policy

The in-memory synthetic fixtures test implementation only. Actual training and evaluation use
existing external resources that pass `configs/provenance.yaml`; DeltaOmni does not create a custom
research dataset.

## Storage boundary

- DeltaThought-owned immutable raw distributions:
  `/mnt/nfs_shared_data/dataset/deltathought/raw`
- Legacy shared annotations under `/mnt/nfs_shared_data/dataset/deltaomni` are read-only context;
  they are not a DeltaThought-owned write target.
- Canonical manifests and extracted/derived media: project-local `intermediates/`
- Generated embedding caches and checkpoints: project-local `intermediates/`
- Retained metrics/reports and logs: project-local `outputs/` and `logs/`

Generated files are never written into the raw NAS tree. Model weights remain in the project-local
`inputs/models` cache because they are pinned runtime inputs rather than dataset distributions.

GitHub `ccho4702/deltathought` is the authoritative copy of code, configs, lockfiles, tests, and
compact result documentation. Selected large checkpoints are preserved only in the checksummed NAS
project-backup tree. The detailed mandatory workflow is in `AGENTS.md` and `docs/MIGRATION.md`.

## Acceptance gate

A core resource requires:

- an official source;
- peer-reviewed documentation;
- a verified license for the exact artifact;
- at least two adoption signals among:
  - recognized standard benchmark/model;
  - at least 100 GitHub stars;
  - at least 500 Hugging Face downloads;
  - at least 100 paper citations.

Popularity does not prove correctness, but prevents core claims from depending on unreleased or
rarely audited artifacts. Exact revisions, checksums, file inventories, media attrition, and terms
must still be recorded before download or use.

## Current audit

Approved for future integration:

- Qwen2.5-Omni-7B as the user-mandated target vision/audio encoders and Thinker. Its official
  technical report is not peer reviewed, so it has a documented research-foundation exception and
  requires independent held-out evaluation;
- DINOv2 as image/video full-embedding teacher;
- CLAP as audio teacher and audio-text alignment model;
- Qwen2.5-0.5B-Instruct as a legacy substitute-backbone caption decoder;
- AudioSet Strong annotations for audio timing and event labels.
- Something-Something V2 for human-verified motion clips and action text;
- NExT-QA annotations for independently authored human final QA.

Blocked pending additional evidence or licensing:

- DESED subsets: adoption passes, but exact subset licenses require audit;
- ActivityNet Captions: established and highly cited, but exact annotation/media terms remain a
  blocking audit item;
- YouCook2 media: annotations are established, but upstream media requires rights/availability audit;
- Kinetics-GEB+: useful structure but below the current adoption threshold and license unresolved;
- CLEVR-Change: development reference only, never core evidence under the current policy.

Recent CodecCap/CodecVDC and OmniDiff artifacts remain literature references until official releases
and adoption meet the same gate.

Official approval in the provenance file does not mean media is ready. `deltaomni-data-audit`
additionally requires local official files, checksums, a non-empty media inventory, and either a
user-created acceptance record when upstream terms have an acceptance flow or a versioned project
media policy when no such flow exists.

NExT-QA and VidOR provide direct downloads and citation requirements rather than click-through
acceptance. Their videos originate from YFCC100M and retain uploader-selected per-item Creative
Commons licenses. `configs/nextqa_media_policy.yaml` therefore limits use to internal non-commercial
research, prohibits media/recoverable-embedding redistribution, and requires a per-item license
metadata audit before external release.

## Role separation

Existing training media may serve multiple objectives inside the training split, but labels retain
separate roles:

- consecutive sequences for delta/full-embedding alignment;
- independently authored change/event text for caption learning;
- timestamped onset/offset or event boundaries for commit learning;
- separately authored QA for downstream evaluation.

Validation selects thresholds. Test sources are never used for representation pretraining, caption
training, pseudo-label generation, or hyperparameter selection.
