# DeltaThought repository rules

## Sources of truth

- GitHub `ccho4702/deltathought` is the source of truth for code, tests, configuration, dependency
  files, plans, status documents, migration instructions, and compact result summaries.
- NAS is the source of truth for immutable raw datasets and selected generated-state backups that
  are too large for Git.
- Never rely on an unpushed local source change or a workstation-local checkpoint as the only copy.

## GitHub management

- Work from this Git repository and preserve a clean, reviewable history.
- After a verified milestone, update `docs/STATUS.md` and relevant result documents, run lint/tests,
  commit the focused changes, and push `main`.
- Before server migration or handoff, verify that local `HEAD` equals `origin/main` and create a
  descriptive annotated migration tag.
- Track source, configs, `uv.lock`, tests, Markdown/HTML documentation, data paths, checksums, and
  small JSON result summaries.
- Never commit raw media, model weights, environments, embedding caches, large checkpoints, secrets,
  credentials, license-acceptance records, or private user identifiers.

## NAS management

- DeltaThought-owned immutable raw distributions belong under
  `/mnt/nfs_shared_data/dataset/deltathought/raw` and must be owned by the project account.
- `/mnt/nfs_shared_data/dataset/deltaomni` is a legacy `donghun.kim`-owned tree and is read-only
  context; do not add, modify, move, or delete files there.
- Existing shared datasets such as SSV2, NExT-QA, and AudioSet are read-only inputs; do not move,
  modify, rename, or duplicate another user's shared copy.
- Do not write canonical manifests, extracted media, embeddings, predictions, checkpoints, or logs
  into the raw dataset tree.
- Active derived artifacts remain project-local under `intermediates/`, `outputs/`, and `logs/`.
- Before server migration, copy only selected non-regenerable checkpoints and compact result/log
  archives to `/mnt/nfs_shared_data/project_backups/deltathought/<base-commit>/`.
- Every NAS backup must include a `SHA256SUMS` manifest and must be verified after copying.

## Reproducibility and migration

- Pin all external models and datasets by official identifier and revision in `configs/`.
- Regenerate `.venv`, official model caches, and embedding caches rather than treating them as
  authoritative backup state.
- Keep `docs/MIGRATION.md` current with Git commit/tag, raw paths, selected backup paths, checksums,
  restore commands, and intentionally omitted artifacts.
- A handoff is complete only when GitHub remote SHA, migration tag, NAS backup checksums, and the
  project test suite all verify successfully.
