# DeltaOmni status

Last updated: 2026-08-26

## Completed

- Independent audio/video full anchors, previous embeddings, accumulated delta slots, commit clocks,
  typed captions, scoped resets, and full refreshes.
- Step reconstruction and section-anchor-plus-accumulated-delta reconstruction.
- Synthetic compositional ablation where the first and last local deltas contain different required
  information.
- Caption, commit, length, zero/shuffle/last-only, independent relational QA, and exact resume tests.
- Pinned DINOv2, CLAP, and frozen Qwen2.5 delta-prefix CPU smokes.
- Official AudioSet Strong and NExT-QA annotation acquisition and full integrity audits.
- Provenance and real-data readiness gates.

## Latest retained functional metrics

Run: `delta-sanity-20260825T235249Z-b374b880`

- Accumulated commit reconstruction MSE: `5.1779e-5`
- Last-delta-only reconstruction MSE: `1.7803e-2`
- Accumulated caption exact: `1.0`
- Last-delta-only caption exact: `0.5`
- Accumulated-caption unseen QA: `1.0`
- Last-delta-caption unseen QA: `0.71875`
- Trigger F1 and length accuracy: `1.0 / 1.0`

These are synthetic functional metrics and are not real-media research results.

## Raw data locations

- DeltaOmni-owned official annotations: `/mnt/nfs_shared_data/dataset/deltaomni`
- Existing shared SSV2 official media: `/mnt/nfs_shared_data/dataset/ssv2`
- Existing shared NExT-QA media: `/mnt/nfs_shared_data/dataset/NExT-QA`

All paths are read-only inputs. Derived manifests, decoded subsets, embeddings, checkpoints, and
reports remain under `/home/changho.choi/deltaomni`.

## First real SSV2 pilot

Data: four direction-sensitive classes, 32 train clips and 16 source-disjoint validation clips, four
uniform frames per clip, frozen DINOv2-base.

- Held-out reconstruction MSE: initial `2.9179` → learned `2.8247`
- Anchor / last-only / shuffled MSE: `2.8951 / 2.8746 / 2.9625`
- Raw-pooled delta MSE: `2.1802` (stronger than the learned codec)
- Frame retrieval R@1: anchor `0.3333` → learned `0.3750`
- Full / reconstructed / delta-state action accuracy: `0.4375 / 0.3750 / 0.4375`
- Chance: `0.25`

Interpretation: a small real reconstruction/retrieval signal exists and downstream action accuracy
does not collapse, but the learned codec does not beat the simple raw-pooled delta baseline.

## First real frozen-Qwen caption pilot

Projector-only target CE reduced and held-out four-way accuracy rose above chance, but the causal
ablation failed.

- Initial → final held-out accuracy: `0.25 → 0.375`
- Normal / zero / last-only / shuffled accuracy: `0.375 / 0.375 / 0.3125 / 0.375`
- Normal target NLL: `1.4586`; zero/shuffled NLL: `1.4186 / 1.4249`

Interpretation: caption alignment is currently inconclusive. Qwen/projector learning occurs, but
there is no evidence that accumulated delta rather than priors drives the held-out caption. More
real clips or joint delta/projector alignment is required later; hyperparameter tuning is deferred.

## Immediate next pilot

Add clip-end commit prediction on cached SSV2 embeddings, then prepare a small NExT-QA media subset
for independent downstream evaluation. AudioSet media needs an indexed ID audit before use; avoid
scanning its very large shared directories directly.
