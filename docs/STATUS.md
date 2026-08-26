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

AudioSet Strong timing pilot is now complete. The direct official ID mapping is
`<youtube-id>_<start-ms>` → `<youtube-id>_<start-ms>_<start+10000-ms>.flac`; shared raw directories
are accessed by exact filename without scanning.

- Calibrated learned exact P/R/F1: `0.493 / 0.600 / 0.541`
- Raw CLAP change-threshold exact F1: `0.507`
- Fixed-final exact F1: `0.423`
- Learned / raw / fixed ±1-second F1: `0.803 / 0.788 / 0.524`

Interpretation: a small real timing signal exists; the learned policy narrowly beats both baselines
at exact and ±1-second tolerance. No tuning beyond train-split threshold calibration was performed.

## Medium SSV2 scaling and NExT-QA transfer

Scaling from 32 to 128 train clips and from 100 to 300 reconstruction updates improved held-out SSV2
MSE from `2.8247` to `2.5756`, with retrieval R@1 rising from `0.375` to `0.396`. The raw-pooled
baseline remained stronger at `2.0680`.

The small codec failed zero-shot NExT-QA reconstruction, but the medium codec changed this result:

- Medium learned / anchor / last / shuffled MSE: `3.3988 / 3.4567 / 3.4507 / 3.5932`
- Learned / anchor retrieval R@1: `0.1875 / 0.1429`
- Raw-pooled MSE: `2.7399`

This is evidence that modestly more real training improves cross-domain preservation, although the
simple raw-pooled delta remains substantially stronger.

## Caption alignment decision

Projector-only accuracy improved with more clips but zero/shuffled causality failed. Jointly training
the delta encoder with caption ranking initially destroyed reconstruction; a 500× lower codec LR and
full reconstruction guard prevented collapse, but held-out caption accuracy stayed at chance and
shuffled delta remained competitive. Current delta-to-frozen-Qwen semantic alignment is therefore a
confirmed blocker, not a passed stage.

Immediate next architecture work: replace the single pooled linear projector with a change-aware
resampler trained first against text embeddings, then caption CE. Final NExT-QA evaluation remains
deferred until normal generated captions beat zero and shuffled delta on held-out media.
