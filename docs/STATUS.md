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

Run: `delta-sanity-20260826T134242Z-808038a4`

- Accumulated reconstruction MSE: `1.1128e-4`
- Last-delta-only reconstruction MSE: `1.7564e-2`
- Accumulated caption exact: `1.0`
- Last-delta-only caption exact: `0.5`
- Accumulated-caption unseen QA: `1.0`
- Last-delta-caption unseen QA: `0.59375`
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

## Change-aware resampler result

The proposed resampler compressed 257 full plus 8 delta tokens into 8 Qwen-width query tokens and
was trained with text-embedding contrastive alignment before caption CE. It also failed the causal
gate: alignment accuracy was `0.281` for normal, zero, last, and shuffled delta; caption accuracy was
`0.375` for all four conditions. The frozen-Qwen caption path is now a no-go until the delta encoder
receives an explicit semantic/action auxiliary objective.

Final NExT-QA answer evaluation is intentionally not run: with captions independent of delta,
normal-vs-shuffled QA cannot isolate useful feedback.

## Semantic auxiliary objective

Adding SSV2 action supervision directly to the accumulated delta state while retaining
reconstruction produced a clear held-out signal:

- Normal / zero / last-only / shuffled semantic accuracy: `0.500 / 0.250 / 0.281 / 0.406`
- Reconstruction MSE: `2.5756 → 2.5596`, still below anchor `2.7185`

This confirms that accumulated real delta can carry semantic action information when explicitly
supervised. Re-running the change-aware Qwen resampler with this checkpoint improved normal text
alignment to `0.469`, but shuffled remained higher at `0.500`; caption accuracy was `0.281` versus
zero `0.313`. The current Qwen bridge still fails.

Decision: preserve reconstruction, timing, cross-domain, and semantic-head positive signals. Stop
open-ended caption/QA work with the current bridge. Next test should use a soft or discrete semantic
token interface before attempting free-form caption generation.

## Layout-aware delta setting

The video direct path now preserves DINO's CLS token and pools its 16×16 patch tokens as a 2D grid.
The previous flat 32-token path remains a measured baseline, but is no longer the default. Three-seed
joint reconstruction/semantic sweeps selected two explicit presets at semantic weight `2.0`:

- Balanced: 17 tokens (`CLS + 4×4`), about 15× fewer tokens than the 257-token full embedding.
- Fidelity: 65 tokens (`CLS + 8×8`), about 4× fewer tokens.

Four-frame SSV2 validation:

- Balanced learned/raw/last/shuffled MSE: `1.6273 / 1.6877 / 2.2311 / 3.5734`.
- Fidelity learned/raw/last/shuffled MSE: `1.1372 / 1.1898 / 2.0400 / 4.0566`.
- Balanced semantic normal/shuffled: `0.500 / 0.438`; fidelity: `0.438 / 0.375`.

Eight-frame SSV2 validation, which accumulates seven consecutive deltas from the first full anchor:

- Balanced learned/raw/last/shuffled MSE: `1.5236 / 1.5802 / 2.2588 / 3.2516`.
- Fidelity learned/raw/last/shuffled MSE: `1.0728 / 1.1236 / 2.1650 / 3.6990`.
- Fidelity semantic normal/shuffled: `0.469 / 0.406` with zero at chance `0.250`.
- Fidelity NExT-QA diagnostic MSE: learned `1.5755` versus raw pooled `1.5996`.

Every candidate above passed all three seeds. The default video setting is the balanced 17-token
grid; use 65 tokens when reconstruction fidelity has priority. This establishes delta preservation
and ordered accumulation, but does not resolve the frozen-Qwen caption bridge.

## A6000 causal semantic-token and caption result

The earlier one-position shuffle on class-grouped manifests was not a valid counterfactual; it
retained the same class for most examples. The corrected path uses repeated balanced cross-label
permutations, three seeds, separate search-validation and untouched-test splits, and explicit
zero/last/raw-pooled controls.

The selected 16-code, one-token setting uses usage-entropy weight `0.05`. On untouched test, hard
semantic accuracy was `0.762 / 0.707 / 0.785`; zero was `0.250`, last-only was
`0.293 / 0.254 / 0.289`, and worst shuffle was `0.102 / 0.125 / 0.078`. Effective code counts were
`6.04 / 4.25 / 4.46`. Learned MSE was `1.9894 / 1.9902 / 1.9905`, beating its matched raw-pooled
baseline `2.0053` in every seed.

Only the semantic token—not the full anchor—was projected into frozen pinned Qwen2.5-7B. Target-CE/
ranking weights `5/2` produced untouched-test candidate and greedy exact caption accuracy
`0.758 / 0.715 / 0.789`. Zero stayed `0.250`, last-only was `0.293 / 0.242 / 0.281`, and worst
shuffle was `0.109 / 0.125 / 0.074`. This passes the bounded four-class caption gate; broader
open-vocabulary captions, independent QA benefit, and learned internal video timing remain open.

Runtime was 4× RTX A6000, BF16, and global delta batch `128`. This host requires NCCL P2P, shared
memory, and IB transports disabled with loopback socket transport. The representative profile was
about `498` samples/s on four GPUs versus `110` on one GPU.
