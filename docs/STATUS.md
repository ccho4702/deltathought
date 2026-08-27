# DeltaOmni status

## Target-architecture correction

The intended method uses the native vision encoder, audio encoder, and Thinker from pinned
Qwen2.5-Omni-7B. All DINOv2/CLAP/separate-Qwen results below are substitute-backbone baselines and
must not be cited as evidence that Qwen2.5-Omni understands first-plus-delta inputs. No target-stack
result had been completed when this correction was recorded. Migration to the official Thinker is
the active experiment line.

The first native-encoder smoke now passes at pinned revision
`ae9e1690543ffd5c0221dc27f79834d0294cba00`. Independently encoded two-second blocks produced 100
video tokens and 50 audio tokens, both width 3584. Repeating the same block with the same batch-one
execution shape was bitwise exact for both modalities; changed real-media blocks were non-identical.
Thinker-only peak reserved memory was 17.03 GiB on one RTX A6000. These token counts are measured at
the configured video resolution and are not universal fixed budgets.

Canonical episode schema v2 is now fixed in Python and JSON Schema. It always exposes nullable
image/video/audio assets, modality/joint captions, transcript/subtitle/OCR, timed events, QA,
dialogue history, and provenance. Null means the source does not provide the field; an empty array
means the field is defined but has no episode items. Dataset revisions are immutable and written as
atomic split JSONL files with a checksummed manifest.

The corrected full conversion, NExT-QA `official-2021-ann-1955d89e-schema-v2-r2`, contains
5,440 episodes and 47,692 QA items, with official train/validation/test counts and no cross-split
source-group overlap. All 5,440 video files were present; 5,406 contain audio streams and 34 record
audio as null. The missing local media-license record is explicitly null and blocks training until
resolved. Manifest SHA-256 is
`06abface4a7ca438e9cffbfc0d63fc182c253ce845be5c8f08f278d86368f00a`.

SSV2 `official-v2-ann-b24e4609-schema-v2` is also fully canonicalized: 168,913 train, 24,777
validation, and 27,157 test episodes, each with one official video caption and null QA/audio fields.
All 220,847 media files were present and no split source IDs overlap. Media discovery took 90.8s,
stream/hash caching 432.8s, and the complete conversion plus reload verification 486.5s. Manifest
SHA-256 is `25b8fb4b524a32be41aece867e439c136f4943a01a96189c7d958d6230cd7459`.
The absent local SSV2 media-license record is explicitly null and remains a training gate.

Native Qwen2.5-Omni video embedding was profiled on four A6000s using canonical SSV2. Two-second
blocks produced 100–120 tokens of width 3584. Batch one per rank was fastest at 87.2 blocks/s total
and 17.06 GiB peak reserved per rank; batch 2/4/8 fell to 80.4/64.4/46.7 blocks/s while memory rose
to 17.39/18.40/22.32 GiB. Cache generation therefore uses one independently encoded causal block
per rank, preserving the bitwise-repeatable execution shape measured by the native encoder smoke.

The first native-Omni S1 cache is complete for four direction-sensitive SSV2 classes: 2,048 train,
256 validation, and 256 untouched-test clips, balanced by class with no source overlap. Clips with
fewer than two canonical blocks are excluded and deterministically replaced, so every record has at
least one actual delta. Each independently encoded two-second block is exactly 128×3584 after
aspect-preserving 224×224 letterboxing; clips contain two to four blocks. Four-GPU cache generation
took 95.1s initially and 5.6s for the eligibility-corrected incremental pass. Manifest SHA-256 is
`24239ece81b38eb34a0ff99112fafc3a3beaa8202bc93d2715a3f9e28df10012`.

A label-free one-token DeltaTok integration run now validates the paper-style MSE trainer on frozen
native-Omni video features. Four-layer encoder/decoder training ran for 1,000 steps in 37.2s. On 389
held-out consecutive pairs, reconstruction MSE was 0.4202 versus 0.5759 for copying the previous
feature (27.0% lower), cosine similarity was 0.6177, and 389-way feature retrieval R@1 was 52.4%
(chance 0.26%). No SSV2 labels were loaded by the trainer. This is an implementation gate, not the
general-video result; the next run uses existing VGGSound media with the paper-scale tokenizer.

The corrected VGGSound S1 subset revision is
`official-2020-s1-subset4608-schema-v2-r2`: 4,096 train, 256 validation, and 256 untouched-test
paired clips. Every episode contains both video and audio, all weak class labels are explicitly
excluded from S1, and YouTube source groups have zero cross-split overlap. A full decode audit of all
4,608 selected videos found one validation source with reproducible H.264 macroblock errors in both
PyAV and FFmpeg. Revision r2 explicitly excludes `53UdZyM9MyE_000252` and deterministically replaces
it with cleanly decoded `yLazKv68TeA_000078`; train and test are unchanged. Canonical manifest
SHA-256 is `c37adcaf659a25b71cb75aa1709f3fc04e6d7db2ed4d42ed7da012495c2107b5`.
The official dataset license is recorded as CC-BY-4.0 while original-video copyright remains with
the source owner, so media and recoverable embeddings are not publication artifacts.

The frozen native Qwen2.5-Omni cache for VGGSound r2 contains 22,675 aligned two-second blocks per
modality, yielding 18,067 consecutive training/evaluation pairs. Each video block is 128×3584 and
each audio block is 50×3584; both are float16 copies of independently computed BF16 encoder output.
The cache occupies 20.81 GB for video and 8.14 GB for audio. Four-GPU generation used at most
17.04 GiB reserved VRAM per rank and successfully resumed twice from atomic per-clip artifacts.
Final manifests reloaded every tensor, checked shape/finiteness/model revision, and recorded a
SHA-256 for every cache file. Video/audio manifest SHA-256 values are respectively
`5f60dc135355516d3c8812410e76cfd4b9cdc39027fcbabe40719e10b8c175b8` and
`6e90c8dd1e767d8cfb5b6ce1b0d0ace67a7ff3b4934cc69e4c180b30a3ddfc12`.

The first exposure-matched native-Omni video DeltaTok run is complete. A 12-layer, width-768
encoder/decoder with exactly one delta token trained for 2,000 four-GPU steps at global batch 128,
equivalent to 14.2 passes over 18,067 train pairs. Training took 965.4s and peak reserved memory was
4.98 GiB/rank. The final checkpoint was evaluated once on the untouched 256-clip test split:

- Teacher-forced MSE was 0.32958 versus copy-previous 0.43404 and zero-delta 0.33659.
- Final autoregressive rollout MSE over one to four deltas was 0.40841 versus anchor-only 0.52843,
  zero-delta 0.42026, and length-matched cross-clip shuffled delta 0.42925.
- Final retrieval R@1 was 83.59% versus cross-clip shuffled 76.95%, but remained below the static
  anchor's 86.72% and zero-delta rollout's 84.77%.
- Horizon-four rollout MSE was 0.40914 versus teacher-forced 0.32645, demonstrating 25.3% drift.

All predeclared one-percent MSE and one-point retrieval content-control gates passed on untouched
test. This is bounded empirical evidence that native Qwen2.5-Omni video delta content remains useful
through four ordered updates. It is not evidence for longer horizons or Thinker caption
understanding. Checkpoint SHA-256 is
`0933e2d29b2cc0b8bbe20959250ef52feb16b87a5b0c5464996f042c6672b0ae`.

Last updated: 2026-08-27

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
- Existing shared VGGSound media: `/mnt/nfs_shared_data/dataset/omniembed/vggsound`

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

## Layout-aware causal integration on the scaled split

The strict three-seed protocol was rerun with the current CLS-plus-2D-grid codec using 800-step
delta training (about 50 sampled epochs) and actual Qwen caption generation.

- 17-token test semantic accuracy: `0.746 / 0.754 / 0.723`.
- 17-token learned/raw MSE: about `1.567 / 1.668`.
- 17-token greedy caption exact: `0.766 / 0.754 / 0.711`.
- 65-token test semantic accuracy: `0.809 / 0.781 / 0.734`.
- 65-token learned/raw MSE: about `1.110 / 1.196`.
- 65-token greedy caption exact: `0.809 / 0.773 / 0.719`.

Every zero/last/worst-cross-label-shuffle accuracy gate passed. The performance-oriented A6000
preset is now 65 tokens; the 17-token preset remains the lower-memory balanced option. Retained
aggregate: `outputs/reports/layout_causal_caption_results.json`.

## Fifteen-update long-horizon causal captions

The 65-token fidelity layout was extended from eight to sixteen uniformly sampled frames. Every
transition emits 65 delta tokens, so each clip processes 15×65 = 975 delta-token updates while the
accumulator remains fixed at 65 tokens. Captioning receives only one discrete semantic token; it
never sees the full DINO anchor or a concatenated delta history.

Three independent 800-step BF16/DDP runs used 2,048 train clips, 256 validation clips, and 256
untouched-test clips. Twelve originally selected clips had fewer than 16 decoded frames. The
deterministic same-class reselection changed 5 train, 2 validation, and 7 test IDs; the two extra
held-out changes arise because validation/test are sliced from one ordered held-out pool. Hard delta
validation accuracy was
`0.730 / 0.762 / 0.719`; untouched test was `0.727 / 0.797 / 0.770` (mean `0.764`). Test zero was
`0.250`, last-only was `0.266 / 0.266 / 0.258`, and worst cross-label shuffle was at most `0.105`.

The frozen-Qwen adapter was then trained for 800 steps per seed. Untouched-test candidate accuracy
was `0.727 / 0.789 / 0.773`; unrestricted greedy exact match was
`0.727 / 0.785 / 0.773` (mean `0.762`). Five of 768 generations were outside the four target strings.
This is direct generation evidence that the end-to-end path survives 15 accumulated updates, but it
remains a bounded four-class classification-as-caption experiment. The 8- and 16-frame clip sets are
not perfectly identical because of the short-clip replacements, so their small mean difference is
not evidence of improvement or degradation. Retained aggregate:
`outputs/reports/layout65_16frame_causal_caption_results.json`.
