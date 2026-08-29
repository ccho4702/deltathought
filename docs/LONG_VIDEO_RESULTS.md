# Video-only long-video results

## Scope

This document records the completed 2026-08-29 video-only milestone. Ego4D GoalStep supplies
caption training; LongVideoBench release
`60d1c89c1919a198b73be39c2babb213b29d6a5c` supplies external multiple-choice QA. Audio remains
out of scope until the video path establishes source-specific temporal use.

The Ego4D cache contains 1,722 train and 481 validation windows, 192,240 one-second blocks, and
10,525 natural caption commits. At each commit the model follows
`FULL → accumulated new DELTA → generated caption → FULL refresh` while retaining all language
tokens in one autoregressive KV state. Matched full-token and delta models trained for 800 steps.

LongVideoBench evaluation covers all 1,337 labeled validation questions from 753 videos. Its cache
contains 3,468 windows and 358,853 one-second blocks representing 99.76 hours of video. Every arm
uses the same question/choice prompt and direct choice-letter logits; all parse rates are `1.0`.

## Ego4D training evidence

On 128 Ego4D validation windows containing 496 caption events, the full-token model obtained
continuous/reset word-F1 `0.1597/0.1585`. DeltaThought obtained continuous/reset/zero word-F1
`0.1783/0.1556/0.0027`. This establishes that the trained caption path can use accumulated deltas
and same-KV context on the training distribution. Its derived final-answer probe did not improve
with memory, so independent LongVideoBench QA remained necessary.

## LongVideoBench accuracy

| Arm | Accuracy | Status |
|---|---:|---|
| `delta_reversed` | 48.62% | Original order control |
| `delta_continuous_kv` | 48.17% | Proposed video-only path |
| `delta_permuted` | 48.17% | Post-hoc deterministic within-window permutation |
| `vanilla_commit` | 48.09% | Post-hoc untouched weights, matched commit/KV interface |
| `delta_cross_video` | 47.64% | Original source-disjoint donor control |
| `full_commit_ft` | 46.67% | Matched full-token caption fine-tuning |
| `delta_anchor_only` | 46.07% | Post-hoc removal of delta tokens |
| `caption_memory_removed` | 45.77% | Original same-KV memory ablation |
| `delta_norm_noise` | 45.10% | Post-hoc per-token norm-matched random noise |
| `delta_zero` | 23.71% | Original zero-value delta control |
| `delta_last_only` | 23.26% | Original last-nonzero-only control |

The key paired comparisons use 100,000 video-cluster bootstrap samples and a paired exact test:

| DeltaThought minus control | Difference | Cluster 95% CI | Paired p | Decision |
|---|---:|---:|---:|---|
| `vanilla_commit` | +0.07 pp | [-2.29, 2.42] | 1.000 | No model gain |
| `full_commit_ft` | +1.50 pp | [-0.74, 3.73] | 0.194 | Inconclusive |
| `caption_memory_removed` | +2.39 pp | [0.54, 4.23] | 0.009 | Memory-removal control passed |
| `delta_anchor_only` | +2.09 pp | [0.31, 3.86] | 0.020 | Delta tokens contribute |
| `delta_norm_noise` | +3.07 pp | [0.76, 5.36] | 0.0086 | Learned delta manifold contributes |
| `delta_cross_video` | +0.52 pp | [-0.98, 2.04] | 0.573 | Source-specific gate failed |
| `delta_permuted` | 0.00 pp | [-1.11, 1.10] | 1.000 | Order gate failed |
| `delta_reversed` | -0.45 pp | [-1.59, 0.68] | 0.519 | Order gate failed |

## Mechanistic diagnosis

Zero and last-only are misleadingly destructive controls: 92.8% of zero-delta captions are empty,
whereas normal, reversed, cross-video, permuted, anchor-only, and vanilla captions are nonempty.
The large 24-point normal-versus-zero/last gaps therefore partly measure out-of-distribution
activation collapse rather than correct video evidence.

The stronger controls give a narrower conclusion. Normal delta beats both no delta tokens and
norm-matched random vectors, so the model recognizes the learned delta distribution. It does not
require correct temporal order: normal and permuted delta have identical accuracy and 93.2% answer
agreement; reversed has 92.9% answer agreement. It also does not require the correct source:
cross-video delta changes most captions but preserves accuracy. The correct per-window FULL anchor,
generic learned-delta structure, language priors, and caption scaffolding remain plausible
alternative explanations.

Continuous same-KV inference beats memory removal, but caption-memory shuffling was not implemented
before aggregate evaluation. The result establishes benefit from retaining the autoregressive
stream, not benefit from the correct semantic contents of earlier captions.

## Validity and decision

The original aggregate run contained seven implemented arms. `vanilla_commit`, anchor-only,
norm-matched noise, and within-window permutation were added after inspecting those results and are
post-hoc diagnostics. The standard raw/uniform-video vanilla and fine-tuned baselines,
shuffled-caption memory, and subtitle-only controls listed in the original protocol were not run.
LongVideoBench validation has been repeatedly inspected and cannot serve as an untouched final set
for a revised model.

The current objective fails the baseline, source-specific causal, and temporal-order gates. The
next valid step is not a broad dataset or inference-config sweep. Train a revised model on Ego4D
with explicit source-disjoint, order-permuted, anchor-only, and norm-noise negative objectives;
select only on Ego4D validation; freeze raw/uniform baselines and prompts; then evaluate once on a
second untouched long-video benchmark. Audio extension remains deferred.

## Reproducibility

- Training checkpoint SHA-256 values:
  - full-token: `d117aeccaf0fafaee37663fba44cfbce05866ec4c05a0ba776099ef494f45b04`
  - delta: `713a022832c794fa2eea6845539f848c947111e054c4d47dc2060ef59d1360b2`
- First aggregate evaluation code: `61d14b148bce9db3754a443c85891908a6f09a73`
- Post-hoc control evaluation code: `8d688eca11d015826378de24880d3673ac5235e8`
- Tracked extended analysis:
  `outputs/reports/longvideobench_video_qa_analysis_extended.json`
- Bootstrap analysis is deterministic; a full rerun reproduced report SHA-256
  `ed676a4b51c43bd808a3e37c90e0fe52bf9354d459c01437dc6b6fc9105a17f4`.
- Final verification: `uv run --frozen ruff check src tests` and
  `uv run --frozen pytest -q` (`148 passed`).
