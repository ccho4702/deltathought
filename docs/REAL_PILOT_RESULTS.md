# First real-data pilot results

Date: 2026-08-26

## Scope

Official shared Something-Something V2 media was read in place. No raw files were copied or
modified. Four motion-sensitive classes were sampled with 8 train and 4 validation clips per class.
Each clip was decoded at four uniform timestamps and embedded by pinned frozen DINOv2-base.

## Delta reconstruction

The delta codec was trained for 100 reconstruction-only updates without action labels. It produced
a small held-out improvement over anchor, last-only, and shuffled states and a small retrieval gain.
It preserved action accuracy within one validation example of the full embedding probe. However,
adaptive pooling of the raw full difference remained a substantially better MSE baseline.

Decision: enough signal to continue the pipeline, not enough to claim a method advantage.

## Delta-caption projection

A separate typed projector mapped the frozen DINO anchor and accumulated delta to frozen
Qwen2.5-0.5B. Target CE and candidate-ranking loss were trained on the four action phrases. Held-out
accuracy moved above chance, but zero and shuffled delta matched it and had slightly better NLL.

Decision: caption stage is inconclusive and cannot support the claim that Qwen understands the
delta. Preserve the failure and continue to the timing/downstream plumbing pilots before returning
with more data and joint alignment.

## AudioSet Strong timing

Thirty-two train and sixteen official evaluation clips were resolved directly against an existing
shared AudioSet copy. Each 10-second clip was split into one-second CLAP chunks. Human strong event
end times were causal commit targets; events ending before the first complete observation were
masked. A learned audio delta/accumulator/policy was trained for 100 updates.

After train-only threshold calibration, learned exact F1 was `0.541`, compared with `0.507` for a
calibrated raw CLAP cosine-change baseline and `0.423` for fixed-final commits. At ±1 second the
scores were `0.803 / 0.788 / 0.524`. This is a small positive timing signal, not a stable result.

## Medium SSV2 and cross-domain reconstruction

Increasing to 128 train clips, 32 validation clips, and 300 updates improved SSV2 reconstruction and
retrieval. More importantly, the medium codec transferred zero-shot to 16 NExT-QA validation videos
covering 140 independent human questions: learned MSE `3.3988` beat anchor `3.4567`, last-only
`3.4507`, and shuffled `3.5932`; retrieval R@1 improved from `0.1429` to `0.1875`. The earlier small
codec had failed this transfer. Raw-pooled delta remained strongest at `2.7399`.

Decision: additional real data improves learned reconstruction generalization, but the architecture
does not yet beat a simple compressed raw-difference baseline.

## Joint delta-caption alignment

The medium caption run first trained the projector only and reached `43.75%` held-out accuracy, but
zero/shuffled deltas remained equally good or better. Joint caption training with the original codec
LR caused reconstruction spikes up to `122` and was interrupted. A stabilized run used a 500× lower
codec LR and a full reconstruction guard; representation loss remained bounded, but normal caption
accuracy was `25%` versus zero `34.4%`, last-only `31.3%`, and shuffled `28.1%`.

Decision: the current typed linear projector into frozen Qwen does not establish semantic delta
understanding. Do not run final QA claims until held-out zero/shuffle ablations pass.

## Change-aware resampler

An 8-query cross-attention resampler replaced the linear projector. It first aligned to frozen Qwen
text hidden states with contrastive loss, then received caption CE, candidate ranking, and an
alignment guard. Held-out text-alignment accuracy was `28.1%` for normal, zero, last-only, and
shuffled delta. Caption accuracy was `37.5%` for every condition.

Decision: the bottleneck is not fixed by a more expressive projector. Reconstruction-only delta
states do not yet expose sufficiently discriminative semantic change information. The next caption
attempt must add a semantic/action auxiliary objective to the delta encoder before LLM alignment.

## Semantic delta supervision and resampler retry

A four-way SSV2 semantic head was jointly trained with the delta encoder and reconstruction guard.
Held-out accumulated-delta accuracy reached `50.0%`, compared with zero `25.0%`, last-only `28.1%`,
and shuffled `40.6%`; reconstruction was preserved and slightly improved. This is the clearest real
evidence so far that accumulated delta can carry task-relevant semantics.

The semantic checkpoint was then passed to the change-aware Qwen resampler. Normal text alignment
rose to `46.9%` and beat zero `40.6%`, but last-only matched it and shuffled reached `50.0%`.
Candidate-caption accuracy remained near chance and did not beat zero/shuffled controls.

Decision: semantic delta learning works at this scale; the current frozen-Qwen prefix bridge does
not. Switch to an explicit soft/discrete semantic-token interface before more free-form caption work.

## Layout-aware delta compression

The original direct path pooled the ordered DINO token sequence in one dimension. A layout-aware
path now preserves the CLS token and pools the 16×16 patch grid in two dimensions. All comparisons
used the same 128/32 SSV2 split, 100 reconstruction warmup steps, 150 joint semantic steps, semantic
weight `2.0`, and seeds 42/43/44.

The balanced 17-token grid reached four-frame MSE `1.6273±0.0020`, beating both its raw pooled
baseline `1.6877` and the flat 32-token learned result `1.6698`. Normal semantic accuracy was `50.0%`
versus shuffled `43.8%`. The fidelity 65-token grid reached `1.1372±0.0017` versus raw `1.1898`,
last-only `2.0400`, and shuffled `4.0566`.

On eight uniformly sampled frames, the codecs accumulated seven consecutive deltas from the first
full anchor. The 17-token grid reached `1.5236±0.0032`; the 65-token grid reached
`1.0728±0.0012`. Both beat raw pooled, last-only, and shuffled controls in every seed. For the
65-token codec, normal semantic accuracy was `46.9%`, shuffled was `40.6%`, and zero was chance at
`25.0%`. Its zero-shot NExT-QA reconstruction MSE was `1.5755`, slightly better than raw pooled
`1.5996`.

Decision: use 17 tokens (`CLS + 4×4`) as the balanced video default and 65 tokens (`CLS + 8×8`)
as the fidelity preset. The result verifies ordered accumulated-delta preservation; it is not yet a
free-form caption or QA result.

## Retained runs

- Reconstruction: `ssv2-pilot-20260826T001020Z-ede4e09b`
- Initial caption: `ssv2-caption-20260826T001348Z-af1ad75d`
- Ranking caption: `ssv2-caption-20260826T001533Z-69d7c333`
- Audio timing: `audioset-timing-20260826T002702Z-4a4b117e`
- Medium reconstruction: `ssv2-pilot-20260826T003323Z-d9dd196a`
- NExT-QA transfer: `nextqa-reconstruction-20260826T003401Z-3f4723b8`
- Stabilized joint caption: `ssv2-caption-20260826T003931Z-8cfcdc23`
- Change-aware resampler: `ssv2-resampler-20260826T004910Z-8859fd13`
- Semantic delta: `ssv2-semantic-20260826T005359Z-bfc8fcc5`
- Semantic resampler retry: `ssv2-resampler-20260826T005656Z-e04f0df0`
- Layout multi-seed: `delta-setting-sweep-20260826T133356Z-57d2ddaf`
- Layout eight-frame multi-seed: `delta-setting-sweep-20260826T133608Z-e6a14402`

Generated caches, checkpoints, and raw metrics remain project-local and are excluded from Git.

## A6000 semantic-token delta search and captions

The corrected experiment used 512 train, 64 search-validation, and 64 untouched-test clips per
class, eight frames per clip, three seeds, repeated cross-label shuffles, and a matched raw-pooled
baseline. Only the `usage_entropy_high` setting passed every validation seed. Balanced, semantic/
reconstruction reweighting, LR changes, compact codebook, longer training, and random initialization
each failed at least one preregistered seed gate; these negative runs are retained.

On untouched test, hard semantic accuracy was `0.762 / 0.707 / 0.785`, while zero was `0.250`,
last-only was `0.293 / 0.254 / 0.289`, and worst shuffle was `0.102 / 0.125 / 0.078`. Learned MSE
was `1.9894 / 1.9902 / 1.9905` versus raw-pooled `2.0053`.

A one-token-only adapter into frozen Qwen2.5-7B, trained with target CE/ranking weights `5/2`,
reached candidate and greedy exact test accuracy `0.758 / 0.715 / 0.789`. Zero remained `0.250`,
last-only was `0.293 / 0.242 / 0.281`, and worst shuffle was `0.109 / 0.125 / 0.074`. All greedy
outputs were valid action labels. This is a bounded four-class causal caption result, not yet an
open-vocabulary caption, NExT-QA improvement, or learned timing result.

Retained aggregate: `outputs/reports/a6000_delta_caption_results.json`.

## Scaled layout-aware full training

The corrected causal protocol was integrated with CLS-preserving 2D pooling. Both layouts used the
same 2,048 real training clips, 256 search-validation clips, 256 untouched-test clips, eight frames,
800 delta steps (about 50 sampled epochs), and three seeds.

For 17 tokens, untouched-test semantic accuracy was `0.746 / 0.754 / 0.723`, learned MSE was about
`1.567` versus raw-pooled `1.668`, and actual greedy caption exact was
`0.766 / 0.754 / 0.711`. For 65 tokens, semantic accuracy was `0.809 / 0.781 / 0.734`, learned MSE
was about `1.110` versus raw `1.196`, and greedy exact was `0.809 / 0.773 / 0.719`.

The 65-token fidelity layout improves both reconstruction and mean real caption generation at a
modest throughput/memory cost, so it is the performance-oriented A6000 preset. The 17-token layout
remains the balanced compression preset. All claims remain limited to the four selected SSV2 action
classes and seven accumulated frame-to-frame deltas.
