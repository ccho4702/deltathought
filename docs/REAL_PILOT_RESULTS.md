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

Generated caches, checkpoints, and raw metrics remain project-local and are excluded from Git.
