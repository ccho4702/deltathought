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

## Retained runs

- Reconstruction: `ssv2-pilot-20260826T001020Z-ede4e09b`
- Initial caption: `ssv2-caption-20260826T001348Z-af1ad75d`
- Ranking caption: `ssv2-caption-20260826T001533Z-69d7c333`
- Audio timing: `audioset-timing-20260826T002702Z-4a4b117e`
- Medium reconstruction: `ssv2-pilot-20260826T003323Z-d9dd196a`
- NExT-QA transfer: `nextqa-reconstruction-20260826T003401Z-3f4723b8`
- Stabilized joint caption: `ssv2-caption-20260826T003931Z-8cfcdc23`

Generated caches, checkpoints, and raw metrics remain project-local and are excluded from Git.
