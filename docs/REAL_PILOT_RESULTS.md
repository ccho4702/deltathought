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

## Retained runs

- Reconstruction: `ssv2-pilot-20260826T001020Z-ede4e09b`
- Initial caption: `ssv2-caption-20260826T001348Z-af1ad75d`
- Ranking caption: `ssv2-caption-20260826T001533Z-69d7c333`

Generated caches, checkpoints, and raw metrics remain project-local and are excluded from Git.

