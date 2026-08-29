# DeltaOmni status

## MSR-VTT video-caption foundation (2026-08-28)

The official local MSR-VTT distribution is now canonicalized under revision
`official-msr-vtt-10k-schema-v2`: 6,513 train, 497 validation, and 2,990 untouched-test videos,
with exactly 20 human captions per video (130,260/9,940/59,800 caption items), no missing media, and
no split source overlap. All 10,000 MP4 files were probed and hashed; canonical conversion plus full
reload/checksum verification took 36.7s. This is the source of truth for the upcoming native video
FULL+variable-delta caption LoRA and its full-input fine-tuned Qwen baseline.

The first native MSR-VTT video prefix cache is complete for all 6,513 train and 497 validation
videos plus a locked 1,000-video test subset. It occupies 3.89 GB and took 1,077s on four A6000s;
every artifact binds the media, canonical manifest, Qwen revision, and video DeltaTok checkpoint
and was reloaded for shape/finiteness verification.

A 20-step video-caption LoRA smoke improved validation normal word-F1 from `0.396` to `0.617` and
normal NLL from `3.476` to `2.706`, proving that native video prefixes can train and generate fluent
captions. It failed the causal delta gate: final shuffled/zero word-F1 was `0.613/0.586`, and normal
NLL was worse than shuffled/zero (`2.706` versus `2.699/2.680`). Qualitatively, several outputs were
reasonable coarse captions, while others hallucinated singing, race cars, or hotel rooms. Long
training with the same objective is therefore blocked. The next comparison must fine-tune full raw
video and first-frame-only Qwen baselines and isolate temporally dependent captions before another
delta-caption run.

The matched raw-video Qwen2.5-Omni LoRA baseline now passes an exact-resume 20-step smoke on the
same canonical training source. A forced stop at step 2 wrote a complete checkpoint and resumed at
step 3 with model, optimizer, deterministic sampling, and per-rank RNG state restored. Validation
word-F1 on 16 videos was `0.442` with the complete 2-fps video versus `0.375` with only the first
two frames. This establishes that the full-input baseline path consumes useful later visual
evidence. It is not yet the retained baseline: several generations produced a reasonable first
clause followed by spurious `Human:` dialogue continuations, so the 1,000-step matched run and
larger validation evaluation remain required.

The matched 1,000-step raw-video baseline is complete on all 6,513 MSR-VTT training videos and a
fixed 128-video validation evaluation. Training took 8,065s on four A6000s with valid atomic
checkpoints through step 1,000. Full-video word-F1/ROUGE-L was `0.4183/0.4014`, versus
`0.3882/0.3821` from only the first two frames, so later visual evidence contributed `+3.01`
word-F1 points. However, this is not a strong caption model: many outputs began with a plausible
clause and then emitted unrelated `Harry Potter`, HTML, `Human:`, or no-input continuations. Longer
training therefore passed the weak later-frame control but did not repair output-format or
hallucination failures. The result is retained as the required fine-tuned full-input baseline and
as negative evidence against assuming that more steps alone solve caption grounding.

A separate matched evaluation confirms that the fine-tuned raw-video baseline is worse than the
actual vanilla model. On the same 128 validation videos, pretrained Qwen2.5-Omni full-video
word-F1 was `0.4861`, compared with `0.4183` after the 1,000-step MSR-VTT LoRA (`-6.78` points).
The fine-tuned model still beat its own first-two-frame control (`0.3882`), so it learned from later
frames while degrading overall generation. The required terminology is therefore fixed:
`vanilla` is untouched Qwen, `full_ft` is this worse raw-video LoRA, and the delta model is reported
separately. Fine-tuning is not treated as the baseline merely because it was trained.

The deliberately aggressive 16-video training-set diagnostic reached caption CE below `1e-4` by
roughly 40 steps and remained there through step 300. On those same training videos, full-video
word-F1 was `0.5831` versus `0.4068` for the first two frames (`+17.63` points). This proves the LoRA
path can memorize the fixed targets and use later visual input. It does not prove clean caption
generation: exact match remained zero because outputs often continued into unrelated role, JSON,
or no-input text after the correct first clause.

The full 1,000-step ordered-delta caption model passed every preregistered 497-video NLL and
128-video generation control. Final normal/zero/cross-video-shuffled NLL was
`2.5623/2.6069/2.6450`; word-F1 was `0.6271/0.6144/0.5898`. Normal therefore beat zero by `1.27`
points and source-disjoint shuffled delta by `3.73` points, while improving from its initial
`0.4126`. Generated outputs were generally concise and avoided the raw baseline's long role/HTML
tails, although some examples remained identical under normal and zero delta and several captions
were semantically wrong. This is bounded MSR-VTT evidence that ordered delta content affects open-
vocabulary captioning; it is not yet multi-commit memory or long-video QA evidence.

The first actual Qwen same-KV multi-commit smoke is complete. Three source-disjoint MSR-VTT
sections are trained in one causal attention graph, and inference incrementally appends each
anchor/delta/prompt and generated caption through `past_key_values` without resetting state. Before
continuous training, the single-caption checkpoint repeated the first section's caption across
later sections: continuous word-F1 was `0.5694` versus `0.6455` when resetting each section. After
40 continuous steps, word-F1 was `0.6171` continuous, `0.6144` reset-each, and `0.5890` with zero
delta. Thus training recovered the `7.61`-point contamination deficit, slightly exceeded reset by
`0.27` points, and retained a `2.81`-point delta-content gap. The mechanics and contamination gate
pass on 24 captions from eight validation streams. Because these streams concatenate unrelated
videos, this is not evidence that prior captions provide useful natural memory; that claim remains
reserved for Ego4D within-video sequences.

The 64-stream/192-caption scale-up falsified the plain caption-CE objective. After 400 CE-only
steps, continuous/reset/zero word-F1 was `0.6181/0.6165/0.6157`: contamination was fixed, but the
zero gap collapsed to `0.24` points and failed the one-point gate. Training had learned to rely on
anchor and language priors rather than preserve delta use. This negative result is retained rather
than selecting the favorable eight-stream smoke.

Adding an explicit sequence-level zero-delta NLL margin repaired that failure. With the same 400
steps and 64 validation streams, final continuous/reset/zero word-F1 was
`0.6149/0.6129/0.4913`. Continuous same-KV inference improved from `0.5644` before training, stayed
within `0.19` points above reset-each, and beat zero delta by `12.35` points. The controlled
objective sacrificed only `0.32` normal word-F1 points relative to CE-only while making delta
content causally necessary. This establishes actual Qwen multi-commit same-KV mechanics and a
causal delta caption gate on unrelated concatenated sections. It still does not establish useful
natural caption memory, since cross-video prior captions should ideally be irrelevant; Ego4D is
required for that next claim.

The final long-video evaluation source is now LongVideoBench, not MSR-VTT. The existing shared
distribution is present as the official 151 GB split-tar release at Hugging Face revision
`60d1c89c1919a198b73be39c2babb213b29d6a5c`: 1,337 labeled validation questions and 5,341
unlabeled test questions. It is used read-only under the versioned project media policy. Validation
will be frozen for external evaluation only; neither its questions nor labels may enter training,
model selection, commit-target construction, or prompt tuning. Ego4D is the intended source for
natural variable-duration training commits from the existing shared official copy.

The long-video execution code is ready and the existing shared media is authorized by the user for
read-only internal research. Ego4D preprocessing selects official GoalStep leaf events and verifies
local 540p media. The
dynamic policy produces 1–8 natural commits per window, at most 118 one-second deltas, and explicit
refreshes after 90-second gaps or 120-second windows. A four-GPU cache stores each native Qwen
anchor, ordered one-token deltas, exact event ranges, and the 64-token native full embedding at
every commit. This supports two matched Ego4D arms with identical captions and initialization:
`full_commit_ft` consumes full native tokens at every commit, while `delta_continuous` consumes one
anchor plus only new deltas and preserves caption KV state. Both actual Qwen forward/generation
paths pass synthetic-schema integration smokes and are ready for the real Ego4D run.

The video-only Ego4D GoalStep experiment is complete. Canonicalization retained 275 train and 69
validation videos with 8,821/2,486 official event captions. The native-Qwen cache contains
1,722/481 dynamic windows, 192,240 one-second blocks, and 10,525 multi-commit captions in 6.15 GB.
Each commit follows `FULL → new DELTA accumulation → caption → FULL refresh`; captions and all
earlier language context remain in the same KV cache.

Matched 800-step full-token and delta arms trained concurrently on two A6000s each and were scored
on 128 validation windows containing 496 caption events. The full-token baseline reached word-F1
`0.1597` continuous versus `0.1585` reset-each and failed its 0.5-point memory gate. DeltaThought
reached `0.1783` continuous, `0.1556` reset-each, and `0.0027` zero-delta. It beat the full-token
baseline by `1.86` points, its reset ablation by `2.27` points, and zero delta by `17.56` points.
All caption gates passed; ordered deltas produced concise GoalStep captions while zero-delta output
mostly collapsed to empty or malformed text.

The same-KV final-answer mechanics also execute end to end, but benefit is not established. The
last-event probe reached word-F1 `0.1356` continuous, `0.1403` reset-each, and `0.0000` zero; the
full-token continuous baseline was `0.1377`. Caption generation, visual refresh, accumulation, and
final-question append all work, but caption memory has not improved an answer. This derived probe
is a mechanics check, not independent QA evidence; LongVideoBench and Video-MME remain required.

LongVideoBench release `60d1c89...` is indexed directly from its 151 GB split tar without duplicate
extraction. The frozen manifest covers all 1,337 validation questions, 753 source videos, 17
question categories, and four duration groups. The final analyzer preregisters ten arms and rejects
any incomplete or duplicate prediction set before computing overall/category/duration accuracy,
video-cluster bootstrap intervals, and paired exact tests. This was the pre-execution contract;
the implemented-arm deviations and post-hoc diagnostics are recorded below and in
`docs/LONG_VIDEO_RESULTS.md`.

The first complete video-only LongVideoBench evaluation is finished on all 1,337 validation
questions from 753 videos. DeltaThought scored `0.4817`, the matched full-token commit model
`0.4667`, caption-memory-removed `0.4577`, zero-delta `0.2371`, last-delta-only `0.2326`, reversed
delta `0.4862`, and source-disjoint cross-video delta `0.4764`; every arm had 100% choice parsing.
DeltaThought beat memory removal by `2.39` points (video-cluster bootstrap 95% CI
`[0.58, 4.23]`, paired exact p=`0.009`) and zero/last-only by more than 24 points. It did not beat
the full-token arm significantly (`+1.50` points, CI `[-0.75, 3.73]`, p=`0.194`), cross-video
delta (`+0.52` points, CI `[-0.98, 2.04]`, p=`0.573`), or reversed delta (`-0.45` points, CI
`[-1.58, 0.68]`, p=`0.519`). The preregistered causal-content and temporal-order gates therefore
fail. Caption inspection explains the zero control but not source specificity: 92.8% of zero-delta
captions are empty, whereas cross-video deltas retain fluent nonempty captions and nearly unchanged
QA. Current evidence supports a useful same-KV caption channel and need for accumulated nonzero
delta activations, but not use of the correct ordered video change. An untouched-Qwen matched
commit baseline was omitted from this run and is the next required correction; no positive
DeltaThought long-video claim is made before that baseline and stronger content controls complete.

That post-hoc correction is now complete. Untouched Qwen under the same native-token commit/KV
interface scored `0.4809`, statistically identical to DeltaThought's `0.4817` (`+0.07` points,
video-cluster bootstrap 95% CI `[-2.29, 2.42]`, paired exact p=`1.0`). Full-token caption
fine-tuning scored `0.4667`, `1.42` points below vanilla (CI `[-3.50, 0.66]`, p=`0.206`). Stronger
post-hoc delta controls separate activation from content: ordered delta beat anchor-only `0.4607`
by `2.09` points (CI `[0.31, 3.86]`, p=`0.020`) and norm-matched random noise `0.4510` by `3.07`
points (CI `[0.76, 5.36]`, p=`0.0086`). However, deterministic within-window permutation produced
exactly the same `0.4817` accuracy (CI of the difference `[-1.11, 1.10]`, p=`1.0`), reversed delta
remained slightly higher, and cross-video delta remained indistinguishable. The model uses
in-distribution learned-delta structure but has not learned temporal order or source-specific
alignment. Same-KV memory remains beneficial relative to removal, but shuffled-caption memory was
not run, so correct semantic memory is not established. These controls were chosen after the first
aggregate result and are diagnostics, not preregistered confirmation. The current objective fails
the primary baseline, order, and source-specific causal gates; changing only dataset or inference
configuration is not justified before training explicitly against cross-video and order-corrupted
negatives. `vanilla_commit` isolates untouched weights under the method's input interface; a
standard raw/uniform-video vanilla evaluation is still required for an external baseline claim.

The Ego4D-only 800-step multi-negative revision is complete. On 496 GoalStep validation caption
events, normal/cross/zero/permuted word-F1 was `0.1337/0.1044/0.0294/0.1379`; normal NLL
`2.7258` beat cross `2.7718`, permuted `2.7691`, and zero `5.8618`. Likelihood separation and
source-specific generation improved, but permuted generation remained better and continuous
same-KV F1 `0.1337` did not clear reset `0.1304` by the fixed memory margin. The run is complete,
finite, exactly resumable, and failed overall. Checkpoint SHA-256 is
`05c036baafba32e3e9fb391cc436fa96acd41dbc7155e3fc68f20871a508760c`.

The first real Ego4D commit-timing head also completed 2,000 steps. Exact F1 was `0.1643`, beating
zero `0.1196`, cross-video `0.1154`, and fixed-12s `0.0610`; at ±3 seconds it reached only
`0.2061`, below fixed-12s `0.3633`. Delta content helps locate exact boundaries, but the learned
policy is not yet a better practical scheduler. Checkpoint SHA-256 is
`ec5fd70140d236eccdb6c844068b0f93f5f82d1b114e05936d01cfe287df5569`.

A source-disjoint LongVideoBench label diagnostic used 603 videos/1,070 QA for a 30-parameter
choice calibration head and 150 videos/267 QA for validation. Raw/calibrated validation accuracy
was `47.57/49.06%` (`+1.50` points, cluster 95% CI `[-0.75, 3.85]`, paired p=`0.344`). This did
not establish a supervised QA gain. More importantly, the prior evaluator generated one caption at
the end of each fixed 120-second cache window; 120 seconds came from an engineering context bound,
not LongVideoBench annotations. These results are now explicitly named fixed-120s diagnostics.
The calibration head is excluded. Final training is Ego4D-only; LongVideoBench labels, subtitles,
categories, durations, and QA do not affect weights, thresholds, prompts, commits, or selection.

Legacy cleanup removed the standalone fixed-period AudioCaps `streaming_commit_train` PoC, the
LongVideoBench label-split/multi-negative diagnostic entrypoints, and the obsolete NExT-QA
lightweight joint-head trainer plus its coupling into the vanilla-Qwen report. Their compact
negative-result reports remain tracked for audit, while Git history preserves deleted producers.
`streaming_sequence.CommitHead` remains because the active Ego4D timing trainer imports that shared
primitive. The architecture documentation now distinguishes historical bounded DINO/CLAP states
from the active native-Qwen path, which currently concatenates deltas and has no fixed-size visual
state or selective visual-KV eviction.

## Integrity correction and execution gate (2026-08-28)

An audit of the rapid Stage 3 path found that its batch-local `roll(1)` delta control was not a
cross-source shuffle. Because questions from one video are contiguous, 119/136 validation QA
(`87.5%`) and 115/132 test QA (`87.1%`) received the same source delta. The reported shuffled
validation accuracy `23.5%` is invalid. A post-hoc source-group-disjoint reassessment of the retained
checkpoint produced `19.1%` versus normal `24.3%`; this is diagnostic only and does not upgrade the
lightweight head into Stage 3 evidence. The tracked v1 report is retained with `passed=false`.

Future NExT-QA caches use a content-signed v2 manifest. Cache reuse now requires matching canonical
manifest, model/config revisions, audio/video DeltaTok checkpoint hashes, media hashes, tensor
shapes, finiteness, and the versioned NExT-QA/VidOR/YFCC100M media policy. The official sources do
not expose a click-through acceptance flow. Internal non-commercial research may proceed, while raw
media and recoverable embeddings remain non-publishable and external release requires a per-item
YFCC100M Creative Commons metadata audit.

The fixed-period multi-commit aggregate has no tracked producer and its cross-sequence shuffled
timing F1 remains `1.0`. It is now classified as a legacy mechanics artifact, not research evidence.
The content-signed NExT-QA v2 cache was regenerated on four A6000s in 34.4s. Matched vanilla
Qwen2.5-Omni controls on all 132 QA from the same 16-video diagnostic subset reached multimodal/
text-only/video-only/audio-only accuracy `77.27/48.48/81.06/55.30%`, all with 100% parse rate.
Video-only beat text-only by `32.58` points (16-video cluster-bootstrap 95% CI
`[23.40, 41.53]`; paired exact p `< 1e-9`). Video-only exceeded multimodal by `3.79` points:
five QA flipped only in video's favor and none only in multimodal's favor, but paired exact
p=`0.0625`. Audio-only versus text-only was inconclusive (`+6.82` points, CI
`[-3.25, 16.15]`, p=`0.150`). The next Stage 3 trainer therefore starts with video anchor+deltas;
audio is added only as a subsequent ablation.

The obsolete single-GPU `deltatok_train.py` runner and its integration config were removed. The
unchanged `DeltaTok` model now lives in `deltaomni.deltatok`; the supported trainer is the scalable
single-/multi-GPU path with content controls, atomic checkpoints, and exact-resume state. Historical
checkpoints load strictly into the extracted model with no missing or unexpected parameters.

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
audio as null. Canonical provenance records the absent click-through license field as null; the
separate versioned media policy governs research use and redistribution. Manifest SHA-256 is
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

The exposure-matched native-Omni audio DeltaTok run is also complete. The same 12-layer,
width-768, one-token architecture trained for 1,000 four-GPU steps at global batch 256, again 14.2
passes over the train pairs. Training plus validation took 479.8s and peak reserved memory was 4.72
GiB/rank. On the once-evaluated untouched 256-clip test split:

- Teacher-forced MSE was 0.68692 versus copy-previous 1.19677 and zero-delta 0.79808.
- Final autoregressive rollout MSE was 0.74355 versus anchor-only 1.28558, zero-delta 0.98447,
  reversed-delta 0.91771, and cross-clip shuffled delta 1.06322.
- Final retrieval R@1 was 92.97% versus anchor 44.14%, reversed 38.67%, zero 0.78%, and cross-clip
  shuffled 0.39%.
- Horizon-four rollout MSE was 0.74295 versus teacher-forced 0.68731, an 8.1% drift penalty.

All predeclared controls passed. Native Qwen2.5-Omni audio delta content and order therefore remain
strongly useful through four updates on this distribution. Checkpoint SHA-256 is
`8c64d86088180e045b75720d1e0270ced413043683914fb17dbbd60328c15796`.

S2 data preparation has started from an authoritative project-owned raw root at
`/mnt/nfs_shared_data/dataset/deltathought/raw`. The official AudioCaps repository is pinned there
at revision `d004db3ea1b01cf4fd0347dd8d27db90cadc8809`. Exact ID matching against existing AudioSet media
found 48,344/49,838 train clips (97.0%), 483/495 validation clips (97.6%), and 943/975 test clips
(96.7%). Missing media will be recorded as deterministic attrition; it will not be downloaded from
unverified mirrors. Ego4D narrations and media remain the video-caption source candidate.

AudioCaps canonical revision `official-original-d004db3-media-schema-v2` is complete: 48,343 train,
483 validation, and 943 untouched-test audio episodes with 48,343/2,415/4,715 captions. Train has
one reference per clip and validation/test have five. Exactly 1,494/12/32 clips are absent from the
existing read-only AudioSet roots, and one zero-byte train FLAC is separately quarantined as invalid.
All YouTube source groups are split-disjoint. Manifest SHA-256 is
`fce373275db9897a8967644c6691ef9678147f6a4bfb252fc7749c469cb0a307`.

The first native-Omni AudioCaps S2 prefix cache is complete for 8,192 train, all 483 validation, and
all 943 untouched-test clips. Every item contains one frozen native audio anchor of shape 50×3584,
four ordered one-token Audio DeltaTok updates of shape 4×1×768, and the official references. The
cache is 3.53 GB, took 1,329.9s on four A6000s, and peaked at 17.75 GiB reserved per rank. Every
artifact was reloaded and hashed. Manifest SHA-256 is
`84f388807195616e43b5214e1ba154b5d030a92f5b571053cbd58a79efa0b2de`.

The Caption LoRA interface now passes real Qwen forward/backward, exact checkpoint resume, and
greedy generation smokes. It trains only PEFT adapters in the 28 Thinker text layers plus the
768→3584 delta interface; trainable audio/vision tower parameters are zero. A 20-step smoke improved
validation NLL from 3.105 to 2.705 and generated recognizable free-form captions, but correctly
failed the eight-example shuffled-delta word-F1 gate. The exposure-matched full run is next.

Audio Caption LoRA S2 has single-seed bounded evidence on the untouched AudioCaps test split.
Validation selected the
200-step checkpoint over the 1,000-step run: the latter improved generation controls but worsened
NLL from 2.958 to 3.326, so it is retained as an overfit negative result and never evaluated on
test. The selected rank-8 LoRA trained at global batch 32 and peaked at 34.04 GiB/rank. On all 943
test clips for NLL and a fixed 64-clip greedy-generation subset:

- NLL normal/zero/cross-clip-shuffled was `2.5523 / 2.8004 / 2.8342`.
- Word-F1 was `0.4072 / 0.3587 / 0.3386`.
- ROUGE-L was `0.3819 / 0.3332 / 0.3021`.
- Normal captions beat zero by 4.85 word-F1 points and shuffled by 6.87 points.

The model generates free-form captions rather than class labels; for example, a door clip produced
`A man speaks and slams a door` under normal delta, while zero and shuffled controls hallucinated a
horse and a clock. This is direct bounded evidence that the Qwen2.5-Omni Thinker LoRA understands a
native audio anchor plus four projected one-token deltas. It does not yet establish longer memory,
video caption transfer, or QA. Selected checkpoint SHA-256 is
`c00e3e4d6466becccbed2616f50b7fa210bafb4e68227b71266ae68d0d9cf872`.

Clotho v2.1 is now the license-clean long-horizon audio-caption extension. All three official Zenodo
archives were range-downloaded, verified against record `4783391` MD5 values, and retained with the
caption license and per-file Freesound license metadata. Canonical revision
`official-zenodo-4783391-v2.1-schema-v2` contains 3,828 train, 1,037 validation, and 1,045 untouched
test episodes, each with five captions. No media are missing or decoder-invalid. Nineteen records
were quarantined to remove official-split overlap at the underlying Freesound `sound_id` level,
preserving test first. Durations span 15–30s and yield 7–15 complete two-second blocks, i.e. 6–14
delta updates. Manifest SHA-256 is
`e973b72444ade41582249fcd7bfb63d02a6ebbd3225be376c4bfb7a758e32bd4`.

The final streaming granularity is now one second. Direct pinned-Qwen measurements produce 64×3584
video tokens and 25×3584 audio tokens per block; Qwen's internal position rate remains 25 audio
tokens/s even though its configured positional chunk is two seconds. The source-disjoint VGGSound
cache contains 45,643 one-second blocks per modality. Video/audio manifest SHA-256 values are
`3a7f8977d869ac8efae3c925e89769593c2ee7718e035245c3061d3e76e5166d` and
`91fd431d633df34f00e53ca8821db88cd8f53a186856a7ed1df408d313f637cc`.

Exposure-matched one-token DeltaTok runs pass every untouched-test content control through nine
one-second updates:

- Audio teacher MSE is `0.2366` versus copy `0.8294` and zero-delta `0.5936`. Final nine-delta
  rollout is `0.2637` versus anchor `0.9028`, zero `0.6606`, reversed `0.7768`, and cross-clip
  shuffled `0.9402`; retrieval is 96.88% versus shuffled 1.17%.
- Video teacher MSE is `0.2938` versus copy `0.3741` and zero-delta `0.3268`. Final nine-delta
  rollout is `0.4415` versus anchor `0.5346`, zero `0.4782`, reversed `0.4554`, and cross-clip
  shuffled `0.4951`; retrieval is 86.33% versus shuffled 41.80%.

Audio horizon-nine drift is 11.6%, while video drift is 54.8%; long visual autoregressive drift is
the main remaining S1 limitation. Audio/video checkpoint SHA-256 values are respectively
`f0706270d06b66821b18e2ed40513917caea1e6852de757ae0f66330985d7b38` and
`2d46816bc45b4dda9b4ef27c4372b5b956cff4b34a0b7a999d94bd295918a4e5`.

## Rapid multi-commit and Stage 3 diagnostic

A deliberately small one-second PoC exercised repeated streaming mechanics rather than one caption
per record. It concatenated three source-disjoint AudioCaps sections into the sequence
`FULL→9 DELTA→CAPTION→FULL→9 DELTA→CAPTION→FULL→9 DELTA→CAPTION`. A recurrent CommitHead trained
with timing BCE reached precision/recall/F1 `1.0/1.0/1.0` on 42 validation sequences, predicting
all 126 fixed-period commits. Eight test sequences generated 24 actual captions; normal/zero/shuffle
word-F1 was `0.399/0.367/0.365`. The aggregate has no tracked producer and is retained only as a
legacy artifact. It does not prove natural event-boundary timing because every section is nine
seconds and cross-sequence shuffled deltas retain timing F1 `1.0`.

The first Stage 3 diagnostic uses actual synchronized NExT-QA video, audio, questions, and five-way
answers from 64/16/16 complete clips (519/136/132 QA). A lightweight joint head was chosen to get a
performance answer before building the expensive Qwen QA LoRA. Accuracy changed as follows:

- Train: `20.8% → 99.2%`.
- Validation: `21.3% → 24.3%`.
- Test: `24.2% → 25.0%` (chance 20%).
- Validation normal/video-zero/audio-zero/delta-zero/invalid-row-shuffle:
  `24.3/19.9/20.6/15.4/23.5%`. The last number is invalid; source-disjoint post-hoc accuracy is
  `19.1%`.

The representation is sufficient to overfit real QA, but the original shuffle experiment cannot
support a causal held-out claim. Generalization is weak at this scale. The current result is a
legacy feasibility diagnostic, not final Qwen QA LoRA evidence.

The matched vanilla Qwen2.5-Omni generative baseline makes the Stage 3 gap explicit. On the same
16 short source-disjoint NExT-QA test videos and all 132 five-way questions, raw synchronized
audio/video plus the original choices reached `77.27%` accuracy with a `100%` parse rate. The
answer-position majority baseline is `26.52%`, so simple position imbalance does not explain the
result. With the choices removed, vanilla Qwen generated short answers with `9.85%` normalized
exact match, `22.59%` word-F1, and `22.59%` ROUGE-L; lexical mapping of those generations back to
the five choices reached `37.12%`, but this mapping is a non-standard diagnostic. On 16 separate
short MSR-VTT test videos with 20 human references each, one-sentence summaries reached best-
reference word-F1 `53.35%` and ROUGE-L `50.20%`.

The `77.27%` vanilla result must not be compared as if the current `25.0%` delta result were a
fine-tuned Qwen. The latter is a temporary hashed-text, mean-pooled-feature classifier trained on
only 64 videos/519 QA; it does not run the Qwen Thinker and does not consume caption memory. Its
test score improved only `24.24% → 25.0%` while train reached `99.23%`. The diagnostic head should
not be scaled further. The next Stage 3 experiment must train and evaluate an actual Qwen Thinker
LoRA using typed video/audio anchor-plus-delta prefixes and prior committed captions. Before that
run, the versioned media-policy gate and the prepared multimodal/text-only/video-only/audio-only
vanilla controls must pass. The repeatedly inspected 16-video subset remains a development
diagnostic and cannot serve as the final untouched test.

Last updated: 2026-08-28

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

- Legacy shared official annotations (read-only): `/mnt/nfs_shared_data/dataset/deltaomni`
- DeltaThought-owned raw root: `/mnt/nfs_shared_data/dataset/deltathought/raw`
- Existing shared SSV2 official media: `/mnt/nfs_shared_data/dataset/ssv2`
- Existing shared NExT-QA media: `/mnt/nfs_shared_data/dataset/NExT-QA`
- Existing shared VGGSound media: `/mnt/nfs_shared_data/dataset/omniembed/vggsound`

All paths are read-only inputs. Derived manifests, decoded subsets, embeddings, checkpoints, and
reports remain under `/home/changho.choi/deltathought`.

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
