# Shot2Story and MovieChat execution plan

## Why the previous setting is insufficient

Ego4D GoalStep provides real event start/end times and local procedural captions, but no independent
whole-video QA. Predicting each local caption does not require earlier captions, and the historical
"last completed event" probe simply asks for the last caption target again. It is an implementation
probe, not QA. The native-Qwen path also concatenates every delta since the anchor and retains old
visual KV; it does not yet implement a fixed-size delta state or language-only persistent memory.
The historical LongVideoBench run added one caption per fixed 120-second engineering window despite
the dataset providing no visual caption timing. Those results are fixed-window diagnostics only.

The replacement evaluation needs intermediate descriptions and a distinct global target on the same
video. Shot2Story supplies the closest supervised structure. MovieChat-1K supplies credible long-
video global and timestamped breakpoint QA, but not intermediate caption supervision.

## Pinned official sources

### Shot2Story

- Official code: `bytedance/Shot2Story`, commit `ae26ac3d2f9e9a91a7fd0653bfb6a2b3cb250308`.
- Official annotations: `mhan/Shot2Story-134K`, revision
  `d6b3d44befd7169c764a02a5a47e188887639e74` (1.10 GB).
- Official cached videos: `mhan/shot2story-videos`, revision
  `0e214aed3c0bc8ac5e9b8b641b150187030b4916` (167.38 GB).
- Annotation license: CC BY-NC-SA 4.0; original-video rights remain with source owners.
- NAS roots:
  - `/mnt/nfs_shared_data/dataset/deltathought/raw/shot2story/annotations-d6b3d44/`
  - `/mnt/nfs_shared_data/dataset/deltathought/raw/shot2story/videos-0e214ae/`

The 43K human train portion contains human whole-video summaries plus human shot-level visual and
narration captions. Validation/test retain the 20K-version splits and provide manually verified QA
covering temporal, holistic, and audio-related questions. The videos average roughly 16 seconds and
four shots, so this tests multi-event composition rather than hour-scale memory.

### MovieChat-1K

- Official code: `wenhaochai/MovieChat`, commit
  `ba9bb802ea209c6d8e8b4333ec917ae3aee55b1d`.
- Official train distribution: `Enxin/MovieChat-1K_train`, revision
  `1369fae920d318bc100230f98415e6e7e77a04dc`.
- Official test distribution: `Enxin/MovieChat-1K-test`, revision
  `beab351b12474f795e22f94beb82290da8389050` (17.12 GB).
- NAS test root:
  `/mnt/nfs_shared_data/dataset/deltathought/raw/moviechat_1k/test-beab351/`.

MovieChat-1K contains one detailed whole-video caption, three global QA pairs, and ten timestamped
breakpoint QA pairs per test video. It does not contain multiple intermediate visual captions per
video. The train repository is gated and reports 12.41 TB; downloading it would consume most of the
remaining shared NAS capacity. Without authenticated access and a storage allocation, MovieChat is
evaluation-only. The 17 GB test release must not be split for final training.

## Training and evaluation contract

1. Train all learned video/caption/memory components on Shot2Story human train only.
2. Treat each shot end as a commit: shot FULL/deltas → generated shot caption → FULL refresh while
   preserving generated language context.
3. Train the final whole-video summary after all shot captions. This target is distinct from any
   single intermediate caption and tests cross-shot composition.
4. Do not feed GT intermediate captions at inference. Report oracle-GT-caption memory only as an
   upper bound beside generated, reset, shuffled, and wrong-video caption memory.
5. Evaluate Shot2Story validation/test QA without using their labels for selection. Report temporal,
   holistic, and audio question groups separately; the initial model remains video-only.
6. Freeze the Shot2Story-selected model before MovieChat evaluation. Report global and breakpoint QA
   separately, plus raw/uniform Qwen, full-token, delta, zero, reversed, shuffled-caption, and
   memory-reset controls.
7. Report actual encoder frames, FULL/delta visual tokens, total Thinker input tokens, peak GPU
   memory, wall time, and retained KV bytes. Token reduction claims compare against exhaustive FULL
   and a token-matched baseline, not the sparse-FULL diagnostic.

## Success criteria

- Generated intermediate caption memory improves the distinct whole-summary/QA target over reset
  and source-disjoint shuffled memory.
- Ordered delta beats zero, reversed, and cross-video delta on both caption and final tasks.
- The method remains within a predeclared accuracy tolerance of raw/uniform Qwen while materially
  reducing Thinker visual/KV tokens.
- Shot2Story results establish multi-shot composition only. MovieChat evaluates long-video transfer;
  neither substitutes for a future dataset with dense intermediate captions and hour-scale QA.

## Execution status (2026-09-02)

- Official downloads and extraction: complete.
- Raw coverage: train/validation/test `36,951/1,982/4,025`, missing media zero.
- Canonical episodes: 42,958; validation/test QA `4,642/6,465`.
- Native-Qwen smoke cache: 80 windows, 1,549 blocks, 348 commits, 199 MB.
- 40-step caption smoke: failed memory and order generation gates.
- 800-step 64-window overfit: training memorized, validation NLL worsened, reset remained better.
- Decision: do not scale the caption-only objective. Implement whole-summary supervision and QA
  memory controls before any full-data training.

Retained checkpoint SHA-256 values:

- 40-step smoke: `d6d5b6d324c33909453424a3b138c253db66e53baa91a418c1f181adf9bfb28b`
- 800-step overfit: `c736564770ac55213138ba1c7a364fccd64b23cbd47e30ca3a88c8e77165e9a3`
