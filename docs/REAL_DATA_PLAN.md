# Real-data execution plan

This plan begins only after `deltaomni-data-audit` reports a source as ready. No custom research
dataset or unofficial mirror is used.

DeltaThought-owned official raw files are read from
`/mnt/nfs_shared_data/dataset/deltathought/raw`. Existing shared datasets and legacy annotations are
read in place without modification. Every canonical episode, embedding cache, checkpoint, and
prediction remains under the project-local `intermediates/` or `outputs/` trees.

## R1 — Video delta representation and scoped action caption

Data: official Something-Something V2 under its research-use agreement. Model: official DINOv2-base
at pinned revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.

1. Audit official train/validation metadata and media hashes.
2. Run a 16-clip media/PTS smoke before broader preprocessing.
3. Extract frozen DINOv2 patch embeddings for consecutive sampled frames.
4. Train only delta encoder, accumulator, and reconstructor first.
5. Add the human-verified action label as the clip-end scoped video caption.
6. Compare current full tokens, anchor-only, raw subtraction, learned delta, zero delta, and shuffled
   delta.

SSV2 provides clip-level timing only. It can validate representation and caption content, but it
cannot support a claim about learned internal event boundaries.

## R2 — Audio delta and timestamped commit

Data: AudioSet Strong annotations. Model: official CLAP HTSAT-unfused at pinned revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.

1. Audit which official source clips remain available and record attrition.
2. Extract fixed-duration CLAP embeddings without mixing video evidence.
3. Use human onset/offset annotations for commit targets.
4. Treat class-name sentences as deterministic label verbalizations, not human captions.
5. Test silence, steady sound, gain change, transient onset, offset, and overlap separately.

Stop if delta magnitude follows gain/noise more strongly than event identity, or if a simple
change-threshold baseline matches the learned trigger.

## R3 — Independent downstream QA

Data: official NExT-QA human QA annotations and properly licensed source media. Caption targets and
QA targets remain in different datasets and no QA answer is synthesized from captions.

Matched evaluation arms:

1. Full video embeddings, no generated captions.
2. Delta reconstruction, captions hidden.
3. Delta reconstruction plus generated caption history.
4. Delta reconstruction plus shuffled/wrong captions.

The claim requires arm 3 to preserve full-input accuracy and beat arms 2 and 4 on the untouched test
split. Caption concatenation, answer-string overlap, or QA built from caption text is invalid.

NExT-QA annotations and all 5,440 media files are present. The official NExT-QA and VidOR pages do
not provide a click-through acceptance mechanism; the media originate from YFCC100M and retain
uploader-selected per-item Creative Commons licenses. The project policy permits internal
non-commercial research, prohibits raw-media and recoverable-embedding redistribution, and requires
a per-item metadata audit before external release. The historical lightweight QA diagnostic also
used an invalid QA-row `roll(1)` shuffle: 87.5% of validation rows retained the same source delta.
Source-group-disjoint controls are now required.

The change-aware resampler and text-alignment pretraining also failed zero/shuffle ablations. Do not
try larger caption runs with the same representation objective. Add semantic/action supervision to
the delta encoder and first require held-out normal delta to beat zero and shuffled conditions.

Semantic/action supervision now passes that delta-state gate, but both text-alignment and caption
resamplers still fail shuffled controls. The next authorized design is a typed soft/discrete semantic
token bottleneck derived from the supervised delta state. Free-form captions and NExT-QA remain
blocked until that interface passes normal/zero/last/shuffle ablations.

Correction: the historical `roll(1)` shuffle was class-preserving for most class-grouped examples.
The active gate replaces it with repeated balanced cross-label permutations and treats the earlier
shuffle comparison as invalid. Setting search uses three seeds on a dedicated validation split;
exactly one preregistered setting may proceed to the untouched test split. Eligibility additionally
requires learned reconstruction to beat anchor, last-delta-only, and raw-pooled delta baselines.

The scaled workload uses 512 training clips, 64 search-validation clips, and 64 untouched-test clips
per action class with eight frames per clip. It supports one or four GPUs through the same config,
BF16 DDP, exact resume, and retained per-trial logs. The final delta gate and the subsequent typed
semantic-token-only frozen Qwen2.5-7B four-class caption gate passed on all three test seeds.

R3 remains blocked despite this caption result. The current semantic codes are trained on four SSV2
action classes and cannot be assumed to express the broader evidence required by NExT-QA. The next
valid step is to define an independently supervised, broader semantic vocabulary or open-vocabulary
caption target and then repeat the normal/zero/last/cross-label-shuffle gate before downstream QA.

## R4 — Video commit timing

SSV2 and MSR-VTT do not contain suitable internal caption boundaries. Ego4D dense narrations and
GoalStep/NLQ temporal annotations are the selected source for variable-duration training commits.
Their timing must be preserved rather than quantized to a fixed number of updates, and caption
targets must describe evidence available up to each causal boundary. The existing shared official
copy is used read-only under `configs/ego4d_media_policy.yaml`; raw media and recoverable embeddings
are never publication artifacts.

LongVideoBench is the frozen external QA evaluation. Use its labeled 1,337-question validation
split only for final comparisons because public test labels are absent. No LongVideoBench question,
answer, subtitle, category, or duration bucket may influence training, checkpoint selection,
threshold selection, prompt development, or commit construction. Predeclare vanilla full-video,
fine-tuned full-video, no-delta, ordered-delta, zero-delta, cross-video shuffled-delta, and
caption-memory shuffle controls before opening aggregate validation results.

## No-go conditions

- Anchor plus compressed delta does not outperform anchor-only and shuffled-delta reconstruction.
- Delta-state task accuracy falls materially below current-full accuracy at a matched token budget.
- Caption quality remains unchanged when delta is zeroed or shuffled.
- Learned commit timing does not beat fixed-rate and change-threshold baselines at matched rate.
- Generated captions fail to improve independent QA or improve it even when shuffled.
- Real-media results depend on missing, unofficial, low-adoption, or license-ambiguous data.
