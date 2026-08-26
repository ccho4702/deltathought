# Real-data execution plan

This plan begins only after `deltaomni-data-audit` reports a source as ready. No custom research
dataset or unofficial mirror is used.

Official raw files are read from `/mnt/nfs_shared_data/dataset/deltaomni`. Every extracted clip,
canonical episode, full/delta embedding cache, checkpoint, and prediction remains under the
project-local `intermediates/` or `outputs/` trees.

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

Current gate: blocked. NExT-QA media and embedding caches are ready, and medium SSV2 reconstruction
transfers with a small positive signal, but the delta-caption projector does not yet beat zero and
shuffled delta. Running final QA now would not isolate useful caption feedback.

The change-aware resampler and text-alignment pretraining also failed zero/shuffle ablations. Do not
try larger caption runs with the same representation objective. Add semantic/action supervision to
the delta encoder and first require held-out normal delta to beat zero and shuffled conditions.

Semantic/action supervision now passes that delta-state gate, but both text-alignment and caption
resamplers still fail shuffled controls. The next authorized design is a typed soft/discrete semantic
token bottleneck derived from the supervised delta state. Free-form captions and NExT-QA remain
blocked until that interface passes normal/zero/last/shuffle ablations.

## R4 — Video commit timing

SSV2 does not contain internal caption boundaries. This phase remains blocked until an established,
licensed temporal-caption source passes the provenance and acquisition gates. ActivityNet Captions
is the leading candidate, but is currently blocked on exact usage terms. Kinetics-GEB+ is not core
under the current adoption policy.

## No-go conditions

- Anchor plus compressed delta does not outperform anchor-only and shuffled-delta reconstruction.
- Delta-state task accuracy falls materially below current-full accuracy at a matched token budget.
- Caption quality remains unchanged when delta is zeroed or shuffled.
- Learned commit timing does not beat fixed-rate and change-threshold baselines at matched rate.
- Generated captions fail to improve independent QA or improve it even when shuffled.
- Real-media results depend on missing, unofficial, low-adoption, or license-ambiguous data.
