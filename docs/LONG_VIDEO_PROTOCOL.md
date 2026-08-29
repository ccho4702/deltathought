# Long-video experiment protocol

This protocol was fixed before running any aggregate LongVideoBench validation evaluation. Ego4D
provides training supervision and natural commit boundaries. LongVideoBench is external evaluation
only. MSR-VTT remains an engineering and short-caption baseline; it is not long-video evidence.

## Data separation

- Ego4D GoalStep official train trains every learned component. Official validation selects
  checkpoints and commit/window settings. There is no use of GoalStep test annotations.
- LongVideoBench release `60d1c89c1919a198b73be39c2babb213b29d6a5c` is frozen. All 1,337
  labeled validation questions are evaluated once after model and prompt selection. The 5,341 test
  questions have no public labels and are not used for local scoring.
- LongVideoBench questions, answers, subtitles, categories, durations, and media never provide a
  training target, threshold, prompt-development example, checkpoint-selection metric, or commit
  boundary.
- Ego4D and LongVideoBench use their existing shared official copies read-only under versioned
  project media policies. Credentials are never recorded.

## Training arms

All fine-tuned arms use the same Ego4D GoalStep source videos, leaf-level captions, official train
split, validation split, caption target tokens, LoRA target layers, optimizer budget, and checkpoint
selection rule.

1. `vanilla`: pinned Qwen2.5-Omni-7B without task fine-tuning.
2. `full_ft`: Qwen2.5-Omni LoRA trained from raw video frames and GoalStep captions, without
   DeltaTok. This is the required fine-tuned baseline.
3. `delta_no_memory`: the same LoRA target with anchor-plus-DeltaTok input, but each caption event is
   evaluated without prior generated captions. This is the memory ablation.
4. `delta_continuous_kv`: anchor-plus-DeltaTok with all caption events appended to one causal
   autoregressive state. Generated caption tokens remain in the same KV cache for later commits and
   QA. This is DeltaThought.

The dynamic commit policy uses one-second DeltaTok blocks but does not use nine fixed updates.
Official segment ends determine caption commits. A window contains at most 120 seconds and eight
commits; every caption commit refreshes visual FULL state without clearing caption KV memory, and a
gap above 90 seconds starts a new bounded window. The frozen local distribution
currently yields train commit spans with median 12, p95 57, and maximum 118 delta updates.

## LongVideoBench evaluation arms

Primary comparisons use identical question/choice prompts and subtitle availability. Report both
overall accuracy and results by official duration group and question category.

- `vanilla_uniform`: vanilla Qwen with the preregistered uniform raw-frame budget.
- `full_ft_uniform`: the full-frame fine-tuned baseline with the same frame budget.
- `delta_continuous_kv`: ordered delta stream and its generated caption memory.
- `delta_zero`: replace every delta with zero while preserving token count and timing.
- `delta_cross_video`: replace deltas with a deterministic source-group-disjoint donor of matched
  length.
- `delta_reversed`: reverse delta order within each refresh window.
- `delta_last_only`: keep only the last nonzero delta at each commit while preserving padding.
- `caption_memory_removed`: answer after ordered deltas without prior generated captions.
- `caption_memory_shuffled`: use captions from a source-group-disjoint video.
- `subtitle_only`: remove video evidence while preserving the official subtitle/question input.

Uniform-frame baselines and full-coverage DeltaThought do not have equal encoder compute. Therefore,
the main result is a long-context accuracy/coverage comparison, not an efficiency claim. Separately
report decoded frames, vision-tower FLOP proxy, wall time, peak GPU memory, context tokens, and
retained KV bytes. Any efficiency claim requires a compute-matched control.

## Metrics and decision gates

- Primary: multiple-choice accuracy with 100% valid answer parsing.
- Uncertainty: source-video clustered bootstrap confidence intervals and paired source-level tests.
- Causal delta gate: ordered delta must beat zero and cross-video delta; failure blocks a positive
  DeltaThought claim even if raw accuracy is high.
- Order gate: ordered delta must beat reversed or the result cannot support temporal-order use.
- Memory gate: continuous caption memory must beat both removed and shuffled caption memory.
- Baseline gate: the fine-tuned full-frame baseline must be reported even when it underperforms
  vanilla; negative or null results are retained.
- Falsification: if delta controls are indistinguishable, treat performance as language/subtitle or
  anchor priors. If caption-memory shuffling does not hurt, captions are not useful memory.

No test-set submission or leaderboard claim is authorized by this local protocol.

## Execution deviation recorded after the first aggregate run

The first implemented aggregate evaluator covered the full-token commit arm, DeltaThought, zero,
cross-video, reversed, last-only, and memory-removed controls. It did not implement the originally
listed raw/uniform, shuffled-caption, or subtitle-only arms before inspecting aggregate validation
results. The later untouched-weight commit baseline, anchor-only, norm-matched-noise, and
within-window-permutation controls are explicitly post-hoc diagnostics. They must not be described
as preregistered confirmation, and the repeatedly inspected LongVideoBench validation split must
not be used to select a revised objective and then presented as an untouched final evaluation.

## Final training-source correction (2026-08-30)

- Ego4D GoalStep train is the only source of caption, commit, LoRA, adapter, and timing supervision.
- Ego4D validation reports oracle-GT, learned, and fixed-12s timing; it may select the retained
  model but never enters gradient updates.
- LongVideoBench is evaluation-only. The temporary 603/150-video label diagnostic and its
  30-parameter calibration head are contaminated diagnostics and are excluded.
- LongVideoBench has total duration and timestamped subtitles but no visual caption/commit labels.
  Subtitle ends are not substituted for video commit ground truth.
- The earlier one-caption-per-120-second evaluator is deprecated. In the replacement evaluator,
  an Ego4D-trained policy decides commits from one-second deltas; caption tokens remain in the same
  KV; the current FULL is retained at each commit; 120 seconds is only a forced safety refresh.
- Model and threshold choices are frozen using Ego4D before any replacement LongVideoBench result
  is read. Because the old validation labels were repeatedly inspected, new LongVideoBench results
  are external generalization diagnostics rather than untouched test claims.
