# DeltaOmni architecture and verification contract

## Research questions

1. Can bounded delta states retain consecutive full modality embeddings?
2. Can each modality independently commit after enough change accumulates?
3. Do caption, commit-timing, and caption-length losses decrease?
4. Does caption feedback help answer a separate question whose answer is absent from captions?

## State semantics

Each modality owns an independent `anchor`, `previous`, `delta_slots`, evidence `load`, and section
age. `previous` is the immediately preceding embedding; `anchor` is the full embedding at the last
caption/reset.

```text
d_t = DeltaEncoder(previous, current)
delta_slots_t = Accumulate(delta_slots_(t-1), d_t)
current_hat_t = Reconstruct(anchor, delta_slots_t)
```

Training checks both `Reconstruct(previous, d_t)` and `Reconstruct(anchor, delta_slots_t)` against
the frozen teacher's current full embedding. Full embeddings are deliberately computed at every
step during this representation experiment. Encoder-compute reduction is a later question.

## Delta and accumulation

The delta encoder combines gated before/after features, their difference, and elementwise
interaction. Learned queries compress them into fixed slots. An identity-initialized direct path
preserves the pooled feature difference; learned attention supplies a nonlinear residual. Identical
inputs produce exact zero delta.

The accumulator keeps an additive evidence path and adds a small gated recurrent residual for order
sensitivity. This retains magnitude while keeping a bounded token count.

## Commit and reset

A modality commits when its learned trigger crosses a threshold, its load reaches capacity, or its
maximum age is reached. At commit it:

1. emits `<CAPTION_D_m> ... </CAPTION_D_m>` from anchor and accumulated delta;
2. refreshes the modality anchor with the current full embedding;
3. resets only that modality's slots, load, and age;
4. emits `<FULL_m>` to mark the next section.

## Loss

```text
L = lambda_recon * 0.5 * (L_step_recon + L_section_recon)
  + lambda_identity * L_same_state_zero
  + lambda_trigger * L_commit_BCE
  + lambda_caption * L_caption_CE
  + lambda_length * L_length_CE
```

Losses are computed independently per modality and averaged. Missing real-data labels must be
masked as unknown, never converted to no-change labels.

## Final QA contract

Synthetic captions state only individual change classes. The final question asks whether the second
change class is the same as, higher than, or lower than the first. None of those relational answers
appears in a caption. A separate downstream QA probe must compose ordered evidence to answer.

Real evaluation must use separately authored, source-disjoint human QA. Caption concatenation or QA
constructed by copying caption text is explicitly invalid as method-level evidence.

## Pinned real-model interface

- DINOv2-base emits 257 full visual tokens of width 768; the initial video delta budget is 8 tokens.
- CLAP HTSAT-unfused emits one 512-dimensional global token per audio chunk; the initial audio delta
  budget is one token, avoiding token expansion.
- A trainable typed projector maps `[FULL][DELTA]` prefixes to the 896-dimensional hidden space of a
  frozen Qwen2.5-0.5B-Instruct decoder.

The CPU smoke verifies deterministic frozen embeddings, nonzero changed-input deltas, exact
same-input zero deltas, finite prefix gradients, frozen LLM parameters, and decreasing real-Qwen
caption CE. This verifies interfaces only; it does not establish real-media semantic quality.

## Language-alignment status

Raw DINOv2/CLAP delta states are not assumed to be directly interpretable by an LLM. The implemented
`DeltaLanguageProjector` maps each modality width to Qwen hidden width and adds independent modality
and `FULL`/`DELTA` type embeddings. The actual training curriculum remains:

1. Train modality-space delta encoder/accumulator/reconstructor against frozen full embeddings.
2. Freeze modality encoder and LLM; train separate video/audio projectors or resamplers with scoped
   caption CE.
3. Verify normal vs zero/shuffled/last-only delta captions on held-out media.
4. Unfreeze a small part of the delta encoder or add modality-specific LLM LoRA only if the frozen
   interface is insufficient.

The artificial Qwen smoke proves gradient flow and optimization only. Semantic alignment will be
complete only after real scoped captions improve on held-out official data.

## Accumulation ablation

The synthetic section target is factorized across time: the first local delta contains one component
of the change class and the final local delta contains another. No single delta determines the
caption. Verification directly compares anchor plus all accumulated delta slots against anchor plus
the last delta only for reconstruction, captioning, and the separate relational QA task.
