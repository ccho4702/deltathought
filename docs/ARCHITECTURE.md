# DeltaOmni architecture and verification contract

The bounded `delta_slots` state below describes the historical DINOv2/CLAP substitute-backbone
experiments. It is not the current native-Qwen caption/QA implementation. The active Qwen path
uses a FULL anchor followed by every one-token delta since the last commit, so visual prefix length
grows with the commit interval. Its same KV also retains earlier visual tokens as well as caption
tokens. Fixed-size visual accumulation and selective visual-KV refresh are pending requirements,
not verified properties.

## Research questions

1. Can bounded delta states retain consecutive full modality embeddings?
2. Can each modality independently commit after enough change accumulates?
3. Do caption, commit-timing, and caption-length losses decrease?
4. Does caption feedback help answer a separate question whose answer is absent from captions?

## State semantics

Each modality owns an independent `anchor`, `previous`, `delta_slots`, evidence `load`, and section
age. `previous` is the immediately preceding embedding; `anchor` is the full embedding at the last
caption commit/FULL refresh. Caption generation and language memory are separate from this representation
state: emitting a caption never clears the Thinker's autoregressive context.

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

## Commit, representation state, and language memory

A modality commits when its learned trigger crosses a threshold, an official temporal boundary is
reached, its load reaches capacity, or its maximum age is reached. At commit it:

1. emits `<CAPTION_D_m> ... </CAPTION_D_m>` from anchor and accumulated delta;
2. retains the emitted caption tokens and all earlier captions in the same Thinker KV cache;
3. refreshes the video anchor with the current full embedding and resets the visual accumulation;
4. continues consecutive deltas from that refreshed visual state.

The `<FULL_m>` refresh after a caption does not erase caption tokens from the Thinker KV cache.
Additional refreshes are allowed at declared context boundaries such as a long unannotated gap.
Evaluation must distinguish modality-state refresh from language-context reset; treating
independent `generate()` calls as continuous memory is invalid.

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

## Target Qwen2.5-Omni interface

The method-level target uses `Qwen/Qwen2.5-Omni-7B` revision
`ae9e1690543ffd5c0221dc27f79834d0294cba00`. The Thinker's native `visual` and `audio_tower`
produce variable-length 3584-dimensional sequences. The active streaming setting uses causal
one-second blocks;
each block is encoded independently so a current embedding cannot attend to future media. The
student Thinker receives the first full block followed by typed, time-positioned delta blocks.

Multi-commit training uses one causal concatenated attention graph with loss only on caption target
tokens. Incremental inference must produce the same logits by appending chunks through
`past_key_values`; generated caption tokens are appended before later delta chunks. This equivalence
is an implementation invariant, not an empirical research claim.

The required evaluation compares raw full Omni input, first plus all ordered deltas, first only,
last delta only, zero delta, temporal reversal, within-class temporal shuffle, and wrong-source
delta. Actual held-out caption and QA performance—not reconstruction or alignment loss—determines
whether the Thinker understands the delta representation.

## Substitute-backbone baseline

- DINOv2-base emits 257 full visual tokens of width 768; the initial video delta budget is 8 tokens.
- CLAP HTSAT-unfused emits one 512-dimensional global token per audio chunk; the initial audio delta
  budget is one token, avoiding token expansion.
- A trainable typed projector maps `[FULL][DELTA]` prefixes to the 896-dimensional hidden space of a
  frozen Qwen2.5-0.5B-Instruct decoder.

The DINOv2/CLAP CPU smoke verifies deterministic frozen embeddings, nonzero changed-input deltas, exact
same-input zero deltas, finite prefix gradients, frozen LLM parameters, and decreasing real-Qwen
caption CE. This baseline does not establish any property of the target Qwen2.5-Omni encoders or
Thinker.

## Legacy language-alignment status

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
