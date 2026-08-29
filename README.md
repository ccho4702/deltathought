# DeltaOmni

Independent modality delta codecs for causal audio/video streams. This repository is intentionally
separate from `omnithought`.

## Current verified scope

The target architecture is the pinned `Qwen/Qwen2.5-Omni-7B` Thinker: its native vision encoder,
native audio encoder, and text model must jointly consume the first full multimodal block plus
ordered delta blocks. The earlier DINOv2/CLAP/separate-Qwen experiments are retained only as
substitute-backbone baselines. They do not validate the target Qwen2.5-Omni architecture.

Deterministic embedding fixtures cover functional and resume behavior. Bounded real-media pilots
use existing shared Something-Something V2, AudioSet, and NExT-QA data. The fixtures are not a new
research dataset and are never used as real-method evidence.

A scaled four-class SSV2 study uses 512 training, 64 search-validation, and 64 untouched-test clips
per class with eight frames per clip. Its scope is these direction-sensitive actions, not general
video understanding.

The current end-to-end video-only milestone trains variable-duration Ego4D GoalStep caption
commits and evaluates 1,337 LongVideoBench questions from 753 videos. DeltaThought scored `48.17%`,
but untouched Qwen under the same commit/KV interface scored `48.09%`. Ordered delta significantly
beat anchor-only and norm-matched noise, and same-KV memory beat memory removal; it did not beat
cross-video, reversed, or within-window-permuted delta. The current model therefore uses learned
delta activations and caption memory without demonstrating temporal order or source-specific
alignment. This is a falsified version of the stronger long-video claim, not a positive benchmark
result. See [the complete long-video results](docs/LONG_VIDEO_RESULTS.md).

The retained LongVideoBench numbers above use a fixed 120-second engineering fallback and are not
dynamic-commit results. The final protocol trains captioning and commit timing only on Ego4D
GoalStep, selects on Ego4D validation, and uses LongVideoBench for evaluation only. A temporary
LongVideoBench label-trained 30-parameter calibration head is retained solely as a contaminated
diagnostic and is excluded from every final model and claim.

For each modality `m`:

```text
d_t^m = DeltaEncoder_m(z_(t-1)^m, z_t^m)
V_(k,t)^m = [A_k^m, d_(k+1)^m, ..., d_t^m]
```

`A_k` is the full embedding at the last caption commit/refresh. Each delta compares immediately
consecutive embeddings. The active Qwen path concatenates every new delta since the anchor; it does
not yet maintain the historical substitute-backbone's fixed-size accumulated state. A caption
commit does not clear the Thinker context: generated caption tokens and earlier visual tokens remain
in the same autoregressive KV cache. A replacement streaming design must separate persistent
language memory from refreshable visual state before claiming constant-memory accumulation.

```text
<FULL_A> <FULL_V>
<DELTA_A> <DELTA_V>
...
<CAPTION_D_V>video change caption</CAPTION_D_V> <FULL_V>
<DELTA_V> ... <CAPTION_D_V>later caption with prior caption still in KV</CAPTION_D_V>
<FULL_V> ...
```

Audio and video may commit at the same timestamp or at different timestamps.

## Verified substitute-backbone functionality

- one-step and full-anchor-plus-accumulated-delta reconstruction;
- exact zero delta for identical consecutive embeddings;
- caption, trigger, and caption-length loss reduction;
- zero/shuffled-delta causal ablations;
- independent A/V resets and full-anchor refreshes;
- final QA whose relational answer is absent from all generated captions;
- exact atomic checkpoint resume after an interrupted run;
- a provenance gate that blocks unverified datasets and models.
- pinned DINOv2, CLAP, and frozen Qwen delta-prefix integration on CPU.
- explicit accumulated-delta versus last-delta-only reconstruction/caption/QA ablations.
- layout-aware video deltas that preserve the DINO CLS token and pool the 16×16 patch grid in 2D.
- four-, eight-, and sixteen-frame real-video accumulation with multi-seed zero/last/shuffle
  controls.

Synthetic checks and bounded real-pilot metrics are reported separately in
`docs/REAL_PILOT_RESULTS.md`.

The selected DINOv2 one-token semantic bottleneck passed all preregistered delta gates on three untouched
test seeds. Hard-token accuracy was `0.762 / 0.707 / 0.785`; learned reconstruction MSE was
`1.9894 / 1.9902 / 1.9905`, compared with raw-pooled `2.0053`. A semantic-token-only frozen
Qwen2.5-7B bridge generated exact captions at `0.758 / 0.715 / 0.789`; full anchors were hidden.

The performance-oriented 65-token layout was also trained for 800 steps on 16-frame clips. Each
clip applies 15 consecutive 65-token deltas to a fixed 65-token state (975 delta-token updates; no
sequence concatenation), then compresses that state to one semantic token for frozen Qwen2.5-7B.
Untouched-test delta accuracy was `0.727 / 0.797 / 0.770`, and unrestricted greedy caption exact
match was `0.727 / 0.785 / 0.773` across three seeds. These measurements establish an engineering
baseline for the delta mechanism, not Qwen2.5-Omni understanding of first-plus-delta inputs.

## Commands

```bash
uv sync --frozen --group dev
uv run ruff check src tests
uv run pytest -q
uv run deltaomni-sanity --config configs/sanity.yaml
uv run deltaomni-verify --config configs/sanity.yaml
uv run deltaomni-provenance --config configs/provenance.yaml
uv run deltaomni-data-audit
# Diagnostic report only; exits zero even when data are blocked:
uv run deltaomni-data-audit --allow-not-ready
uv run deltaomni-annotation-audit
uv run deltaomni-audit-longvideobench --config configs/canonical/longvideobench.yaml
uv run deltaomni-preprocess-ego4d-goalstep \
  --config configs/canonical/ego4d_goalstep.yaml
uv run deltaomni-backbone-smoke
uv run deltaomni-omni-backbone-smoke
uv run deltaomni-language-smoke
uv run deltaomni-delta-setting-sweep --config configs/delta_setting_sweep.yaml
uv run torchrun --standalone --nproc-per-node=4 -m deltaomni.ssv2_semantic_token_pilot \
  --config configs/ssv2_semantic_token_layout65_a6000.yaml
uv run torchrun --standalone --nproc-per-node=4 -m deltaomni.ssv2_semantic_caption_pilot \
  --config configs/ssv2_semantic_caption_layout65_a6000.yaml
uv run torchrun --standalone --nproc-per-node=4 -m deltaomni.ssv2_semantic_token_pilot \
  --config configs/ssv2_semantic_token_layout65_16frames_a6000.yaml
uv run torchrun --standalone --nproc-per-node=4 -m deltaomni.ssv2_semantic_caption_pilot \
  --config configs/ssv2_semantic_caption_layout65_16frames_a6000.yaml
uv run python -m deltaomni.ssv2_generated_caption_eval \
  --config configs/ssv2_semantic_caption_layout65_16frames_test_a6000.yaml \
  --checkpoint-run-id layout65-16f-caption-seed42-validation
uv run deltaomni-report
```

The current NExT-QA Stage 3 preflight uses content-signed caches and four matched vanilla controls.
Run it only from a clean, pushed commit:

```bash
make nextqa-stage3-preflight GPU_IDS=0,1,2,3 NPROC_PER_NODE=4
```

This runs the NExT-QA readiness audit, regenerates the v2 native-Omni joint cache, and evaluates
multimodal, text-only, video-only, and audio-only Qwen baselines sequentially.

Configuration is centralized in `configs/`. Retained runs are stored under unique IDs in
`outputs/sanity/` and `logs/experiments/`.

Official raw datasets and shared read-only annotations live under `/mnt/nfs_shared_data/dataset`.
DeltaThought-owned immutable raw distributions live under
`/mnt/nfs_shared_data/dataset/deltathought/raw`; `/dataset/deltaomni` is legacy read-only context.
Canonical manifests, extracted/derived media, embedding caches, checkpoints, logs, and reports stay
under this project in `intermediates/`, `outputs/`, and `logs/`.

Repository and storage management rules are mandatory and documented in [AGENTS.md](AGENTS.md):
GitHub is authoritative for source/reproducibility material, while NAS is authoritative for raw data
and selected large migration backups.

## Documentation

- [Architecture and losses](docs/ARCHITECTURE.md)
- [Data and model provenance](docs/DATA_POLICY.md)
- [Real-data gated execution plan](docs/REAL_DATA_PLAN.md)
- [Long-video preregistered protocol](docs/LONG_VIDEO_PROTOCOL.md)
- [Long-video video-only results and decision](docs/LONG_VIDEO_RESULTS.md)
- [Current status and retained metrics](docs/STATUS.md)
- [First real-data pilot results](docs/REAL_PILOT_RESULTS.md)
- [Server migration handoff](docs/MIGRATION.md)
- `index.html` is retained as historical research context.
