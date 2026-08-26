# DeltaOmni

Independent modality delta codecs for causal audio/video streams. This repository is intentionally
separate from `omnithought`.

## Current verified scope

The implementation currently operates on deterministic embedding fixtures. They exist only to test
code paths; they are not a new research dataset and are never used as real-method evidence. No
external media or model has been downloaded.

For each modality `m`:

```text
d_t^m = DeltaEncoder_m(z_(t-1)^m, z_t^m)
S_t^m = Accumulate_m(S_(t-1)^m, d_t^m)
z_hat_t^m = Reconstruct_m(A_k^m, S_t^m)
```

`A_k` is the full embedding at the last caption/reset. Each delta compares immediately consecutive
embeddings. A commit generates a scoped change caption, refreshes only that modality's full anchor,
and resets only its delta state.

```text
<FULL_A> <FULL_V>
<DELTA_A> <DELTA_V>
...
<CAPTION_D_V>video change caption</CAPTION_D_V> <FULL_V>
```

Audio and video may commit at the same timestamp or at different timestamps.

## Verified functionality

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

These are synthetic functional checks, not real audio/video performance claims.

The typed delta-to-Qwen projector is implemented and trainable. It has passed a real frozen-Qwen
gradient/loss smoke, but it is not yet semantically aligned on real scoped caption data.

## Commands

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest -q
uv run deltaomni-sanity --config configs/sanity.yaml
uv run deltaomni-verify --config configs/sanity.yaml
uv run deltaomni-provenance --config configs/provenance.yaml
uv run deltaomni-data-audit --allow-not-ready
uv run deltaomni-annotation-audit
uv run deltaomni-backbone-smoke
uv run deltaomni-language-smoke
uv run deltaomni-report
```

Configuration is centralized in `configs/`. Retained runs are stored under unique IDs in
`outputs/sanity/` and `logs/experiments/`.

Official raw datasets and annotations live only under `/mnt/nfs_shared_data/dataset/deltaomni`.
Canonical manifests, extracted/derived media, embedding caches, checkpoints, logs, and reports stay
under this project in `intermediates/`, `outputs/`, and `logs/`.

## Documentation

- [Architecture and losses](docs/ARCHITECTURE.md)
- [Data and model provenance](docs/DATA_POLICY.md)
- [Real-data gated execution plan](docs/REAL_DATA_PLAN.md)
- [Current status and retained metrics](docs/STATUS.md)
- [First real-data pilot results](docs/REAL_PILOT_RESULTS.md)
- [Server migration handoff](docs/MIGRATION.md)
- `index.html` is retained as historical research context.
