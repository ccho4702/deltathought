.PHONY: setup lint test sanity verify provenance data-audit annotation-audit preprocess-nextqa preprocess-ssv2 backbone-smoke omni-backbone-smoke language-smoke ssv2-pilot ssv2-caption-pilot ssv2-semantic-pilot ssv2-semantic-token-pilot ssv2-semantic-token-4gpu ssv2-semantic-token-16frames-4gpu ssv2-delta-search ssv2-semantic-caption ssv2-semantic-caption-16frames ssv2-generated-caption-16frames ssv2-resampler-pilot delta-setting-sweep audioset-timing-pilot nextqa-reconstruction-pilot report verify-all

setup:
	uv sync --group dev

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

sanity:
	uv run deltaomni-sanity --config configs/sanity.yaml

verify:
	uv run deltaomni-verify --config configs/sanity.yaml

provenance:
	uv run deltaomni-provenance --config configs/provenance.yaml

data-audit:
	uv run deltaomni-data-audit --allow-not-ready

annotation-audit:
	uv run deltaomni-annotation-audit

preprocess-nextqa:
	uv run deltaomni-preprocess-nextqa

preprocess-ssv2:
	uv run deltaomni-preprocess-ssv2

backbone-smoke:
	uv run deltaomni-backbone-smoke

omni-backbone-smoke:
	uv run deltaomni-omni-backbone-smoke

language-smoke:
	uv run deltaomni-language-smoke

ssv2-pilot:
	uv run deltaomni-ssv2-pilot --config configs/ssv2_pilot.yaml

ssv2-caption-pilot:
	uv run deltaomni-ssv2-caption-pilot --config configs/ssv2_pilot.yaml

ssv2-semantic-pilot:
	uv run deltaomni-ssv2-semantic-pilot --config configs/ssv2_semantic_pilot.yaml

ssv2-semantic-token-pilot:
	uv run deltaomni-ssv2-semantic-token-pilot \
		--config configs/ssv2_semantic_token_layout65_a6000.yaml

ssv2-semantic-token-4gpu:
	uv run torchrun --standalone --nproc-per-node=4 \
		-m deltaomni.ssv2_semantic_token_pilot \
		--config configs/ssv2_semantic_token_layout65_a6000.yaml

ssv2-semantic-token-16frames-4gpu:
	uv run torchrun --standalone --nproc-per-node=4 \
		-m deltaomni.ssv2_semantic_token_pilot \
		--config configs/ssv2_semantic_token_layout65_16frames_a6000.yaml

ssv2-delta-search:
	uv run deltaomni-ssv2-delta-search --config configs/ssv2_delta_search_a6000.yaml

ssv2-semantic-caption:
	uv run torchrun --standalone --nproc-per-node=4 \
		-m deltaomni.ssv2_semantic_caption_pilot \
		--config configs/ssv2_semantic_caption_layout65_a6000.yaml

ssv2-semantic-caption-16frames:
	uv run torchrun --standalone --nproc-per-node=4 \
		-m deltaomni.ssv2_semantic_caption_pilot \
		--config configs/ssv2_semantic_caption_layout65_16frames_a6000.yaml

ssv2-generated-caption-16frames:
	uv run python -m deltaomni.ssv2_generated_caption_eval \
		--config configs/ssv2_semantic_caption_layout65_16frames_test_a6000.yaml \
		--checkpoint-run-id layout65-16f-caption-seed42-validation

ssv2-resampler-pilot:
	uv run deltaomni-ssv2-resampler-pilot --config configs/ssv2_resampler_pilot.yaml

delta-setting-sweep:
	uv run deltaomni-delta-setting-sweep --config configs/delta_setting_sweep.yaml

audioset-timing-pilot:
	uv run deltaomni-audioset-timing-pilot --config configs/audioset_timing_pilot.yaml

nextqa-reconstruction-pilot:
	uv run deltaomni-nextqa-reconstruction-pilot --config configs/nextqa_reconstruction_pilot.yaml

report:
	uv run deltaomni-report

verify-all: lint test provenance data-audit annotation-audit verify backbone-smoke language-smoke report
