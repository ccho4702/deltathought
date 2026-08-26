.PHONY: setup lint test sanity verify provenance data-audit annotation-audit backbone-smoke language-smoke ssv2-pilot ssv2-caption-pilot ssv2-semantic-pilot ssv2-resampler-pilot delta-setting-sweep audioset-timing-pilot nextqa-reconstruction-pilot report verify-all

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

backbone-smoke:
	uv run deltaomni-backbone-smoke

language-smoke:
	uv run deltaomni-language-smoke

ssv2-pilot:
	uv run deltaomni-ssv2-pilot --config configs/ssv2_pilot.yaml

ssv2-caption-pilot:
	uv run deltaomni-ssv2-caption-pilot --config configs/ssv2_pilot.yaml

ssv2-semantic-pilot:
	uv run deltaomni-ssv2-semantic-pilot --config configs/ssv2_semantic_pilot.yaml

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
