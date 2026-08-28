GPU_IDS ?= 0,1,2,3
NPROC_PER_NODE ?= 4
EVAL_GPU_ID ?= 0

.PHONY: setup lint test sanity verify provenance data-audit data-audit-report annotation-audit preprocess-nextqa preprocess-ssv2 preprocess-audioset-strong audit-longvideobench analyze-longvideobench-qa preprocess-ego4d-goalstep-smoke preprocess-ego4d-goalstep cache-ego4d-goalstep-smoke cache-ego4d-goalstep cache-longvideobench-smoke cache-longvideobench ego4d-goalstep-caption-smoke ego4d-goalstep-caption ego4d-goalstep-full-caption-smoke ego4d-goalstep-full-caption backbone-smoke omni-backbone-smoke language-smoke ssv2-pilot ssv2-caption-pilot ssv2-semantic-pilot ssv2-semantic-token-pilot ssv2-semantic-token-4gpu ssv2-semantic-token-16frames-4gpu ssv2-delta-search ssv2-semantic-caption ssv2-semantic-caption-16frames ssv2-generated-caption-16frames ssv2-resampler-pilot delta-setting-sweep audioset-timing-pilot nextqa-reconstruction-pilot nextqa-readiness nextqa-joint-cache-v2 nextqa-joint-cache-s3 nextqa-vanilla-controls-v2 nextqa-vanilla-controls-analysis nextqa-video-lora-smoke nextqa-video-lora-train nextqa-stage3-preflight msrvtt-raw-caption-smoke msrvtt-raw-caption msrvtt-raw-caption-overfit msrvtt-raw-caption-compare msrvtt-continuous-caption-smoke msrvtt-continuous-caption report verify-code verify-research verify-all

setup:
	uv sync --frozen --group dev

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
	uv run --frozen deltaomni-data-audit

data-audit-report:
	uv run --frozen deltaomni-data-audit --allow-not-ready

annotation-audit:
	uv run deltaomni-annotation-audit

preprocess-nextqa:
	uv run deltaomni-preprocess-nextqa

preprocess-ssv2:
	uv run deltaomni-preprocess-ssv2

preprocess-audioset-strong:
	uv run deltaomni-preprocess-audioset-strong

preprocess-msrvtt:
	uv run --frozen deltaomni-preprocess-msrvtt

audit-longvideobench:
	uv run --frozen deltaomni-audit-longvideobench

analyze-longvideobench-qa:
	uv run --frozen deltaomni-analyze-longvideobench-qa \
		--config configs/longvideobench_qa_analysis.yaml

preprocess-ego4d-goalstep:
	uv run --frozen deltaomni-preprocess-ego4d-goalstep

preprocess-ego4d-goalstep-smoke:
	uv run --frozen deltaomni-preprocess-ego4d-goalstep \
		--config configs/canonical/ego4d_goalstep_smoke.yaml

cache-ego4d-goalstep:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_ego4d_goalstep_cache \
		--config configs/omni_ego4d_goalstep_cache.yaml

cache-ego4d-goalstep-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_ego4d_goalstep_cache \
		--config configs/omni_ego4d_goalstep_cache_smoke.yaml

cache-longvideobench-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_longvideobench_cache \
		--config configs/omni_longvideobench_cache_smoke.yaml

cache-longvideobench:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_longvideobench_cache \
		--config configs/omni_longvideobench_cache.yaml

ego4d-goalstep-caption-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.ego4d_goalstep_caption_lora \
		--config configs/ego4d_goalstep_caption_smoke.yaml

ego4d-goalstep-caption:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.ego4d_goalstep_caption_lora \
		--config configs/ego4d_goalstep_caption.yaml

ego4d-goalstep-full-caption-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.ego4d_goalstep_caption_lora \
		--config configs/ego4d_goalstep_full_caption_smoke.yaml

ego4d-goalstep-full-caption:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.ego4d_goalstep_caption_lora \
		--config configs/ego4d_goalstep_full_caption.yaml

msrvtt-video-cache-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_msrvtt_video_cache \
		--config configs/omni_msrvtt_video_cache_smoke.yaml

msrvtt-video-cache:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_msrvtt_video_cache \
		--config configs/omni_msrvtt_video_cache.yaml

msrvtt-video-caption-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.audiocaps_caption_lora \
		--config configs/msrvtt_video_caption_smoke.yaml

msrvtt-video-caption:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.audiocaps_caption_lora \
		--config configs/msrvtt_video_caption.yaml

msrvtt-raw-caption-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.msrvtt_raw_caption_lora \
		--config configs/msrvtt_raw_caption_smoke.yaml

msrvtt-raw-caption:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.msrvtt_raw_caption_lora \
		--config configs/msrvtt_raw_caption.yaml

msrvtt-raw-caption-overfit:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.msrvtt_raw_caption_lora \
		--config configs/msrvtt_raw_caption_overfit.yaml

msrvtt-raw-caption-compare:
	CUDA_VISIBLE_DEVICES=$(EVAL_GPU_ID) uv run --frozen \
		deltaomni-compare-msrvtt-raw-caption \
		--config configs/msrvtt_raw_caption_compare.yaml

msrvtt-continuous-caption-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.msrvtt_continuous_caption_lora \
		--config configs/msrvtt_continuous_caption_smoke.yaml

msrvtt-continuous-caption:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.msrvtt_continuous_caption_lora \
		--config configs/msrvtt_continuous_caption.yaml

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

nextqa-readiness:
	uv run --frozen deltaomni-data-audit --source nextqa \
		--output outputs/reports/data_readiness_nextqa.json

nextqa-joint-cache-v2:
	$(MAKE) nextqa-readiness
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_nextqa_joint_cache \
		--config configs/omni_nextqa_joint_poc.yaml

nextqa-joint-cache-s3:
	$(MAKE) nextqa-readiness
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_nextqa_joint_cache \
		--config configs/omni_nextqa_joint_s3.yaml

nextqa-vanilla-controls-v2:
	$(MAKE) nextqa-readiness
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.omni_vanilla_baseline \
		--config configs/qwen2_5_omni_vanilla_baseline_poc.yaml

nextqa-vanilla-controls-analysis:
	uv run --frozen deltaomni-analyze-omni-vanilla

nextqa-video-lora-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.nextqa_video_lora \
		--config configs/nextqa_video_lora_smoke.yaml

nextqa-video-lora-train:
	CUDA_VISIBLE_DEVICES=$(GPU_IDS) uv run --frozen torchrun --standalone \
		--nproc-per-node=$(NPROC_PER_NODE) -m deltaomni.nextqa_video_lora \
		--config configs/nextqa_video_lora.yaml

nextqa-stage3-preflight:
	$(MAKE) nextqa-joint-cache-v2
	$(MAKE) nextqa-vanilla-controls-v2
	$(MAKE) nextqa-vanilla-controls-analysis

report:
	uv run deltaomni-report

verify-code: lint test provenance

verify-research: data-audit annotation-audit verify omni-backbone-smoke report

verify-all: verify-code verify-research
