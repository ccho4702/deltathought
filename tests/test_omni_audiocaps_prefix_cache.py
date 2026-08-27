from pathlib import Path

import numpy as np
import soundfile as sf

from deltaomni.omni_audiocaps_prefix_cache import _audio_blocks, load_config


def test_audiocaps_prefix_config_uses_four_one_token_deltas() -> None:
    config = load_config(Path("configs/omni_audiocaps_prefix_cache.yaml"))

    assert config.blocks_per_clip == 5
    assert config.expected_audio_tokens == 50
    assert config.train_count == 8192
    assert config.validation_count == 483
    assert config.test_count == 943

    one_second = load_config(Path("configs/omni_audiocaps_prefix_cache_1s.yaml"))
    assert one_second.block_seconds == 1.0
    assert one_second.blocks_per_clip == 10
    assert one_second.expected_audio_tokens == 25
    assert one_second.encoder_batch_size == 10
    assert one_second.runtime.cpu_threads == 16

    poc = load_config(Path("configs/omni_audiocaps_prefix_cache_1s_poc.yaml"))
    assert (poc.train_count, poc.validation_count, poc.test_count) == (1024, 128, 128)
    assert poc.cache_root == one_second.cache_root


def test_audio_blocks_resample_stereo_to_exact_independent_chunks(tmp_path: Path) -> None:
    source_rate = 48_000
    waveform = np.zeros((source_rate * 10, 2), dtype=np.float32)
    path = tmp_path / "audio.flac"
    sf.write(path, waveform, source_rate)

    blocks = _audio_blocks(
        path,
        target_rate=16_000,
        block_seconds=2.0,
        blocks_per_clip=5,
    )

    assert len(blocks) == 5
    assert all(block.shape == (32_000,) for block in blocks)
    assert all(block.flags.c_contiguous for block in blocks)
