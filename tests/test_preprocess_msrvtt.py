import json
from pathlib import Path

import pytest

from deltaomni.data.preprocess_msrvtt import _metadata, load_config


def test_msrvtt_config_uses_official_complete_splits() -> None:
    config = load_config(Path("configs/canonical/msrvtt.yaml"))

    assert config.dataset == "msrvtt"
    assert set(config.annotations) == {"train", "validation", "test"}
    assert config.chunk_seconds == 1.0


def test_msrvtt_metadata_requires_twenty_nonempty_captions(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"video1": {str(i): f"caption {i}" for i in range(20)}}))

    assert len(_metadata(path)["video1"]) == 20

    path.write_text(json.dumps({"video1": {"0": "only one"}}))
    with pytest.raises(ValueError, match="20 captions"):
        _metadata(path)
