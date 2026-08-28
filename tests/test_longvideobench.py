import io
import json
import tarfile
from pathlib import Path

import pytest

from deltaomni.data.longvideobench import (
    IndexedLongVideoBenchStore,
    MemberView,
    MultipartFile,
    _annotations,
    index_tar,
    load_config,
)


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:") as archive:
        for name, value in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    return output.getvalue()


def test_multipart_tar_index_and_member_view_cross_part_boundaries(tmp_path: Path) -> None:
    payload = _tar_bytes({"videos/a.mp4": b"video-a", "videos/b.mp4": b"video-b-long"})
    boundaries = (731, 3107)
    parts = []
    start = 0
    for index, end in enumerate((*boundaries, len(payload))):
        path = tmp_path / f"videos.tar.part.{index:02d}"
        path.write_bytes(payload[start:end])
        parts.append(path)
        start = end

    with MultipartFile(parts) as archive:
        members = index_tar(archive)
    assert [member.name for member in members] == ["videos/a.mp4", "videos/b.mp4"]

    selected = next(member for member in members if member.name.endswith("b.mp4"))
    with MemberView(MultipartFile(parts), selected) as view:
        assert view.read(5) == b"video"
        assert view.seek(-4, io.SEEK_END) == selected.size - 4
        assert view.read() == b"long"


def test_indexed_store_opens_video_without_extracting_the_archive(tmp_path: Path) -> None:
    payload = _tar_bytes({"videos/a.mp4": b"video-bytes"})
    part = tmp_path / "videos.tar.part.aa"
    part.write_bytes(payload)
    with MultipartFile([part]) as archive:
        member = index_tar(archive)[0]
    subtitles_payload = _tar_bytes({"subtitles/a_en.json": b"[]"})
    subtitles = tmp_path / "subtitles.tar"
    subtitles.write_bytes(subtitles_payload)
    with subtitles.open("rb") as stream:
        subtitle = index_tar(stream)[0]
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema": "deltaomni.longvideobench_archive_index.v1",
                "parts": [{"path": str(part), "bytes": len(payload)}],
                "subtitles_archive": str(subtitles),
                "videos": {"a.mp4": member.__dict__},
                "subtitles": {"a_en.json": subtitle.__dict__},
            }
        ),
        encoding="utf-8",
    )

    store = IndexedLongVideoBenchStore(index)
    with store.open_video("a.mp4") as video:
        assert video.read() == b"video-bytes"
    with store.open_subtitle("a_en.json") as subtitle_view:
        assert subtitle_view.read() == b"[]"


def test_longvideobench_annotations_enforce_hidden_test_labels(tmp_path: Path) -> None:
    row = {
        "id": "video_0",
        "video_id": "video",
        "video_path": "video.mp4",
        "subtitle_path": "video_en.json",
        "question": "What happened?",
        "candidates": ["A", "B"],
        "duration": 10.0,
        "duration_group": 15,
        "question_category": "T1",
        "correct_choice": 1,
    }
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    assert _annotations(path, labeled=True)[0]["correct_choice"] == 1
    with pytest.raises(ValueError, match="label visibility"):
        _annotations(path, labeled=False)


def test_longvideobench_config_pins_official_release_and_all_validation() -> None:
    config = load_config(Path("configs/canonical/longvideobench.yaml"))

    assert config.dataset_revision == "60d1c89c1919a198b73be39c2babb213b29d6a5c"
    assert config.resource_name == "longvideobench"
    assert config.expected_validation_questions == 1337
    assert config.expected_test_questions == 5341
    assert config.expected_archive_videos == 3991
    assert config.expected_annotated_videos == 3761
    assert config.license_acceptance.name == "longvideobench.accepted.json"
