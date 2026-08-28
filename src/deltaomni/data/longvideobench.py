from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import yaml

from deltaomni.provenance import audit as audit_provenance
from deltaomni.provenance import require_approved
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class LongVideoBenchConfig:
    dataset_revision: str
    resource_name: str
    raw_root: Path
    validation_annotations: Path
    test_annotations: Path
    video_parts_glob: str
    subtitles_archive: Path
    license_acceptance: Path
    expected_validation_questions: int
    expected_test_questions: int
    expected_question_categories: int
    expected_archive_videos: int
    expected_annotated_videos: int
    index_path: Path
    frozen_validation_manifest: Path
    report_path: Path


@dataclass(frozen=True)
class TarMember:
    name: str
    offset: int
    size: int


def load_config(path: Path) -> LongVideoBenchConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = LongVideoBenchConfig(
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        raw_root=resolve(raw["raw_root"]),
        validation_annotations=resolve(raw["validation_annotations"]),
        test_annotations=resolve(raw["test_annotations"]),
        video_parts_glob=str(raw["video_parts_glob"]),
        subtitles_archive=resolve(raw["subtitles_archive"]),
        license_acceptance=resolve(raw["license_acceptance"]),
        expected_validation_questions=int(raw["expected_validation_questions"]),
        expected_test_questions=int(raw["expected_test_questions"]),
        expected_question_categories=int(raw["expected_question_categories"]),
        expected_archive_videos=int(raw["expected_archive_videos"]),
        expected_annotated_videos=int(raw["expected_annotated_videos"]),
        index_path=resolve(raw["index_path"]),
        frozen_validation_manifest=resolve(raw["frozen_validation_manifest"]),
        report_path=resolve(raw["report_path"]),
    )
    if not config.dataset_revision or not config.resource_name or min(
        config.expected_validation_questions,
        config.expected_test_questions,
        config.expected_question_categories,
        config.expected_archive_videos,
        config.expected_annotated_videos,
    ) <= 0:
        raise ValueError("LongVideoBench revision and expected counts are required")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _huggingface_metadata(
    raw_root: Path, path: Path, expected_revision: str
) -> dict[str, str]:
    metadata = raw_root / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0] != expected_revision:
        raise ValueError(f"Hugging Face revision mismatch for {path}")
    return {"revision": lines[0], "etag": lines[1]}


class MultipartFile(io.RawIOBase):
    """Seekable read-only view over bytewise-concatenated files."""

    def __init__(self, parts: list[Path]) -> None:
        super().__init__()
        if not parts or any(not part.is_file() for part in parts):
            raise FileNotFoundError("Multipart archive is incomplete")
        self.parts = tuple(parts)
        self.sizes = tuple(part.stat().st_size for part in self.parts)
        self.starts = []
        total = 0
        for size in self.sizes:
            self.starts.append(total)
            total += size
        self.total_size = total
        self.position = 0
        self._part_index: int | None = None
        self._stream: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if target < 0:
            raise ValueError("Cannot seek before a multipart archive")
        self.position = min(target, self.total_size)
        return self.position

    def _open_part(self, index: int) -> BinaryIO:
        if self._part_index != index:
            if self._stream is not None:
                self._stream.close()
            self._stream = self.parts[index].open("rb")
            self._part_index = index
        assert self._stream is not None
        self._stream.seek(self.position - self.starts[index])
        return self._stream

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed multipart archive")
        if self.position >= self.total_size or size == 0:
            return b""
        available_total = self.total_size - self.position
        remaining = available_total if size < 0 else min(size, available_total)
        chunks = []
        while remaining:
            index = bisect.bisect_right(self.starts, self.position) - 1
            stream = self._open_part(index)
            available = self.sizes[index] - (self.position - self.starts[index])
            chunk = stream.read(min(remaining, available))
            if not chunk:
                raise OSError(f"Unexpected EOF in multipart archive part {self.parts[index]}")
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


class MemberView(io.RawIOBase):
    """Bounded seekable view of one indexed member in a multipart archive."""

    def __init__(self, archive: MultipartFile, member: TarMember) -> None:
        super().__init__()
        self.archive = archive
        self.member = member
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.member.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if not 0 <= target <= self.member.size:
            raise ValueError("Seek is outside the indexed member")
        self.position = target
        return target

    def read(self, size: int = -1) -> bytes:
        remaining = self.member.size - self.position
        requested = remaining if size < 0 else min(size, remaining)
        if requested <= 0:
            return b""
        self.archive.seek(self.member.offset + self.position)
        value = self.archive.read(requested)
        self.position += len(value)
        return value

    def close(self) -> None:
        self.archive.close()
        super().close()


def index_tar(fileobj: BinaryIO) -> list[TarMember]:
    members = []
    with tarfile.open(fileobj=fileobj, mode="r:") as archive:
        for member in archive:
            if member.isfile():
                members.append(TarMember(member.name, member.offset_data, member.size))
    return members


class IndexedLongVideoBenchStore:
    def __init__(self, index_path: Path) -> None:
        value = json.loads(index_path.read_text(encoding="utf-8"))
        if value.get("schema") != "deltaomni.longvideobench_archive_index.v1":
            raise ValueError(f"Unsupported LongVideoBench archive index: {index_path}")
        self.parts = [Path(item["path"]) for item in value["parts"]]
        self.subtitles_archive = Path(value["subtitles_archive"])
        self.videos = {
            name: TarMember(**member) for name, member in value["videos"].items()
        }
        self.subtitles = {
            name: TarMember(**member) for name, member in value["subtitles"].items()
        }

    @staticmethod
    def _resolve(values: dict[str, TarMember], name: str) -> TarMember:
        member = values.get(PurePosixPath(name).name)
        if member is None:
            raise FileNotFoundError(f"LongVideoBench archive member is not indexed: {name}")
        return member

    def open_video(self, name: str) -> MemberView:
        return MemberView(MultipartFile(self.parts), self._resolve(self.videos, name))

    def open_subtitle(self, name: str) -> MemberView:
        archive = MultipartFile([self.subtitles_archive])
        return MemberView(archive, self._resolve(self.subtitles, name))


def _keyed_by_basename(members: list[TarMember], suffix: str) -> dict[str, TarMember]:
    result = {}
    for member in members:
        name = PurePosixPath(member.name).name
        if not name.endswith(suffix):
            continue
        if name in result:
            raise ValueError(f"Duplicate archive basename: {name}")
        result[name] = member
    return result


def _annotations(path: Path, *, labeled: bool) -> list[dict[str, Any]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"LongVideoBench annotations must be a list: {path}")
    required = {
        "id",
        "video_id",
        "video_path",
        "subtitle_path",
        "question",
        "candidates",
        "duration",
        "duration_group",
        "question_category",
    }
    seen = set()
    for row in values:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Malformed LongVideoBench row in {path}")
        if labeled != ("correct_choice" in row):
            raise ValueError(f"Unexpected LongVideoBench label visibility in {path}")
        if row["id"] in seen or not row["candidates"]:
            raise ValueError(f"Duplicate ID or empty candidates in {path}: {row['id']}")
        if labeled and not 0 <= int(row["correct_choice"]) < len(row["candidates"]):
            raise ValueError(f"Invalid correct choice in {path}: {row['id']}")
        if float(row["duration"]) <= 0:
            raise ValueError(f"Invalid duration in {path}: {row['id']}")
        seen.add(row["id"])
    return values


def _acceptance(path: Path, dataset_revision: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset", "terms_url", "accepted_at_utc", "accepted_by"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError(f"Malformed LongVideoBench acceptance record: {path}")
    if value["dataset"] != "LongVideoBench":
        raise ValueError("LongVideoBench acceptance record names another dataset")
    return {**value, "dataset_revision": dataset_revision}


def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    provenance = audit_provenance(provenance_path)
    require_approved(provenance, [config.resource_name])
    validation = _annotations(config.validation_annotations, labeled=True)
    test = _annotations(config.test_annotations, labeled=False)
    if len(validation) != config.expected_validation_questions:
        raise ValueError("LongVideoBench validation count does not match the pinned release")
    if len(test) != config.expected_test_questions:
        raise ValueError("LongVideoBench test count does not match the pinned release")
    validation_ids = {str(row["id"]) for row in validation}
    test_ids = {str(row["id"]) for row in test}
    validation_videos = {str(row["video_id"]) for row in validation}
    test_videos = {str(row["video_id"]) for row in test}
    if validation_ids & test_ids or validation_videos & test_videos:
        raise ValueError("LongVideoBench validation/test source overlap")
    categories = {str(row["question_category"]) for row in validation + test}
    if len(categories) != config.expected_question_categories:
        raise ValueError("LongVideoBench question category count does not match the pinned release")

    parts = sorted(config.raw_root.glob(config.video_parts_glob))
    with MultipartFile(parts) as multipart:
        video_members = index_tar(multipart)
    with tarfile.open(config.subtitles_archive, mode="r:") as subtitles:
        subtitle_members = [
            TarMember(member.name, member.offset_data, member.size)
            for member in subtitles
            if member.isfile()
        ]
    videos = _keyed_by_basename(video_members, ".mp4")
    subtitle_files = _keyed_by_basename(subtitle_members, ".json")
    rows = validation + test
    annotated_videos = {str(row["video_id"]) for row in rows}
    if len(videos) != config.expected_archive_videos:
        raise ValueError("LongVideoBench archive video count does not match the pinned release")
    if len(annotated_videos) != config.expected_annotated_videos:
        raise ValueError("LongVideoBench annotated video count does not match the pinned release")
    missing_video = sorted({str(row["video_path"]) for row in rows} - videos.keys())
    missing_subtitle = sorted({str(row["subtitle_path"]) for row in rows} - subtitle_files.keys())
    if missing_video or missing_subtitle:
        raise ValueError(
            f"LongVideoBench archive coverage failed: videos={len(missing_video)}, "
            f"subtitles={len(missing_subtitle)}"
        )

    acceptance = _acceptance(config.license_acceptance, config.dataset_revision)
    source_metadata = {
        path.name: _huggingface_metadata(config.raw_root, path, config.dataset_revision)
        for path in (
            config.validation_annotations,
            config.test_annotations,
            config.subtitles_archive,
            *parts,
        )
    }
    annotation_hashes = {
        "validation": _sha256(config.validation_annotations),
        "test": _sha256(config.test_annotations),
    }
    index = {
        "schema": "deltaomni.longvideobench_archive_index.v1",
        "dataset_revision": config.dataset_revision,
        "parts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                **source_metadata[path.name],
            }
            for path in parts
        ],
        "videos": {name: asdict(member) for name, member in sorted(videos.items())},
        "subtitles_archive": str(config.subtitles_archive),
        "subtitles": {
            name: asdict(member) for name, member in sorted(subtitle_files.items())
        },
        "annotation_sha256": annotation_hashes,
        "source_metadata": source_metadata,
        "provenance_sha256": _sha256(provenance_path),
    }
    frozen = {
        "schema": "deltaomni.longvideobench_frozen_validation.v1",
        "dataset_revision": config.dataset_revision,
        "annotation_sha256": annotation_hashes["validation"],
        "selection": "all_official_validation_questions_in_release_order",
        "questions": [
            {
                "id": row["id"],
                "video_id": row["video_id"],
                "duration_group": row["duration_group"],
                "question_category": row["question_category"],
            }
            for row in validation
        ],
    }
    report = {
        "schema": "deltaomni.longvideobench_readiness.v1",
        "status": "ready" if acceptance else "blocked_missing_license_acceptance",
        "ready": acceptance is not None,
        "dataset_revision": config.dataset_revision,
        "validation_questions": len(validation),
        "test_questions_without_labels": len(test),
        "unique_annotated_videos": len(annotated_videos),
        "video_archive_members": len(videos),
        "subtitle_archive_members": len(subtitle_files),
        "question_categories": sorted(categories),
        "duration_groups": sorted({int(row["duration_group"]) for row in rows}),
        "annotation_sha256": annotation_hashes,
        "license_acceptance": acceptance,
        "provenance_approved": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.index_path, index)
    _atomic_json(config.frozen_validation_manifest, frozen)
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and index official LongVideoBench")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/canonical/longvideobench.yaml")
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.provenance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
