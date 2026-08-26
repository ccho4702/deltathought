# Canonical data format v2

Every dataset adapter must emit `deltaomni.episode.v2` records after preprocessing. Model and
training code may depend on this schema but must not contain dataset-specific field handling.

The machine-readable contract is
[`schemas/deltaomni.episode.v2.schema.json`](../schemas/deltaomni.episode.v2.schema.json). The Python
validator, serializer, and atomic JSONL I/O live in `src/deltaomni/data/schema.py`.

## Storage layout

```text
intermediates/canonical/<dataset>/<dataset-revision>/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

A dataset revision is immutable. Preprocessing refuses to overwrite an existing revision. The
manifest records the episode schema, source-file hashes, preprocessing-config hash, code revision,
split-file hashes, record counts, and field coverage. Loading verifies checksums, counts, record
validity, and cross-split source separation.

The first concrete preprocessor is:

```bash
uv run deltaomni-preprocess-nextqa --config configs/canonical/nextqa.yaml
uv run deltaomni-preprocess-ssv2 --config configs/canonical/ssv2.yaml
```

It caches immutable media hashes and stream metadata under `intermediates/cache/`, displays progress
and ETA, and resumes from those per-media cache entries after interruption. It never writes to the
shared raw-data tree.

Completed NExT-QA revision `official-2021-ann-1955d89e-schema-v2` contains 3,870 train, 570
validation, and 1,000 test episodes with 47,692 human QA items. All split files pass JSON Schema,
checksum, count, round-trip, and source-group disjointness checks. Of 5,440 video containers, 5,406
contain an audio stream and 34 correctly serialize `media.audio` as null. The retained compact
summary is `outputs/reports/canonical_nextqa_v2.json`; generated canonical records remain under
`intermediates/canonical/` and are reproducible from the immutable shared media.

## Null semantics

- `null`: the source dataset does not provide this media or annotation type.
- `[]`: the source dataset defines this annotation type, but this episode has no items.
- Non-empty array: one or more annotations are available.

Missing keys are invalid. Every record contains the same top-level and nested keys.

## Top-level fields

| Field | Meaning |
|---|---|
| `schema` | Always `deltaomni.episode.v2`. |
| `episode_id` | Unique canonical record ID. |
| `dataset`, `dataset_revision`, `split` | Immutable dataset identity and normalized split. |
| `source_id` | Dataset-local source ID. |
| `source_group_id` | Cross-dataset identity used to prevent source leakage. Prefer normalized upstream URL/video ID or media hash. |
| `media` | Always contains nullable `image`, `video`, and `audio` assets. |
| `duration_seconds` | Unified timeline duration; null only for image-only episodes. |
| `temporal_blocks` | Contiguous preprocessing blocks covering the entire temporal episode. |
| `captions` | Always contains nullable `image`, `video`, `audio`, and `joint` arrays. |
| `text` | Always contains nullable `transcript`, `subtitle`, and `ocr` arrays. |
| `events` | Nullable timed event annotations. |
| `qa` | Nullable list of question/answer annotations. |
| `provenance` | Resource, license, annotation, preprocessing, and code provenance. |
| `metadata` | Dataset-specific information that is not a training target. |

## Example

```json
{
  "schema": "deltaomni.episode.v2",
  "episode_id": "nextqa:validation:4010069381",
  "dataset": "nextqa",
  "dataset_revision": "official-2021",
  "split": "validation",
  "source_id": "4010069381",
  "source_group_id": "nextqa:4010069381",
  "media": {
    "image": null,
    "video": {
      "path": "/immutable/4010069381.mp4",
      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "duration_seconds": 2.0,
      "mime_type": "video/mp4",
      "width": 640,
      "height": 480,
      "fps": 30.0,
      "sample_rate": null,
      "channels": null
    },
    "audio": null
  },
  "duration_seconds": 2.0,
  "temporal_blocks": [
    {"block_index": 0, "start_seconds": 0.0, "end_seconds": 2.0}
  ],
  "captions": {"image": null, "video": null, "audio": null, "joint": null},
  "text": {"transcript": null, "subtitle": null, "ocr": null},
  "events": null,
  "qa": [
    {
      "question_id": "6",
      "question": "How do the two men play the instrument?",
      "answer": "roll the handle",
      "choices": ["roll the handle", "tap their feet", "strum the string", "hit with sticks", "pat with hand"],
      "answer_index": 0,
      "question_type": "CH",
      "required_modalities": ["video"],
      "evidence_spans": null,
      "annotation_origin": "human_nextqa",
      "independent_from_captions": true,
      "acceptable_answers": null,
      "dialogue_history": null,
      "turn_index": null
    }
  ],
  "provenance": {
    "resource_name": "nextqa_annotations",
    "source_url": null,
    "license_record": null,
    "annotation_path": null,
    "annotation_sha256": null,
    "preprocessing_config_sha256": null,
    "code_revision": null,
    "processed_at_utc": null
  },
  "metadata": {}
}
```

## Training gates

- Stage 1 accepts any validated temporal record with the required modality.
- Stage 2 requires at least one caption annotation and must never infer missing captions from null.
- Stage 3 requires QA annotations. Independent-transfer evaluation additionally requires every QA
  item to have `independent_from_captions: true`.
- Caption and QA dataset pools must have disjoint `source_group_id` sets, including across dataset
  names.
