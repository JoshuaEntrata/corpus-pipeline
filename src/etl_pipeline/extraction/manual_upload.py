from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from etl_pipeline.common.io import write_csv, write_json
from etl_pipeline.common.paths import local_data_dir, stage_run_dir
from etl_pipeline.common.time import utc_now_iso
from etl_pipeline.extraction.schemas import EXTRACTION_FIELDS, normalize_extraction_row


DEFAULT_MANUAL_INPUT_DIR = "local_data/manual_upload"
DEFAULT_MANUAL_OUTPUT_FILE = "manual_extraction_raw.csv"


@dataclass
class ManualUploadStats:
    input_files: int = 0
    input_rows: int = 0
    output_rows: int = 0
    skipped_empty_text_rows: int = 0
    invalid_json_rows: int = 0
    json_results_extracted: int = 0
    rows_by_file: dict[str, int] = field(default_factory=dict)


def load_manual_upload_rows(input_dir: str | Path = DEFAULT_MANUAL_INPUT_DIR) -> tuple[list[dict[str, str]], ManualUploadStats]:
    manual_dir = Path(input_dir)
    stats = ManualUploadStats()
    rows: list[dict[str, str]] = []
    id_counts: dict[tuple[str, str], int] = {}

    for csv_path in sorted(manual_dir.glob("*.csv")):
        stats.input_files += 1
        file_rows = 0
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                stats.input_rows += 1
                extracted = _extract_source_row(csv_path, row_number, row, stats, id_counts)
                rows.extend(extracted)
                file_rows += len(extracted)
        stats.rows_by_file[csv_path.name] = file_rows

    stats.output_rows = len(rows)
    return rows, stats


def write_manual_upload_outputs(
    input_dir: str | Path = DEFAULT_MANUAL_INPUT_DIR,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows, stats = load_manual_upload_rows(input_dir)
    output = Path(output_path) if output_path else Path("local_data/manual") / DEFAULT_MANUAL_OUTPUT_FILE
    write_csv(output, rows, EXTRACTION_FIELDS)
    summary = {
        "stage": "manual_upload",
        "input_dir": str(input_dir),
        "output_file": str(output),
        **asdict(stats),
    }
    write_json(output.parent / "summary.json", summary)
    return summary


def run_manual_upload_extraction(
    pipeline_config: dict[str, Any],
    *,
    run_id: str,
    input_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manual_config = pipeline_config.get("manual_upload", {})
    source_dir = Path(input_dir or manual_config.get("input_dir", DEFAULT_MANUAL_INPUT_DIR))
    manual_dir = local_data_dir(pipeline_config) / "manual"
    output = Path(output_path or manual_config.get("output_file", manual_dir / DEFAULT_MANUAL_OUTPUT_FILE))

    rows, stats = load_manual_upload_rows(source_dir)
    run_dir = stage_run_dir(pipeline_config, "manual", run_id)
    write_csv(output, rows, EXTRACTION_FIELDS)
    write_csv(run_dir / DEFAULT_MANUAL_OUTPUT_FILE, rows, EXTRACTION_FIELDS)

    summary = {
        "stage": "manual_upload",
        "run_id": run_id,
        "input_dir": str(source_dir),
        "output_file": str(output),
        **asdict(stats),
    }
    write_json(output.parent / "summary.json", summary)
    write_json(run_dir / "summary.json", summary)
    return summary


def _extract_source_row(
    csv_path: Path,
    row_number: int,
    row: dict[str, str],
    stats: ManualUploadStats,
    id_counts: dict[tuple[str, str], int],
) -> list[dict[str, str]]:
    if _value(row, "json"):
        return _extract_json_results(csv_path, row_number, row, stats, id_counts)

    text = _value(row, "text")
    if not text:
        stats.skipped_empty_text_rows += 1
        return []

    platform = _value(row, "platform", "source_platform", "source") or "manual"
    source_id = _value(row, "id", "source_item_id", "post_id")
    row_id = _unique_manual_id(platform, source_id, csv_path, row_number, row, id_counts)
    metadata = _metadata(
        csv_path,
        row_number,
        {
            "source_item_id": source_id,
            "original_columns": {key: value for key, value in row.items() if key and value},
        },
    )
    return [
        _normalized_row(
            platform=platform,
            collection_method=_value(row, "collection_method") or "manual_upload",
            id_=row_id,
            text=text,
            url=_value(row, "url"),
            metadata=metadata,
            provided_language_label=_value(row, "provided_language_label", "language_label"),
            provided_classification_label=_value(row, "provided_classification_label", "classification_label"),
        )
    ]


def _extract_json_results(
    csv_path: Path,
    row_number: int,
    row: dict[str, str],
    stats: ManualUploadStats,
    id_counts: dict[tuple[str, str], int],
) -> list[dict[str, str]]:
    try:
        payload = json.loads(_value(row, "json"))
    except json.JSONDecodeError:
        stats.invalid_json_rows += 1
        return []

    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        stats.invalid_json_rows += 1
        return []

    output = []
    platform = _value(row, "platform", "source_platform", "source") or "manual"
    collection_method = _value(row, "collection_method") or "manual_upload_json"
    query = _value(row, "query")
    for result_index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("message") or item.get("message_rich") or "").strip()
        if not text:
            stats.skipped_empty_text_rows += 1
            continue
        source_id = str(item.get("post_id") or item.get("id") or "").strip()
        item_id = _unique_manual_id(platform, source_id, csv_path, row_number, item, id_counts)
        author = item.get("author") if isinstance(item.get("author"), dict) else {}
        metadata = _metadata(
            csv_path,
            row_number,
            {
                "query": query,
                "result_index": result_index,
                "result_type": item.get("type"),
                "comments_count": item.get("comments_count"),
                "reactions_count": item.get("reactions_count"),
                "reshare_count": item.get("reshare_count"),
                "reactions": item.get("reactions"),
                "author_id": author.get("id"),
                "author_url": author.get("url"),
                "external_url": item.get("external_url"),
                "associated_group_id": item.get("associated_group_id"),
                "attached_post_url": item.get("attached_post_url"),
            },
        )
        output.append(
            _normalized_row(
                platform=platform,
                collection_method=collection_method,
                id_=item_id,
                text=text,
                url=str(item.get("url") or ""),
                author=str(author.get("name") or ""),
                created_at_utc=_timestamp_to_utc(item.get("timestamp")),
                metadata=metadata,
                provided_language_label=_value(row, "provided_language_label", "language_label"),
                provided_classification_label=_value(row, "provided_classification_label", "classification_label"),
            )
        )
    stats.json_results_extracted += len(output)
    return output


def _normalized_row(
    *,
    platform: str,
    collection_method: str,
    id_: str,
    text: str,
    url: str = "",
    author: str = "",
    created_at_utc: str = "",
    metadata: dict[str, Any] | None = None,
    provided_language_label: str = "",
    provided_classification_label: str = "",
) -> dict[str, str]:
    return normalize_extraction_row(
        {
            "platform": platform.strip(),
            "collection_method": collection_method.strip(),
            "id": id_,
            "text": text.strip(),
            "title": "",
            "author": author,
            "created_at_utc": created_at_utc,
            "url": url.strip(),
            "comments_json": "[]",
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            "provided_language_label": provided_language_label.strip(),
            "provided_classification_label": provided_classification_label.strip(),
            "extracted_at_utc": utc_now_iso(),
        }
    )


def _unique_manual_id(
    platform: str,
    source_id: str,
    csv_path: Path,
    row_number: int,
    row: dict[str, Any],
    id_counts: dict[tuple[str, str], int],
) -> str:
    candidate = source_id.strip() if source_id else ""
    if not candidate:
        candidate = f"{csv_path.stem}_{row_number:05d}_{_short_hash(row)}"
    key = (platform, candidate)
    id_counts[key] = id_counts.get(key, 0) + 1
    if id_counts[key] == 1:
        return candidate
    return f"{candidate}__manual_{row_number:05d}_{_short_hash(row)}"


def _short_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


def _timestamp_to_utc(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(timestamp, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata(csv_path: Path, row_number: int, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": csv_path.name,
        "source_row_number": row_number,
        **{key: value for key, value in values.items() if value not in (None, "", [], {})},
    }


def _value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""
