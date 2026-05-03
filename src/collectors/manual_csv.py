import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import RAW_COLLECTION_FIELDS
from src.storage.ids import IdRegistry, hash_text
from src.storage.read_write import append_csv_rows, set_csv_field_size_limit


SOURCE_PLATFORM = "manual_csv"
DEFAULT_INPUT_FOLDER = Path("data/manual_uploads")
DEFAULT_OUTPUT_CSV = Path("data/raw/manual_csv_scraped.csv")

set_csv_field_size_limit()

ID_COLUMNS = ["source_item_id", "id", "post_id", "comment_id", "row_id"]
TEXT_COLUMNS = ["body_text", "text", "post_text", "comment_text", "content", "message"]
TITLE_COLUMNS = ["title", "post_title"]
URL_COLUMNS = ["source_url", "url", "permalink", "link"]
CREATED_COLUMNS = ["created_at_utc", "created_utc", "created_at", "date", "timestamp"]
AUTHOR_COLUMNS = ["author_id", "author", "username", "user_name", "profile_name"]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_text(text):
    if text in (None, ""):
        return None
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def first_value(row, columns):
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


def hash_identifier(value):
    if not value:
        return None

    salt = os.getenv("AUTHOR_HASH_SALT", "")
    payload = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generated_item_id(file_path, row_number, text):
    payload = f"{file_path.name}:{row_number}:{text or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def sanitized_raw_row(row):
    clean_row = dict(row)
    for column in AUTHOR_COLUMNS:
        clean_row.pop(column, None)
    return clean_row


def map_manual_row(row, file_path, row_number, run_id, collected_at_utc):
    body_text = clean_text(first_value(row, TEXT_COLUMNS))
    title = clean_text(first_value(row, TITLE_COLUMNS))
    description = clean_text(row.get("description"))
    source_item_id = first_value(row, ID_COLUMNS) or generated_item_id(
        file_path, row_number, body_text or title or description
    )
    source_url = first_value(row, URL_COLUMNS)
    author_value = first_value(row, AUTHOR_COLUMNS)

    return {
        "run_id": run_id,
        "source_platform": SOURCE_PLATFORM,
        "source_item_id": str(source_item_id),
        "source_url": source_url,
        "collection_method": "manual_csv",
        "collection_query": file_path.name,
        "collected_at_utc": collected_at_utc,
        "created_at_utc": first_value(row, CREATED_COLUMNS),
        "author_id_hash": hash_identifier(author_value),
        "title": title,
        "body_text": body_text,
        "description": description,
        "transcript": clean_text(row.get("transcript")),
        "comments_json": row.get("comments_json") or None,
        "engagement_json": row.get("engagement_json") or None,
        "raw_json": json.dumps(sanitized_raw_row(row), ensure_ascii=False),
        "manual_file_name": file_path.name,
    }


def collect_from_folder(
    input_folder=DEFAULT_INPUT_FOLDER,
    output_csv=DEFAULT_OUTPUT_CSV,
    registry_path="data/registry/collected_ids.csv",
    run_id=None,
):
    input_folder = Path(input_folder)
    output_csv = Path(output_csv)
    run_id = run_id or utc_now_iso()
    collected_at_utc = utc_now_iso()
    registry = IdRegistry(registry_path)

    if not input_folder.exists():
        input_folder.mkdir(parents=True, exist_ok=True)
        return 0, 0, 0

    collected = 0
    skipped = 0
    failed = 0
    rows_to_write = []

    for file_path in sorted(input_folder.glob("*.csv")):
        try:
            with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    failed += 1
                    continue

                for row_number, row in enumerate(reader, start=2):
                    record = map_manual_row(
                        row, file_path, row_number, run_id, collected_at_utc
                    )
                    has_text = any(
                        record.get(column)
                        for column in ("title", "body_text", "description", "transcript")
                    )
                    if not has_text:
                        failed += 1
                        continue

                    if registry.has(SOURCE_PLATFORM, record["source_item_id"]):
                        skipped += 1
                        continue

                    registry.add(
                        SOURCE_PLATFORM,
                        record["source_item_id"],
                        source_url=record["source_url"],
                        text_hash=hash_text(record["body_text"] or record["title"]),
                        first_seen_at_utc=record["collected_at_utc"],
                    )
                    rows_to_write.append(record)
                    collected += 1
        except Exception:
            failed += 1

    append_csv_rows(output_csv, rows_to_write, RAW_COLLECTION_FIELDS)
    registry.save()

    return collected, skipped, failed


if __name__ == "__main__":
    result = collect_from_folder()
    print(
        "Manual CSV results - "
        f"Collected: {result[0]} | Skipped: {result[1]} | Failed: {result[2]}"
    )
