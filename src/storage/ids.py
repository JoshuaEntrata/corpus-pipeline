import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.storage.read_write import set_csv_field_size_limit


REGISTRY_FIELDS = [
    "source_platform",
    "source_item_id",
    "source_url",
    "text_hash",
    "first_seen_at_utc",
]

set_csv_field_size_limit()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_text(value):
    if value is None:
        value = ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class IdRegistry:
    def __init__(self, path="data/registry/collected_ids.csv"):
        self.path = Path(path)
        self.records = {}
        self.load()

    def load(self):
        self.records = {}
        if not self.path.exists():
            return

        with open(self.path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("source_platform"), row.get("source_item_id"))
                if all(key):
                    self.records[key] = {
                        field: row.get(field) for field in REGISTRY_FIELDS
                    }

    def has(self, source_platform, source_item_id):
        return (source_platform, str(source_item_id)) in self.records

    def add(
        self,
        source_platform,
        source_item_id,
        source_url=None,
        text_hash=None,
        first_seen_at_utc=None,
    ):
        if not source_platform or source_item_id in (None, ""):
            return False

        key = (source_platform, str(source_item_id))
        if key in self.records:
            return False

        self.records[key] = {
            "source_platform": source_platform,
            "source_item_id": str(source_item_id),
            "source_url": source_url,
            "text_hash": text_hash,
            "first_seen_at_utc": first_seen_at_utc or utc_now_iso(),
        }
        return True

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(self.records.values())

    def refresh_from_raw_csv(self, path, default_platform=None):
        path = Path(path)
        if not path.exists():
            return 0

        added = 0
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                source_platform = row.get("source_platform") or default_platform
                source_item_id = row.get("source_item_id") or row.get("id")
                text = (
                    row.get("body_text")
                    or row.get("description")
                    or row.get("title")
                    or row.get("submission")
                    or ""
                )
                if self.add(
                    source_platform=source_platform,
                    source_item_id=source_item_id,
                    source_url=row.get("source_url") or row.get("url"),
                    text_hash=hash_text(text) if text else None,
                    first_seen_at_utc=row.get("collected_at_utc") or None,
                ):
                    added += 1

        return added
