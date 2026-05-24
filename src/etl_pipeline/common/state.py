from __future__ import annotations

from pathlib import Path
from typing import Callable

from etl_pipeline.common.ids import extraction_key, row_key_from_record
from etl_pipeline.common.io import append_csv, read_csv
from etl_pipeline.common.time import utc_now_iso

STATE_FIELDS = ["stage", "record_key", "platform", "id", "associated_id", "processed_at_utc"]


def read_state_keys(path: str | Path) -> set[str]:
    return {row.get("record_key", "") for row in read_csv(path) if row.get("record_key")}


def master_keys(path: str | Path, key_func: Callable[[dict[str, str]], str]) -> set[str]:
    return {key_func(row) for row in read_csv(path)}


def extraction_master_keys(path: str | Path) -> set[str]:
    return master_keys(path, lambda row: extraction_key(row.get("platform", ""), row.get("id", "")))


def row_master_keys(path: str | Path) -> set[str]:
    return master_keys(path, row_key_from_record)


def append_state_rows(path: str | Path, stage: str, rows: list[dict], key_func: Callable[[dict], str]) -> int:
    now = utc_now_iso()
    state_rows = [
        {
            "stage": stage,
            "record_key": key_func(row),
            "platform": row.get("platform", ""),
            "id": row.get("id", ""),
            "associated_id": row.get("associated_id", ""),
            "processed_at_utc": now,
        }
        for row in rows
    ]
    return append_csv(path, state_rows, STATE_FIELDS)

