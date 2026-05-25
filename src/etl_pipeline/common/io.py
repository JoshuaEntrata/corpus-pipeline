from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

def _increase_csv_field_size_limit() -> None:
    max_size = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


_increase_csv_field_size_limit()

def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    csv_path = Path(path)
    ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    rows_list = list(rows)
    if not rows_list:
        return 0
    csv_path = Path(path)
    ensure_parent(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != fieldnames:
            write_csv(csv_path, read_csv(csv_path), fieldnames)
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows_list:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return len(rows_list)


def read_json(path: str | Path, default: Any | None = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return default
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    json_path = Path(path)
    ensure_parent(json_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def json_dumps(value: Any, fallback: Any) -> str:
    if value in (None, ""):
        value = fallback
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def json_loads(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback
