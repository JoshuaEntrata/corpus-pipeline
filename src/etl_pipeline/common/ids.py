from __future__ import annotations

from typing import Mapping


def normalize_key_part(value: object) -> str:
    return str(value or "").strip()


def extraction_key(platform: str, id_: str) -> str:
    return "|".join([normalize_key_part(platform), normalize_key_part(id_)])


def row_key(platform: str, category: str, id_: str, associated_id: str) -> str:
    return "|".join(
        [
            normalize_key_part(platform),
            normalize_key_part(category),
            normalize_key_part(id_),
            normalize_key_part(associated_id),
        ]
    )


def row_key_from_record(row: Mapping[str, object]) -> str:
    return row_key(
        str(row.get("platform", "")),
        str(row.get("category", "")),
        str(row.get("id", "")),
        str(row.get("associated_id", "")),
    )

