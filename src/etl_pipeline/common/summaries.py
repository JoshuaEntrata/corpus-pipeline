from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def distribution(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "") or "") for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def platform_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return distribution(rows, "platform")


def collection_method_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return distribution(rows, "collection_method")


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return distribution(rows, "category")

