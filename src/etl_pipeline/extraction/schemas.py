EXTRACTION_FIELDS = [
    "platform",
    "collection_method",
    "id",
    "text",
    "title",
    "author",
    "created_at_utc",
    "url",
    "comments_json",
    "metadata_json",
    "extracted_at_utc",
]


def normalize_extraction_row(row: dict) -> dict:
    normalized = {field: row.get(field, "") for field in EXTRACTION_FIELDS}
    normalized["comments_json"] = normalized.get("comments_json") or "[]"
    normalized["metadata_json"] = normalized.get("metadata_json") or "{}"
    return normalized

