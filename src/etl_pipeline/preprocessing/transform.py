from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from etl_pipeline.common.ids import row_key_from_record
from etl_pipeline.common.io import json_loads
from etl_pipeline.common.text import clean_text

ROOT_CATEGORY = {
    "reddit": "post",
    "youtube": "video",
    "twitter": "post",
}


@dataclass
class TransformStats:
    input_rows: int = 0
    output_rows: int = 0
    dropped_empty_text_rows: int = 0
    deduplicated_rows: int = 0
    comments_exploded: int = 0
    replies_exploded: int = 0
    invalid_comments_json_rows: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def transform_extraction_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], TransformStats]:
    stats = TransformStats(input_rows=len(raw_rows))
    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw in raw_rows:
        root_id = str(raw.get("id", "") or "")
        platform = str(raw.get("platform", "") or "")
        collection_method = str(raw.get("collection_method", "") or "")
        root_text = _combine_title_and_text(raw.get("title", ""), raw.get("text", ""))
        root_row = {
            "platform": platform,
            "collection_method": collection_method,
            "id": root_id,
            "text": clean_text(root_text),
            "category": ROOT_CATEGORY.get(platform, "post"),
            "associated_id": root_id,
            "source_url": str(raw.get("url", "") or ""),
        }
        _append_if_valid(root_row, output, seen, stats)

        comments = json_loads(raw.get("comments_json", "[]"), [])
        if not isinstance(comments, list):
            comments = []
            stats.invalid_comments_json_rows += 1
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            comment_row = _comment_to_row(comment, raw, root_id, "comment")
            if _append_if_valid(comment_row, output, seen, stats):
                stats.comments_exploded += 1
            for reply in _iter_replies(comment):
                reply_row = _comment_to_row(reply, raw, root_id, "reply")
                if _append_if_valid(reply_row, output, seen, stats):
                    stats.replies_exploded += 1

    stats.output_rows = len(output)
    return output, stats


def _combine_title_and_text(title: object, text: object) -> str:
    title_text = clean_text(title)
    body_text = clean_text(text)
    if title_text and body_text and title_text not in body_text:
        return f"{title_text} {body_text}"
    return body_text or title_text


def _comment_to_row(comment: dict[str, Any], raw: dict[str, Any], root_id: str, category: str) -> dict[str, str]:
    return {
        "platform": str(raw.get("platform", "") or ""),
        "collection_method": str(raw.get("collection_method", "") or ""),
        "id": str(comment.get("id", "") or ""),
        "text": clean_text(comment.get("text", "")),
        "category": category,
        "associated_id": root_id,
        "source_url": str(comment.get("url", "") or raw.get("url", "") or ""),
    }


def _iter_replies(comment: dict[str, Any]) -> list[dict[str, Any]]:
    replies = comment.get("replies", [])
    if not isinstance(replies, list):
        return []
    return [reply for reply in replies if isinstance(reply, dict)]


def _append_if_valid(
    row: dict[str, str],
    output: list[dict[str, str]],
    seen: set[str],
    stats: TransformStats,
) -> bool:
    if not row.get("id") or not row.get("text"):
        stats.dropped_empty_text_rows += 1
        return False
    key = row_key_from_record(row)
    if key in seen:
        stats.deduplicated_rows += 1
        return False
    seen.add(key)
    output.append(row)
    return True

