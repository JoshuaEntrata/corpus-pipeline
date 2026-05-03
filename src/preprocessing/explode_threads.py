import json


ROOT_TEXT_TYPES = {
    "reddit": {
        "title": "post_title",
        "body_text": "post_body",
        "description": "post_description",
        "transcript": "transcript_chunk",
    },
    "youtube": {
        "title": "video_title",
        "description": "video_description",
        "body_text": "video_body",
        "transcript": "transcript_chunk",
    },
    "manual_csv": {
        "title": "post_title",
        "body_text": "manual_text",
        "description": "manual_description",
        "transcript": "transcript_chunk",
    },
}


def parse_json(value, default):
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def compact_metadata(value):
    parsed = parse_json(value, {})
    if isinstance(parsed, dict):
        return parsed
    return {}


def text_present(value):
    return value not in (None, "") and str(value).strip() != ""


def root_text_items(record):
    source_platform = record.get("source_platform") or ""
    mapping = ROOT_TEXT_TYPES.get(source_platform, ROOT_TEXT_TYPES["manual_csv"])

    for column, text_type in mapping.items():
        raw_text = record.get(column)
        if text_present(raw_text):
            yield {
                "source_platform": source_platform,
                "source_item_id": record.get("source_item_id"),
                "conversation_root_id": record.get("source_item_id"),
                "parent_item_id": None,
                "text_type": text_type,
                "raw_text": raw_text,
                "source_url": record.get("source_url"),
                "created_at_utc": record.get("created_at_utc"),
                "author_id_hash": record.get("author_id_hash"),
                "metadata": {
                    "source_column": column,
                    "engagement": compact_metadata(record.get("engagement_json")),
                    "manual_file_name": record.get("manual_file_name"),
                },
            }


def comment_text_items(record):
    source_platform = record.get("source_platform") or ""
    comments = parse_json(record.get("comments_json"), [])
    if not isinstance(comments, list):
        return

    for comment in comments:
        if not isinstance(comment, dict) or not text_present(comment.get("body")):
            continue

        text_type = "reply" if comment.get("is_reply") else "comment"
        yield {
            "source_platform": source_platform,
            "source_item_id": comment.get("source_item_id") or comment.get("id"),
            "conversation_root_id": comment.get("conversation_root_id")
            or record.get("source_item_id"),
            "parent_item_id": comment.get("parent_item_id"),
            "text_type": text_type,
            "raw_text": comment.get("body"),
            "source_url": comment.get("source_url") or record.get("source_url"),
            "created_at_utc": comment.get("created_at_utc")
            or comment.get("created_utc"),
            "author_id_hash": comment.get("author_id_hash"),
            "metadata": {
                key: value
                for key, value in comment.items()
                if key
                not in {
                    "source_item_id",
                    "id",
                    "parent_item_id",
                    "conversation_root_id",
                    "source_url",
                    "author_id_hash",
                    "body",
                    "created_at_utc",
                    "created_utc",
                    "is_reply",
                }
            },
        }


def explode_raw_record(record):
    yield from root_text_items(record)
    yield from comment_text_items(record)

