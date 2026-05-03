import csv
import hashlib
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import NORMALIZED_TEXT_ROW_FIELDS
from src.preprocessing.clean_text import clean_text
from src.preprocessing.explode_threads import explode_raw_record
from src.storage.read_write import append_csv_rows, set_csv_field_size_limit


set_csv_field_size_limit()

DEFAULT_AI_TERMS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "chatgpt",
    "chatbot",
    "algorithm",
    "llm",
    "generative ai",
    "modelo",
    "teknolohiya",
    "awtomasyon",
]

DEFAULT_HEALTH_TERMS = [
    "health",
    "healthcare",
    "medical",
    "medicine",
    "doctor",
    "nurse",
    "hospital",
    "clinic",
    "patient",
    "diagnosis",
    "treatment",
    "symptoms",
    "public health",
    "mental health",
    "kalusugan",
    "doktor",
    "ospital",
    "pasyente",
    "sakit",
    "sintomas",
    "gamot",
    "nars",
    "klinika",
    "panglawas",
    "tambal",
]


def hash_value(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def stable_text_item_id(source_platform, source_item_id, text_type, text_hash):
    value = f"{source_platform}|{source_item_id}|{text_type}|{text_hash}"
    return hash_value(value)


def load_raw_csv(path):
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_keyword_terms(path=None):
    if not path:
        return DEFAULT_AI_TERMS, DEFAULT_HEALTH_TERMS

    path = Path(path)
    if not path.exists():
        return DEFAULT_AI_TERMS, DEFAULT_HEALTH_TERMS

    terms = {"ai_terms": [], "health_terms": []}
    current_group = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped in ("ai_terms:", "health_terms:"):
                current_group = stripped[:-1]
                continue
            if current_group and stripped.startswith("- "):
                terms[current_group].append(stripped[2:].strip().lower())

    ai_terms = terms["ai_terms"] or DEFAULT_AI_TERMS
    health_terms = terms["health_terms"] or DEFAULT_HEALTH_TERMS
    return ai_terms, health_terms


def contains_term(text, terms):
    text = text.lower()
    return any(
        re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text)
        for term in terms
    )


def normalize_item(item, raw_record, seen_text_hashes, ai_terms, health_terms, min_chars):
    raw_text = item.get("raw_text") or ""
    cleaned = clean_text(raw_text)
    text_hash = hash_value(cleaned.lower())
    is_duplicate_text = text_hash in seen_text_hashes
    seen_text_hashes.add(text_hash)

    is_empty_text = cleaned == ""
    is_too_short = len(cleaned) < min_chars
    has_ai_keyword = contains_term(cleaned, ai_terms) if cleaned else False
    has_health_keyword = contains_term(cleaned, health_terms) if cleaned else False
    needs_language_detection = not is_empty_text and not is_too_short
    needs_classification = needs_language_detection

    metadata = item.get("metadata") or {}
    metadata.update(
        {
            "raw_source_item_id": raw_record.get("source_item_id"),
            "raw_source_platform": raw_record.get("source_platform"),
        }
    )

    return {
        "text_item_id": stable_text_item_id(
            item.get("source_platform"),
            item.get("source_item_id"),
            item.get("text_type"),
            text_hash,
        ),
        "run_id": raw_record.get("run_id"),
        "source_platform": item.get("source_platform"),
        "source_item_id": item.get("source_item_id"),
        "conversation_root_id": item.get("conversation_root_id"),
        "parent_item_id": item.get("parent_item_id"),
        "text_type": item.get("text_type"),
        "raw_text": raw_text,
        "clean_text": cleaned,
        "source_url": item.get("source_url"),
        "created_at_utc": item.get("created_at_utc"),
        "collected_at_utc": raw_record.get("collected_at_utc"),
        "collection_method": raw_record.get("collection_method"),
        "collection_query": raw_record.get("collection_query"),
        "author_id_hash": item.get("author_id_hash"),
        "is_duplicate_text": is_duplicate_text,
        "is_empty_text": is_empty_text,
        "is_too_short": is_too_short,
        "has_ai_keyword": has_ai_keyword,
        "has_health_keyword": has_health_keyword,
        "needs_classification": needs_classification,
        "needs_language_detection": needs_language_detection,
        "text_hash": text_hash,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }


def normalize_raw_records(records, ai_terms=None, health_terms=None, min_chars=15):
    ai_terms = ai_terms or DEFAULT_AI_TERMS
    health_terms = health_terms or DEFAULT_HEALTH_TERMS
    normalized_rows = []
    seen_text_hashes = set()

    for raw_record in records:
        for item in explode_raw_record(raw_record):
            normalized_rows.append(
                normalize_item(
                    item,
                    raw_record,
                    seen_text_hashes,
                    ai_terms,
                    health_terms,
                    min_chars,
                )
            )

    return normalized_rows


def normalize_raw_files(
    input_paths,
    output_path,
    keywords_path=None,
    min_chars=15,
    source_filter=None,
):
    ai_terms, health_terms = load_keyword_terms(keywords_path)
    raw_records = []

    for input_path in input_paths:
        for record in load_raw_csv(input_path):
            if source_filter and source_filter != "all":
                if record.get("source_platform") != source_filter:
                    continue
            raw_records.append(record)

    rows = normalize_raw_records(
        raw_records,
        ai_terms=ai_terms,
        health_terms=health_terms,
        min_chars=min_chars,
    )
    append_csv_rows(output_path, rows, NORMALIZED_TEXT_ROW_FIELDS)
    return {
        "input_record_count": len(raw_records),
        "normalized_row_count": len(rows),
        "duplicate_text_count": sum(1 for row in rows if row["is_duplicate_text"]),
        "empty_text_count": sum(1 for row in rows if row["is_empty_text"]),
        "too_short_count": sum(1 for row in rows if row["is_too_short"]),
        "output_path": str(output_path),
    }
