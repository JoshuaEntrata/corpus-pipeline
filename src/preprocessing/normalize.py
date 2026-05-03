import csv
import hashlib
import re
import sys
from pathlib import Path
from tqdm import tqdm


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
    "openai",
    "chatgpt",
    "gpt",
    "gpt-4",
    "gpt-4o",
    "llm",
    "large language model",
    "generative ai",
    "anthropic",
    "claude",
    "gemini",
    "deepmind",
    "llama",
    "copilot",
    "openevidence",
    "open evidence",
    "med-palm",
    "medpalm",
    "medical ai",
    "clinical ai",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "nlp",
    "chatbot",
    "algorithm",
    "decision support",
    "clinical decision support",
    "automation",
    "automated",
    "ai scribe",
    "ambient scribe",
    "speech recognition",
    "computer vision",
    "predictive analytics",
    "predictive model",
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
    "radiology",
    "pharmacy",
    "surgery",
    "clinical",
    "health records",
    "electronic medical record",
    "electronic health record",
    "medical record",
    "emr",
    "ehr",
    "telehealth",
    "telemedicine",
    "triage",
    "emergency room",
    "er",
    "icu",
    "laboratory",
    "lab results",
    "blood test",
    "vital signs",
    "blood pressure",
    "glucose",
    "insulin",
    "vaccine",
    "vaccination",
    "infection",
    "sepsis",
    "cancer",
    "tumor",
    "oncology",
    "chemotherapy",
    "radiotherapy",
    "biopsy",
    "screening",
    "diabetes",
    "hypertension",
    "heart disease",
    "stroke",
    "asthma",
    "copd",
    "kidney disease",
    "alzheimer",
    "dementia",
    "depression",
    "anxiety",
    "pregnancy",
    "maternal health",
    "cardiology",
    "neurology",
    "pathology",
    "dermatology",
    "kalusugan",
    "pangkalusugan",
    "doktor",
    "duktor",
    "ospital",
    "pasyente",
    "sakit",
    "sintomas",
    "gamot",
    "lunas",
    "nars",
    "klinika",
    "konsultasyon",
    "bakuna",
    "impeksyon",
    "kanser",
    "diyabetis",
    "altapresyon",
    "alta presyon",
    "presyon",
    "pagbubuntis",
    "kahimsog",
    "panglawas",
    "tambal",
    "pagtambal",
    "hilanat",
    "salun-at",
    "pasiente",
    "agas",
    "panangagas",
    "balatian",
    "bulong",
    "pagbulong",
    "pagbusong",
]


def hash_value(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def stable_text_item_id(source_platform, source_item_id, text_type, text_hash):
    value = f"{source_platform}|{source_item_id}|{text_type}|{text_hash}"
    return hash_value(value)[:24]


def iter_raw_csv(path):
    path = Path(path)
    if not path.exists():
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def load_raw_csv(path):
    return list(iter_raw_csv(path))


def count_raw_csv_records(input_paths, source_filter=None):
    count = 0
    for input_path in input_paths:
        for record in iter_raw_csv(input_path):
            if source_filter and source_filter != "all":
                if record.get("source_platform") != source_filter:
                    continue
            count += 1
    return count


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
    needs_language_detection = (
        not is_empty_text and not is_too_short and not is_duplicate_text
    )
    needs_classification = needs_language_detection

    return {
        "source_platform": item.get("source_platform"),
        "id": stable_text_item_id(
            item.get("source_platform"),
            item.get("source_item_id"),
            item.get("text_type"),
            text_hash,
        ),
        "preprocessed_text": cleaned,
        "collection_method": item.get("collection_method")
        or raw_record.get("collection_method"),
        "has_ai_keyword": has_ai_keyword,
        "has_health_keyword": has_health_keyword,
        "needs_classification": needs_classification,
        "needs_language_detection": needs_language_detection,
        "_is_duplicate_text": is_duplicate_text,
        "_is_empty_text": is_empty_text,
        "_is_too_short": is_too_short,
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
    batch_size=1000,
    show_progress=True,
):
    ai_terms, health_terms = load_keyword_terms(keywords_path)
    seen_text_hashes = set()
    batch = []
    input_record_count = 0
    normalized_row_count = 0
    duplicate_text_count = 0
    empty_text_count = 0
    too_short_count = 0
    total_records = count_raw_csv_records(input_paths, source_filter)

    progress = tqdm(
        total=total_records,
        desc="preprocessing normalize",
        unit="record",
        disable=not show_progress,
    )

    with progress:
        for input_path in input_paths:
            for record in iter_raw_csv(input_path):
                if source_filter and source_filter != "all":
                    if record.get("source_platform") != source_filter:
                        continue
                input_record_count += 1

                for item in explode_raw_record(record):
                    row = normalize_item(
                        item,
                        record,
                        seen_text_hashes,
                        ai_terms,
                        health_terms,
                        min_chars,
                    )
                    normalized_row_count += 1
                    duplicate_text_count += 1 if row["_is_duplicate_text"] else 0
                    empty_text_count += 1 if row["_is_empty_text"] else 0
                    too_short_count += 1 if row["_is_too_short"] else 0
                    batch.append(row)

                    if len(batch) >= batch_size:
                        append_csv_rows(output_path, batch, NORMALIZED_TEXT_ROW_FIELDS)
                        batch.clear()

                progress.update(1)
                if input_record_count % 25 == 0:
                    progress.set_postfix(
                        rows=normalized_row_count,
                        duplicates=duplicate_text_count,
                        short=too_short_count,
                        refresh=False,
                    )

        progress.set_postfix(
            rows=normalized_row_count,
            duplicates=duplicate_text_count,
            short=too_short_count,
            refresh=False,
        )

    append_csv_rows(output_path, batch, NORMALIZED_TEXT_ROW_FIELDS)

    return {
        "input_record_count": input_record_count,
        "normalized_row_count": normalized_row_count,
        "duplicate_text_count": duplicate_text_count,
        "empty_text_count": empty_text_count,
        "too_short_count": too_short_count,
        "output_path": str(output_path),
    }
