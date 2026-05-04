import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import LANGUAGE_DETECTION_FIELDS
from src.language.ensemble import LANGUAGE_DETECTOR_VERSION, LanguageEnsembleDetector
from src.orchestration.run_collectors import deep_merge, simple_yaml_load
from src.storage.read_write import append_csv_rows, set_csv_field_size_limit


set_csv_field_size_limit()


DEFAULT_CONFIG = {
    "input_folder": "data/classified",
    "output_folder": "data/language_detected",
    "target_languages": LanguageEnsembleDetector.default_target_labels(),
    "thresholds": {
        "min_chars_for_detection": 15,
        "strong_confidence": 0.80,
        "fallback_confidence": 0.60,
        "min_rule_score": 1.0,
        "mixed_score_ratio": 0.35,
    },
    "detectors": {
        "fasttext_model_path": "models/lid.176.ftz",
    },
    "fallback": {
        "use_openai_when_uncertain": True,
        "use_openai_for_short_text": False,
        "model": "gpt-5.4-mini",
    },
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_timestamp(value):
    return value.replace(":", "-")


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_language_config(path):
    path = project_path(path)
    if not path.exists():
        return DEFAULT_CONFIG

    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        loaded = simple_yaml_load(text)
    else:
        loaded = yaml.safe_load(text) or {}
    return deep_merge(DEFAULT_CONFIG, loaded)


def latest_classified_file(input_folder):
    input_folder = Path(input_folder)
    candidates = sorted(
        input_folder.glob("ai_healthcare_classified_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def default_output_path(run_id, output_folder=None):
    folder = project_path(output_folder or DEFAULT_CONFIG["output_folder"])
    return folder / f"language_detected_{safe_timestamp(run_id)}.csv"


def iter_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def row_text(row):
    return (
        row.get("text")
        or row.get("preprocessed_text")
        or row.get("clean_text")
        or ""
    ).strip()


def stable_text_id(text):
    digest = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:24]
    return f"text_{digest}"


def row_id(row, text):
    return (
        row.get("text_item_id")
        or row.get("id")
        or row.get("source_item_id")
        or stable_text_id(text)
    )


def row_source_platform(row):
    return row.get("source_platform") or ""


def is_valid_ai_healthcare(row):
    return row.get("ai_healthcare_label") == "valid_ai_healthcare"


def count_valid_rows(path, limit=None):
    count = 0
    for row in iter_rows(path):
        if not is_valid_ai_healthcare(row):
            continue
        count += 1
        if limit and count >= limit:
            break
    return count


def read_existing_keys(path):
    path = Path(path)
    if not path.exists():
        return set()

    keys = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("id") or row.get("text")
            if key:
                keys.add(key)
    return keys


def language_output_row(row, detection):
    text = row_text(row)
    return {
        "id": row_id(row, text),
        "source_platform": row_source_platform(row),
        "text": text,
        "language_label": detection["label"],
        "detected_languages_json": json.dumps(
            detection.get("languages") or [], ensure_ascii=False
        ),
        "language_confidence": detection["confidence"],
        "language_detector_version": LANGUAGE_DETECTOR_VERSION,
        "detector_votes_json": json.dumps(
            detection["votes"], ensure_ascii=False, sort_keys=True
        ),
        "used_openai_fallback": detection["used_openai_fallback"],
        "language_detected_at_utc": utc_now_iso(),
    }


def run_language_detection(
    input_path,
    output_path,
    config,
    limit=None,
    batch_size=1000,
    show_progress=True,
):
    detector = LanguageEnsembleDetector(config)
    existing_keys = read_existing_keys(output_path)
    total_valid_rows = count_valid_rows(input_path, limit)

    output_rows = []
    input_row_count = 0
    valid_ai_healthcare_count = 0
    language_detected_count = 0
    skipped_non_valid_count = 0
    skipped_empty_count = 0
    skipped_existing_count = 0
    fallback_needed_count = 0
    openai_fallback_count = 0
    fallback_skipped_missing_key_count = 0
    detector_disagreement_count = 0
    label_counts = {}

    progress = tqdm(
        total=total_valid_rows,
        desc="language detection",
        unit="row",
        disable=not show_progress,
    )

    with progress:
        for row in iter_rows(input_path):
            if limit and valid_ai_healthcare_count >= limit:
                break
            input_row_count += 1
            if not is_valid_ai_healthcare(row):
                skipped_non_valid_count += 1
                continue

            valid_ai_healthcare_count += 1
            text = row_text(row)
            current_id = row_id(row, text)
            existing_key = current_id or text
            if not text:
                skipped_empty_count += 1
                progress.update(1)
                continue
            if existing_key in existing_keys or text in existing_keys:
                skipped_existing_count += 1
                progress.update(1)
                continue

            detection = detector.detect(text)
            fallback_needed_count += 1 if detection["fallback_needed"] else 0
            openai_fallback_count += 1 if detection["used_openai_fallback"] else 0
            fallback_skipped_missing_key_count += (
                1
                if detection.get("fallback_skipped_reason") == "missing_openai_api_key"
                else 0
            )
            local_source = detection["votes"].get("local", {}).get("source")
            detector_disagreement_count += 1 if local_source == "detector_disagreement" else 0

            output_row = language_output_row(row, detection)
            output_rows.append(output_row)
            existing_keys.add(output_row["id"])
            existing_keys.add(output_row["text"])
            language_detected_count += 1
            label = output_row["language_label"]
            label_counts[label] = label_counts.get(label, 0) + 1

            if len(output_rows) >= batch_size:
                append_csv_rows(output_path, output_rows, LANGUAGE_DETECTION_FIELDS)
                output_rows.clear()

            progress.update(1)
            if valid_ai_healthcare_count % 25 == 0:
                progress.set_postfix(
                    detected=language_detected_count,
                    fallback=openai_fallback_count,
                    refresh=False,
                )

        progress.set_postfix(
            detected=language_detected_count,
            fallback=openai_fallback_count,
            refresh=False,
        )

    append_csv_rows(output_path, output_rows, LANGUAGE_DETECTION_FIELDS)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_row_count": input_row_count,
        "valid_ai_healthcare_count": valid_ai_healthcare_count,
        "skipped_non_valid_ai_healthcare_count": skipped_non_valid_count,
        "language_detected_count": language_detected_count,
        "skipped_existing_count": skipped_existing_count,
        "skipped_empty_text_count": skipped_empty_count,
        "fallback_needed_count": fallback_needed_count,
        "openai_fallback_count": openai_fallback_count,
        "fallback_skipped_missing_key_count": fallback_skipped_missing_key_count,
        "detector_disagreement_count": detector_disagreement_count,
        "label_counts": label_counts,
        "target_languages": detector.target_labels,
        "language_detector_version": LANGUAGE_DETECTOR_VERSION,
    }


def save_summary(run_id, summary):
    summary_dir = PROJECT_ROOT / "logs" / "language_detection"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"language_detection_summary_{safe_timestamp(run_id)}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect language for classified rows labeled valid_ai_healthcare. "
            "Defaults to the newest data/classified/ai_healthcare_classified_*.csv."
        )
    )
    parser.add_argument("--input", help="Classified AI-healthcare CSV.")
    parser.add_argument("--output", help="Language detection output CSV.")
    parser.add_argument(
        "--config",
        default="configs/language_detection.yaml",
        help="Language detection config path.",
    )
    parser.add_argument("--limit", type=int, help="Only detect the first N valid rows.")
    parser.add_argument("--model", help="OpenAI fallback model override.")
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--use-openai-fallback",
        action="store_true",
        help="Use OpenAI for uncertain local language decisions.",
    )
    fallback_group.add_argument(
        "--no-openai-fallback",
        action="store_true",
        help="Disable OpenAI fallback and keep local uncertain labels.",
    )
    args = parser.parse_args()

    run_id = utc_now_iso()
    config = load_language_config(args.config)

    input_path = (
        project_path(args.input)
        if args.input
        else latest_classified_file(project_path(config["input_folder"]))
    )
    if input_path is None:
        raise FileNotFoundError("No ai_healthcare_classified_*.csv file found.")

    if args.model:
        config.setdefault("fallback", {})["model"] = args.model
    if args.use_openai_fallback:
        config.setdefault("fallback", {})["use_openai_when_uncertain"] = True
    if args.no_openai_fallback:
        config.setdefault("fallback", {})["use_openai_when_uncertain"] = False

    output_path = (
        project_path(args.output)
        if args.output
        else default_output_path(run_id, config.get("output_folder"))
    )

    summary = run_language_detection(
        input_path=input_path,
        output_path=output_path,
        config=config,
        limit=args.limit,
    )
    summary["run_id"] = run_id
    summary["used_input_default"] = args.input is None
    summary_path = save_summary(run_id, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Language detection summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
