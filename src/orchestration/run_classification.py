import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.ai_healthcare_classifier import (
    AIHealthcareClassifier,
    estimate_tokens,
)
from src.classification.prefilter import parse_keyword_config, prefilter_text
from src.classification.prompts import CLASSIFIER_VERSION
from src.contracts import CLASSIFICATION_FIELDS
from src.orchestration.run_collectors import deep_merge, simple_yaml_load
from src.storage.read_write import append_csv_rows, set_csv_field_size_limit


set_csv_field_size_limit()


DEFAULT_CONFIG = {
    "input_folder": "data/processed",
    "output_folder": "data/classified",
    "keywords_path": "configs/keywords.yaml",
    "classifier": {
        "version": CLASSIFIER_VERSION,
        "model": "gpt-4o-mini",
        "use_model": False,
        "classify_likely_relevant_with_model": True,
        "max_model_calls": 100,
        "input_usd_per_1m_tokens": 0.15,
        "output_usd_per_1m_tokens": 0.60,
    },
    "prefilter": {"min_chars": 15},
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_timestamp(value):
    return value.replace(":", "-")


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_classification_config(path):
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


def latest_processed_file(input_folder):
    input_folder = Path(input_folder)
    candidates = sorted(
        input_folder.glob("normalized_text_rows_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def default_output_path(run_id):
    return (
        PROJECT_ROOT
        / "data"
        / "classified"
        / f"ai_healthcare_classified_{safe_timestamp(run_id)}.csv"
    )


def iter_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def read_existing_texts(path):
    path = Path(path)
    if not path.exists():
        return set()

    with open(path, "r", newline="", encoding="utf-8") as f:
        return {
            row.get("text")
            for row in csv.DictReader(f)
            if row.get("text")
        }


def bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def float_arg(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_arg(value, default=0):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def usage_cost_usd(usage, input_usd_per_1m_tokens, output_usd_per_1m_tokens):
    input_tokens = int_arg(usage.get("input_tokens"))
    output_tokens = int_arg(usage.get("output_tokens"))
    return (
        (input_tokens / 1_000_000) * input_usd_per_1m_tokens
        + (output_tokens / 1_000_000) * output_usd_per_1m_tokens
    )


def row_text(row):
    return (
        row.get("preprocessed_text")
        or row.get("clean_text")
        or row.get("text")
        or ""
    )


def classify_row(text, prefilter, use_model, classifier=None):
    used_prefilter = True

    if use_model and prefilter["needs_model"]:
        result = classifier.classify(text)
        return (
            {
                "text": text,
                "ai_healthcare_label": result["label"],
                "ai_healthcare_confidence": result["confidence"],
                "classification_reason_short": result["reason_short"],
                "classified_at_utc": utc_now_iso(),
                "used_prefilter": used_prefilter,
                "prefilter_result": prefilter["prefilter_result"],
            },
            result.get("usage") or {},
        )

    return (
        {
            "text": text,
            "ai_healthcare_label": prefilter["label"],
            "ai_healthcare_confidence": prefilter["confidence"],
            "classification_reason_short": prefilter["reason_short"],
            "classified_at_utc": utc_now_iso(),
            "used_prefilter": used_prefilter,
            "prefilter_result": prefilter["prefilter_result"],
        },
        {},
    )


def run_classification(
    input_path,
    output_path,
    keywords_path,
    use_model=False,
    model=None,
    max_model_calls=100,
    min_chars=15,
    input_usd_per_1m_tokens=0.15,
    output_usd_per_1m_tokens=0.60,
    limit=None,
    batch_size=1000,
):
    ai_terms, health_terms = parse_keyword_config(keywords_path)
    existing_texts = read_existing_texts(output_path)
    classifier = AIHealthcareClassifier(model=model) if use_model else None

    output_rows = []
    input_row_count = 0
    classified_count = 0
    skipped_existing = 0
    skipped_empty = 0
    model_calls = 0
    model_input_tokens = 0
    model_output_tokens = 0
    model_total_tokens = 0
    total_usd_amount = 0.0
    estimated_prefilter_tokens = 0
    prefilter_counts = {}
    label_counts = {}

    for row in iter_rows(input_path):
        if limit and input_row_count >= limit:
            break
        input_row_count += 1

        text = row_text(row)
        estimated_prefilter_tokens += estimate_tokens(text)
        if not text:
            skipped_empty += 1
            continue
        if text in existing_texts:
            skipped_existing += 1
            continue

        prefilter = prefilter_text(
            text,
            is_empty=False,
            is_too_short=len(text) < min_chars,
            ai_terms=ai_terms,
            health_terms=health_terms,
        )
        should_use_model = (
            use_model
            and prefilter["needs_model"]
            and model_calls < max_model_calls
        )
        if should_use_model:
            model_calls += 1

        classified, usage = classify_row(text, prefilter, should_use_model, classifier)
        if should_use_model:
            model_input_tokens += int_arg(usage.get("input_tokens"))
            model_output_tokens += int_arg(usage.get("output_tokens"))
            model_total_tokens += int_arg(usage.get("total_tokens"))
            total_usd_amount += usage_cost_usd(
                usage,
                input_usd_per_1m_tokens,
                output_usd_per_1m_tokens,
            )
        output_rows.append(classified)
        existing_texts.add(text)
        classified_count += 1
        prefilter_counts[prefilter["prefilter_result"]] = (
            prefilter_counts.get(prefilter["prefilter_result"], 0) + 1
        )
        label = classified["ai_healthcare_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

        if len(output_rows) >= batch_size:
            append_csv_rows(output_path, output_rows, CLASSIFICATION_FIELDS)
            output_rows.clear()

    append_csv_rows(output_path, output_rows, CLASSIFICATION_FIELDS)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_row_count": input_row_count,
        "classified_count": classified_count,
        "skipped_existing_count": skipped_existing,
        "skipped_empty_text_count": skipped_empty,
        "model_call_count": model_calls,
        "use_model": use_model,
        "model_name": model if use_model else None,
        "model_input_token_count": model_input_tokens,
        "model_output_token_count": model_output_tokens,
        "model_total_token_count": model_total_tokens,
        "input_usd_per_1m_tokens": input_usd_per_1m_tokens,
        "output_usd_per_1m_tokens": output_usd_per_1m_tokens,
        "total_usd_amount": round(total_usd_amount, 8),
        "estimated_text_tokens_seen": estimated_prefilter_tokens,
        "prefilter_counts": prefilter_counts,
        "label_counts": label_counts,
    }


def save_summary(run_id, summary):
    summary_dir = PROJECT_ROOT / "logs" / "classification"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"classification_summary_{safe_timestamp(run_id)}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Classify normalized text rows for AI-in-healthcare relevance."
    )
    parser.add_argument("--input", help="Normalized text rows CSV.")
    parser.add_argument("--output", help="Classification output CSV.")
    parser.add_argument(
        "--config",
        default="configs/classification.yaml",
        help="Classification config path.",
    )
    parser.add_argument("--keywords", help="Keyword config path.")
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="Call OpenAI for likely relevant rows. Defaults to prefilter only.",
    )
    parser.add_argument("--model", help="OpenAI model override.")
    parser.add_argument(
        "--max-model-calls",
        type=int,
        help="Maximum model calls for this run.",
    )
    parser.add_argument(
        "--input-usd-per-1m-tokens",
        type=float,
        help="Input token price used for the run summary cost estimate.",
    )
    parser.add_argument(
        "--output-usd-per-1m-tokens",
        type=float,
        help="Output token price used for the run summary cost estimate.",
    )
    parser.add_argument("--limit", type=int, help="Only classify the first N rows.")
    args = parser.parse_args()

    run_id = utc_now_iso()
    config = load_classification_config(args.config)
    classifier_config = {
        **DEFAULT_CONFIG["classifier"],
        **config.get("classifier", {}),
    }

    input_path = (
        project_path(args.input)
        if args.input
        else latest_processed_file(project_path(config["input_folder"]))
    )
    if input_path is None:
        raise FileNotFoundError("No normalized_text_rows_*.csv file found.")

    output_path = project_path(args.output) if args.output else default_output_path(run_id)
    keywords_path = project_path(args.keywords or config["keywords_path"])
    use_model = args.use_model or bool_arg(classifier_config.get("use_model"))
    model = args.model or classifier_config.get("model")
    max_model_calls = (
        args.max_model_calls
        if args.max_model_calls is not None
        else int(classifier_config.get("max_model_calls", 100))
    )
    min_chars = int(config.get("prefilter", {}).get("min_chars", 15))
    input_usd_per_1m_tokens = (
        args.input_usd_per_1m_tokens
        if args.input_usd_per_1m_tokens is not None
        else float_arg(classifier_config.get("input_usd_per_1m_tokens"))
    )
    output_usd_per_1m_tokens = (
        args.output_usd_per_1m_tokens
        if args.output_usd_per_1m_tokens is not None
        else float_arg(classifier_config.get("output_usd_per_1m_tokens"))
    )

    summary = run_classification(
        input_path=input_path,
        output_path=output_path,
        keywords_path=keywords_path,
        use_model=use_model,
        model=model,
        max_model_calls=max_model_calls,
        min_chars=min_chars,
        input_usd_per_1m_tokens=input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=output_usd_per_1m_tokens,
        limit=args.limit,
    )
    summary["run_id"] = run_id
    summary_path = save_summary(run_id, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Classification summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
