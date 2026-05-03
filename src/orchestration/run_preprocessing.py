import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.run_collectors import load_config, project_path
from src.preprocessing.normalize import normalize_raw_files


DEFAULT_RAW_INPUTS = [
    "data/raw/reddit_scraped.csv",
    "data/raw/youtube_scraped.csv",
    "data/raw/manual_csv_scraped.csv",
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_timestamp(value):
    return value.replace(":", "-")


def default_output_path(run_id):
    return PROJECT_ROOT / "data" / "processed" / f"normalized_text_rows_{safe_timestamp(run_id)}.csv"


def configured_raw_inputs(config):
    paths = []
    for source_config in config.get("collectors", {}).values():
        output_path = source_config.get("output_path")
        if output_path:
            paths.append(output_path)
    return paths or DEFAULT_RAW_INPUTS


def existing_paths(paths):
    return [path for path in paths if Path(path).exists()]


def save_summary(run_id, summary):
    summary_dir = PROJECT_ROOT / "logs" / "preprocessing"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"preprocessing_summary_{safe_timestamp(run_id)}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Normalize raw collector CSVs into standalone text rows."
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Raw CSV input path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--source",
        choices=["all", "reddit", "youtube", "manual_csv"],
        default="all",
        help="Only normalize rows from this source platform.",
    )
    parser.add_argument("--output", help="Output CSV path.")
    parser.add_argument(
        "--config",
        default="configs/collectors.yaml",
        help="Collector config used to discover raw output paths.",
    )
    parser.add_argument(
        "--keywords",
        default="configs/keywords.yaml",
        help="Keyword config for AI/health flags.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=15,
        help="Minimum cleaned text length before a row is considered classifiable.",
    )
    args = parser.parse_args()

    run_id = utc_now_iso()
    config = load_config(args.config)
    input_paths = args.input or configured_raw_inputs(config)
    input_paths = [project_path(path) for path in input_paths]
    input_paths = existing_paths(input_paths)
    output_path = project_path(args.output) if args.output else default_output_path(run_id)

    summary = normalize_raw_files(
        input_paths=input_paths,
        output_path=output_path,
        keywords_path=project_path(args.keywords),
        min_chars=args.min_chars,
        source_filter=args.source,
    )
    summary["run_id"] = run_id
    summary["source_filter"] = args.source
    summary["input_paths"] = [str(path) for path in input_paths]
    summary_path = save_summary(run_id, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preprocessing summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

