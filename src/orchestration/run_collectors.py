import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.ids import IdRegistry
from src.storage.read_write import set_csv_field_size_limit


set_csv_field_size_limit()


DEFAULT_CONFIG = {
    "run": {
        "save_raw": True,
        "skip_existing_ids": True,
        "log_level": "INFO",
        "rate_limit_sec": 2,
        "limit_per_query": 10,
        "registry_path": "data/registry/collected_ids.csv",
    },
    "collectors": {
        "reddit": {
            "enabled": True,
            "modes": ["targeted", "subreddit_keyword", "keyword"],
            "inputs": {
                "post_ids": "src/collectors/inputs/reddit_post_ids.csv",
                "post_id_column": "post_id",
                "keywords": "src/collectors/inputs/keywords.csv",
                "keyword_column": "keyword",
                "subreddits": "src/collectors/inputs/subreddits.csv",
                "subreddit_column": "subreddit",
            },
            "output_path": "data/raw/reddit_scraped.csv",
        },
        "youtube": {
            "enabled": True,
            "modes": ["targeted", "keyword"],
            "inputs": {
                "video_ids": "src/collectors/inputs/youtube_post_ids.csv",
                "video_id_column": "video_id",
                "keywords": "src/collectors/inputs/keywords.csv",
                "keyword_column": "keyword",
            },
            "output_path": "data/raw/youtube_scraped.csv",
        },
        "manual_csv": {
            "enabled": True,
            "modes": ["manual_csv"],
            "input_folder": "data/manual_uploads",
            "output_path": "data/raw/manual_csv_scraped.csv",
        },
        "ensembledata": {"enabled": False},
    },
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_scalar(value):
    value = value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none", "~"):
        return None
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def simple_yaml_load(text):
    """Small YAML subset parser for this repo's config files."""
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_map(index, indent):
        data = {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent or content.startswith("- "):
                break

            key, separator, value = content.partition(":")
            if not separator:
                index += 1
                continue

            key = key.strip()
            value = value.strip()
            if value:
                data[key] = parse_scalar(value)
                index += 1
                continue

            if index + 1 >= len(lines) or lines[index + 1][0] <= current_indent:
                data[key] = None
                index += 1
                continue

            next_indent, next_content = lines[index + 1]
            if next_content.startswith("- "):
                data[key], index = parse_list(index + 1, next_indent)
            else:
                data[key], index = parse_map(index + 1, next_indent)

        return data, index

    def parse_list(index, indent):
        values = []
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent or not content.startswith("- "):
                break
            values.append(parse_scalar(content[2:].strip()))
            index += 1
        return values, index

    parsed, _ = parse_map(0, lines[0][0] if lines else 0)
    return parsed


def load_config(config_path):
    path = PROJECT_ROOT / config_path
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)

    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        loaded = simple_yaml_load(text)
        return deep_merge(DEFAULT_CONFIG, loaded)

    loaded = yaml.safe_load(text) or {}
    return deep_merge(DEFAULT_CONFIG, loaded)


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_column(path, column):
    path = project_path(path)
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [
            row[column].strip()
            for row in reader
            if column in row and row[column] not in (None, "")
        ]


def selected_modes(source_config, requested_mode):
    modes = source_config.get("modes", [])
    if requested_mode == "all":
        return modes
    return [requested_mode] if requested_mode in modes else []


def add_result(results, source, mode, collected, skipped, failed):
    results.append(
        {
            "source": source,
            "mode": mode,
            "collected": collected,
            "skipped": skipped,
            "failed": failed,
        }
    )


def refresh_registry(registry, output_path, source_platform):
    added = registry.refresh_from_raw_csv(
        project_path(output_path), default_platform=source_platform
    )
    registry.save()
    return added


def run_reddit(config, args, registry, results, run_id):
    source_config = config["collectors"]["reddit"]
    modes = selected_modes(source_config, args.mode)
    if not modes:
        return

    from src.collectors import reddit_praw

    inputs = source_config["inputs"]
    rate_limit = args.rate_limit or config["run"]["rate_limit_sec"]
    limit = args.limit or config["run"]["limit_per_query"]

    if "targeted" in modes:
        post_ids = read_csv_column(inputs["post_ids"], inputs["post_id_column"])
        collected, skipped, failed = reddit_praw.scrape_by_submission_id(
            post_ids, rate_limit_sec=rate_limit, run_id=run_id
        )
        add_result(results, "reddit", "targeted", collected, skipped, failed)

    if "subreddit_keyword" in modes:
        subreddits = read_csv_column(inputs["subreddits"], inputs["subreddit_column"])
        keywords = read_csv_column(inputs["keywords"], inputs["keyword_column"])
        collected, skipped, failed = reddit_praw.scrape_by_subreddits_keywords(
            subreddits,
            keywords,
            limit_per_query=limit,
            rate_limit_sec=rate_limit,
            run_id=run_id,
        )
        add_result(
            results, "reddit", "subreddit_keyword", collected, skipped, failed
        )

    if "keyword" in modes:
        keywords = read_csv_column(inputs["keywords"], inputs["keyword_column"])
        collected, skipped, failed = reddit_praw.scrape_by_keywords(
            keywords, limit_per_query=limit, rate_limit_sec=rate_limit, run_id=run_id
        )
        add_result(results, "reddit", "keyword", collected, skipped, failed)

    refresh_registry(registry, source_config["output_path"], "reddit")


def run_youtube(config, args, registry, results, run_id):
    source_config = config["collectors"]["youtube"]
    modes = selected_modes(source_config, args.mode)
    if not modes:
        return

    from src.collectors import youtube_data

    inputs = source_config["inputs"]
    rate_limit = args.rate_limit or config["run"]["rate_limit_sec"]
    limit = args.limit or config["run"]["limit_per_query"]

    if "targeted" in modes:
        video_ids = read_csv_column(inputs["video_ids"], inputs["video_id_column"])
        collected, skipped, failed = youtube_data.scrape_by_video_id(
            video_ids, rate_limit_sec=rate_limit, run_id=run_id
        )
        add_result(results, "youtube", "targeted", collected, skipped, failed)

    if "keyword" in modes:
        keywords = read_csv_column(inputs["keywords"], inputs["keyword_column"])
        collected, skipped, failed = youtube_data.scrape_by_keywords(
            keywords, limit_per_query=limit, rate_limit_sec=rate_limit, run_id=run_id
        )
        add_result(results, "youtube", "keyword", collected, skipped, failed)

    refresh_registry(registry, source_config["output_path"], "youtube")


def run_manual_csv(config, args, registry, results, run_id):
    source_config = config["collectors"]["manual_csv"]
    modes = selected_modes(source_config, args.mode)
    if not modes:
        return

    from src.collectors.manual_csv import collect_from_folder

    collected, skipped, failed = collect_from_folder(
        input_folder=project_path(source_config["input_folder"]),
        output_csv=project_path(source_config["output_path"]),
        registry_path=project_path(config["run"]["registry_path"]),
        run_id=run_id,
    )
    add_result(results, "manual_csv", "manual_csv", collected, skipped, failed)
    refresh_registry(registry, source_config["output_path"], "manual_csv")


def save_summary(run_id, results, registry_path):
    summary_dir = PROJECT_ROOT / "logs" / "collectors"
    summary_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = run_id.replace(":", "-")
    summary_path = summary_dir / f"run_summary_{safe_run_id}.json"
    summary = {
        "run_id": run_id,
        "sources_run": sorted({result["source"] for result in results}),
        "collected_count": sum(result["collected"] for result in results),
        "skipped_existing_id_count": sum(result["skipped"] for result in results),
        "failed_count": sum(result["failed"] for result in results),
        "results": results,
        "registry_path": str(registry_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary_path, summary


def main():
    parser = argparse.ArgumentParser(description="Run configured data collectors.")
    parser.add_argument(
        "--source",
        choices=["all", "reddit", "youtube", "manual_csv"],
        default="all",
        help="Collector source to run. Defaults to all enabled collectors.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "targeted", "keyword", "subreddit_keyword", "manual_csv"],
        default="all",
        help="Collection mode to run. Defaults to every configured mode.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Shortcut for --source all --mode all.",
    )
    parser.add_argument(
        "--config",
        default="configs/collectors.yaml",
        help="Collector config path.",
    )
    parser.add_argument("--limit", type=int, help="Override results per query.")
    parser.add_argument("--rate-limit", type=int, help="Override request delay.")
    args = parser.parse_args()

    if args.all:
        args.source = "all"
        args.mode = "all"

    config = load_config(args.config)
    registry = IdRegistry(project_path(config["run"]["registry_path"]))
    run_id = utc_now_iso()
    results = []

    selected_sources = (
        ["reddit", "youtube", "manual_csv"]
        if args.source == "all"
        else [args.source]
    )

    for source in selected_sources:
        source_config = config["collectors"].get(source, {})
        if not source_config.get("enabled", False):
            continue

        if source == "reddit":
            run_reddit(config, args, registry, results, run_id)
        elif source == "youtube":
            run_youtube(config, args, registry, results, run_id)
        elif source == "manual_csv":
            run_manual_csv(config, args, registry, results, run_id)

    summary_path, summary = save_summary(
        run_id, results, project_path(config["run"]["registry_path"])
    )
    print(json.dumps(summary, indent=2))
    print(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
