from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from etl_pipeline.common.config import ConfigError, should_skip_seen
from etl_pipeline.common.ids import extraction_key
from etl_pipeline.common.io import append_csv, json_dumps, write_json
from etl_pipeline.common.logging import log_stage_error
from etl_pipeline.common.paths import logs_dir, master_dir, stage_run_dir, state_dir
from etl_pipeline.common.state import append_state_rows, extraction_master_keys, read_state_keys
from etl_pipeline.common.summaries import collection_method_counts, platform_counts
from etl_pipeline.extraction.clients.reddit_client import RedditExtractor
from etl_pipeline.extraction.clients.twitter_client import TwitterExtractor
from etl_pipeline.extraction.clients.youtube_client import YouTubeExtractor
from etl_pipeline.extraction.inputs import load_input_jobs
from etl_pipeline.extraction.schemas import EXTRACTION_FIELDS, normalize_extraction_row
from tqdm import tqdm

EXTRACTOR_REGISTRY = {
    "reddit": RedditExtractor,
    "youtube": YouTubeExtractor,
    "twitter": TwitterExtractor,
}


def run_extraction(
    pipeline_config: dict[str, Any],
    *,
    inputs_dir: str | Path = "inputs",
    run_id: str,
    force: bool = False,
) -> dict[str, Any]:
    sources = load_input_jobs(inputs_dir)
    enabled_platforms = pipeline_config.get("stages", {}).get("extraction", {}).get("platforms", {})
    sources = {
        platform: platform_config
        for platform, platform_config in sources.items()
        if enabled_platforms.get(platform, True)
    }
    run_dir = stage_run_dir(pipeline_config, "extraction", run_id)
    master_path = master_dir(pipeline_config) / "extraction_raw.csv"
    state_path = state_dir(pipeline_config) / "extraction_seen_ids.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    if should_skip_seen(pipeline_config, "extraction", force):
        seen |= extraction_master_keys(master_path)
        seen |= read_state_keys(state_path)

    rows_by_platform: dict[str, list[dict]] = defaultdict(list)
    skipped_existing = 0
    duplicate_rows = 0
    errors = 0
    configured_jobs = sum(len(platform_config.get("jobs", [])) for platform_config in sources.values())

    with tqdm(total=configured_jobs, desc="extraction", unit="job") as pbar:
        for platform, platform_config in sources.items():
            if not platform_config.get("enabled", False):
                continue
            extractor_class = EXTRACTOR_REGISTRY.get(platform)
            if not extractor_class:
                raise ConfigError(f"No extractor registered for platform: {platform}")
            extractor = extractor_class()
            for job in platform_config.get("jobs", []):
                method = job.get("collection_method", "")
                try:
                    rows = _run_job(extractor, job)
                except Exception as exc:
                    errors += 1
                    log_stage_error(
                        logs_dir(pipeline_config),
                        stage="extraction",
                        platform=platform,
                        collection_method=method,
                        id_or_query=str(job.get("keyword") or ",".join(job.get("ids", []))),
                        error=exc,
                    )
                    pbar.update(1)
                    continue

                for row in rows:
                    normalized = normalize_extraction_row(row)
                    key = extraction_key(normalized["platform"], normalized["id"])
                    if key in seen:
                        skipped_existing += 1
                        duplicate_rows += 1
                        continue
                    if job.get("include_comments", False):
                        try:
                            comments = extractor.extract_comments(normalized["id"], job.get("comment_limit"))
                            normalized["comments_json"] = json_dumps(comments, [])
                        except Exception as exc:
                            errors += 1
                            normalized["comments_json"] = "[]"
                            log_stage_error(
                                logs_dir(pipeline_config),
                                stage="extraction",
                                platform=platform,
                                collection_method=method,
                                id_or_query=normalized["id"],
                                error=exc,
                            )
                    seen.add(key)
                    rows_by_platform[platform].append(normalized)
                    _persist_extraction_row(normalized, run_dir, master_path, state_path)
                pbar.update(1)

    all_rows = [row for platform_rows in rows_by_platform.values() for row in platform_rows]

    summary = {
        "stage": "extraction",
        "run_id": run_id,
        "inputs_dir": str(inputs_dir),
        "configured_jobs": configured_jobs,
        "output_rows": len(all_rows),
        "new_rows_added": len(all_rows),
        "skipped_existing_rows": skipped_existing,
        "duplicate_rows_skipped": duplicate_rows,
        "errors": errors,
        "num_platforms": len({row["platform"] for row in all_rows}),
        "platform_distribution": platform_counts(all_rows),
        "collection_method_distribution": collection_method_counts(all_rows),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def _persist_extraction_row(row: dict[str, Any], run_dir: Path, master_path: Path, state_path: Path) -> None:
    append_csv(master_path, [row], EXTRACTION_FIELDS)
    append_csv(run_dir / f"{row.get('platform', 'unknown')}.csv", [row], EXTRACTION_FIELDS)
    append_state_rows(state_path, "extraction", [row], lambda item: extraction_key(item["platform"], item["id"]))


def _run_job(extractor: Any, job: dict[str, Any]) -> list[dict]:
    method = job.get("collection_method")
    if method == "targeted_id":
        return extractor.extract_by_id([str(item) for item in job.get("ids", [])])
    if method == "subreddit_keyword":
        return extractor.extract_by_keyword(
            subreddit=job["subreddit"],
            keyword=job["keyword"],
            limit=job.get("limit", 100),
        )
    if method == "keyword":
        return extractor.extract_by_keyword(keyword=job["keyword"], limit=job.get("limit", 100))
    raise ConfigError(f"Unsupported collection method: {method}")
