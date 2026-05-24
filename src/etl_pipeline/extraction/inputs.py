from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from etl_pipeline.common.io import read_csv

DEFAULT_LIMIT = 100
DEFAULT_COMMENT_LIMIT = 500
DEFAULT_INCLUDE_COMMENTS = True
KEYWORD_PLATFORMS = ("reddit", "youtube", "twitter")


def load_input_jobs(inputs_dir: str | Path = "inputs") -> dict[str, dict[str, Any]]:
    base = Path(inputs_dir)
    jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keywords = _read_values(base / "keywords.csv", "keyword")
    subreddits = _read_values(base / "subreddits.csv", "subreddit")

    _load_id_jobs(base / "reddit_ids.csv", "reddit", jobs)
    _load_id_jobs(base / "youtube_ids.csv", "youtube", jobs)
    _load_id_jobs(base / "twitter_ids.csv", "twitter", jobs)
    _load_subreddit_jobs(subreddits, keywords, jobs)
    _load_keyword_jobs(keywords, jobs)

    return {
        platform: {"enabled": True, "jobs": platform_jobs}
        for platform, platform_jobs in sorted(jobs.items())
        if platform_jobs
    }


def _load_id_jobs(path: Path, platform: str, jobs: dict[str, list[dict[str, Any]]]) -> None:
    ids = _read_values(path, "id")
    if not ids:
        ids = _read_values(path, "ids")
    if ids:
        jobs[platform].append(
            {
                "collection_method": "targeted_id",
                "ids": ids,
                "include_comments": DEFAULT_INCLUDE_COMMENTS,
                "comment_limit": DEFAULT_COMMENT_LIMIT,
            }
        )


def _load_subreddit_jobs(subreddits: list[str], keywords: list[str], jobs: dict[str, list[dict[str, Any]]]) -> None:
    for subreddit in subreddits:
        for keyword in keywords:
            jobs["reddit"].append(
                {
                    "collection_method": "subreddit_keyword",
                    "subreddit": subreddit,
                    "keyword": keyword,
                    "limit": DEFAULT_LIMIT,
                    "include_comments": DEFAULT_INCLUDE_COMMENTS,
                    "comment_limit": DEFAULT_COMMENT_LIMIT,
                }
            )


def _load_keyword_jobs(keywords: list[str], jobs: dict[str, list[dict[str, Any]]]) -> None:
    for keyword in keywords:
        for platform in KEYWORD_PLATFORMS:
            jobs[platform].append(
                {
                    "collection_method": "keyword",
                    "keyword": keyword,
                    "limit": DEFAULT_LIMIT,
                    "include_comments": DEFAULT_INCLUDE_COMMENTS,
                    "comment_limit": DEFAULT_COMMENT_LIMIT,
                }
            )


def _read_values(path: Path, field: str) -> list[str]:
    values = []
    for row in read_csv(path):
        value = str(row.get(field, "")).strip()
        if value:
            values.append(value)
    return values

