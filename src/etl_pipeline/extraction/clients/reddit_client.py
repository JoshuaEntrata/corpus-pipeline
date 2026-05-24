from __future__ import annotations

import os
from typing import Any

from etl_pipeline.common.config import ConfigError
from etl_pipeline.common.io import json_dumps
from etl_pipeline.common.time import utc_now_iso
from etl_pipeline.extraction.base import BaseExtractor


class RedditExtractor(BaseExtractor):
    platform = "reddit"

    def __init__(self) -> None:
        if not all(os.getenv(key) for key in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")):
            raise ConfigError("Missing Reddit credentials in environment.")
        try:
            import praw
        except ImportError as exc:
            raise ConfigError("Install the reddit extra to use Reddit extraction: pip install .[reddit]") from exc
        self.reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )

    def search_post_by_id(self, post_id: str) -> dict:
        submission = self.reddit.submission(id=post_id)
        return self._submission_to_row(submission, "targeted_id", {})

    def search_posts_by_subreddit_keyword(
        self,
        subreddit: str,
        keyword: str,
        limit: int,
        collection_method: str = "subreddit_keyword",
    ) -> list[dict]:
        rows = []
        for submission in self.reddit.subreddit(subreddit).search(keyword, limit=limit):
            rows.append(self._submission_to_row(submission, collection_method, {"subreddit": subreddit, "query": keyword}))
        return rows

    def extract_by_id(self, ids: list[str]) -> list[dict]:
        return [self.search_post_by_id(post_id) for post_id in ids]

    def extract_by_keyword(self, **kwargs: Any) -> list[dict]:
        subreddit = str(kwargs.get("subreddit") or "all")
        collection_method = "subreddit_keyword" if kwargs.get("subreddit") else "keyword"
        return self.search_posts_by_subreddit_keyword(
            subreddit=subreddit,
            keyword=str(kwargs["keyword"]),
            limit=int(kwargs.get("limit", 100)),
            collection_method=collection_method,
        )

    def extract_comments(self, source_id: str, limit: int | None = None) -> list[dict]:
        submission = self.reddit.submission(id=source_id)
        submission.comments.replace_more(limit=0)
        comments = []
        for comment in submission.comments.list()[:limit]:
            comments.append(
                {
                    "id": comment.id,
                    "text": getattr(comment, "body", ""),
                    "author": str(getattr(comment, "author", "") or ""),
                    "created_at_utc": _epoch_to_iso(getattr(comment, "created_utc", None)),
                    "url": f"https://www.reddit.com{getattr(comment, 'permalink', '')}",
                    "replies": [],
                }
            )
        return comments

    def _submission_to_row(self, submission: Any, collection_method: str, metadata: dict) -> dict:
        metadata = {
            **metadata,
            "subreddit": str(getattr(submission, "subreddit", "") or metadata.get("subreddit", "")),
            "score": getattr(submission, "score", ""),
            "upvote_ratio": getattr(submission, "upvote_ratio", ""),
        }
        return {
            "platform": self.platform,
            "collection_method": collection_method,
            "id": submission.id,
            "text": getattr(submission, "selftext", "") or "",
            "title": getattr(submission, "title", "") or "",
            "author": str(getattr(submission, "author", "") or ""),
            "created_at_utc": _epoch_to_iso(getattr(submission, "created_utc", None)),
            "url": f"https://www.reddit.com{getattr(submission, 'permalink', '')}",
            "comments_json": "[]",
            "metadata_json": json_dumps(metadata, {}),
            "extracted_at_utc": utc_now_iso(),
        }


def _epoch_to_iso(value: float | None) -> str:
    if value is None:
        return ""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(float(value), UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
