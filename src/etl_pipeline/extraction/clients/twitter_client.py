from __future__ import annotations

import os
from typing import Any

import requests

from etl_pipeline.common.config import ConfigError
from etl_pipeline.common.io import json_dumps
from etl_pipeline.common.time import utc_now_iso
from etl_pipeline.extraction.base import BaseExtractor


class TwitterExtractor(BaseExtractor):
    platform = "twitter"
    base_url = "https://api.x.com/2"

    def __init__(self) -> None:
        if not os.getenv("TWITTER_BEARER_TOKEN"):
            raise ConfigError("Missing Twitter/X bearer token in environment.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {os.environ['TWITTER_BEARER_TOKEN']}"})

    def search_posts_by_keyword(self, keyword: str, limit: int) -> list[dict]:
        response = self.session.get(
            f"{self.base_url}/tweets/search/recent",
            params={
                "query": keyword,
                "max_results": min(max(limit, 10), 100),
                "tweet.fields": "created_at,author_id,conversation_id,public_metrics",
            },
            timeout=30,
        )
        response.raise_for_status()
        return [self._tweet_to_row(tweet, "keyword", {"query": keyword}) for tweet in response.json().get("data", [])]

    def get_post_by_id(self, post_id: str) -> dict:
        response = self.session.get(
            f"{self.base_url}/tweets/{post_id}",
            params={"tweet.fields": "created_at,author_id,conversation_id,public_metrics"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json().get("data")
        if not data:
            raise LookupError(f"Twitter/X post not found: {post_id}")
        return self._tweet_to_row(data, "targeted_id", {})

    def extract_by_id(self, ids: list[str]) -> list[dict]:
        return [self.get_post_by_id(post_id) for post_id in ids]

    def extract_by_keyword(self, **kwargs: Any) -> list[dict]:
        return self.search_posts_by_keyword(str(kwargs["keyword"]), int(kwargs.get("limit", 100)))

    def extract_comments(self, source_id: str, limit: int | None = None) -> list[dict]:
        query = f"conversation_id:{source_id}"
        response = self.session.get(
            f"{self.base_url}/tweets/search/recent",
            params={
                "query": query,
                "max_results": min(max(limit or 10, 10), 100),
                "tweet.fields": "created_at,author_id,conversation_id",
            },
            timeout=30,
        )
        response.raise_for_status()
        comments = []
        for tweet in response.json().get("data", [])[:limit]:
            if tweet.get("id") == source_id:
                continue
            comments.append(
                {
                    "id": tweet.get("id", ""),
                    "text": tweet.get("text", ""),
                    "author": tweet.get("author_id", ""),
                    "created_at_utc": tweet.get("created_at", ""),
                    "url": f"https://x.com/i/web/status/{tweet.get('id', '')}",
                    "replies": [],
                }
            )
        return comments

    def _tweet_to_row(self, tweet: dict, collection_method: str, metadata: dict) -> dict:
        metadata = {
            **metadata,
            "author_id": tweet.get("author_id", ""),
            "conversation_id": tweet.get("conversation_id", ""),
            "public_metrics": tweet.get("public_metrics", {}),
        }
        return {
            "platform": self.platform,
            "collection_method": collection_method,
            "id": tweet.get("id", ""),
            "text": tweet.get("text", ""),
            "title": "",
            "author": tweet.get("author_id", ""),
            "created_at_utc": tweet.get("created_at", ""),
            "url": f"https://x.com/i/web/status/{tweet.get('id', '')}",
            "comments_json": "[]",
            "metadata_json": json_dumps(metadata, {}),
            "extracted_at_utc": utc_now_iso(),
        }

