from __future__ import annotations

import os
from typing import Any

from etl_pipeline.common.config import ConfigError
from etl_pipeline.common.io import json_dumps
from etl_pipeline.common.time import utc_now_iso
from etl_pipeline.extraction.base import BaseExtractor


class YouTubeExtractor(BaseExtractor):
    platform = "youtube"

    def __init__(self) -> None:
        if not os.getenv("YOUTUBE_API_KEY"):
            raise ConfigError("Missing YouTube API key in environment.")
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise ConfigError("Install the youtube extra to use YouTube extraction: pip install .[youtube]") from exc
        self.youtube = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])

    def search_videos_by_keyword(self, keyword: str, limit: int) -> list[dict]:
        response = (
            self.youtube.search()
            .list(part="snippet", q=keyword, type="video", maxResults=min(limit, 50))
            .execute()
        )
        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
        rows = self.extract_by_id(video_ids)
        for row in rows:
            row["collection_method"] = "keyword"
            row["metadata_json"] = json_dumps({**json_load(row.get("metadata_json", "{}")), "query": keyword}, {})
        return rows

    def get_video_by_id(self, video_id: str) -> dict:
        response = self.youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        items = response.get("items", [])
        if not items:
            raise LookupError(f"YouTube video not found: {video_id}")
        return self._video_to_row(items[0], "targeted_id", {})

    def extract_by_id(self, ids: list[str]) -> list[dict]:
        return [self.get_video_by_id(video_id) for video_id in ids]

    def extract_by_keyword(self, **kwargs: Any) -> list[dict]:
        return self.search_videos_by_keyword(str(kwargs["keyword"]), int(kwargs.get("limit", 100)))

    def extract_comments(self, source_id: str, limit: int | None = None) -> list[dict]:
        request = self.youtube.commentThreads().list(
            part="snippet,replies",
            videoId=source_id,
            maxResults=min(limit or 100, 100),
            textFormat="plainText",
        )
        response = request.execute()
        comments = []
        for item in response.get("items", [])[:limit]:
            top = item["snippet"]["topLevelComment"]
            top_snippet = top["snippet"]
            comments.append(
                {
                    "id": top["id"],
                    "text": top_snippet.get("textDisplay", ""),
                    "author": top_snippet.get("authorDisplayName", ""),
                    "created_at_utc": top_snippet.get("publishedAt", ""),
                    "url": "",
                    "replies": [
                        {
                            "id": reply["id"],
                            "text": reply["snippet"].get("textDisplay", ""),
                            "author": reply["snippet"].get("authorDisplayName", ""),
                            "created_at_utc": reply["snippet"].get("publishedAt", ""),
                            "url": "",
                        }
                        for reply in item.get("replies", {}).get("comments", [])
                    ],
                }
            )
        return comments

    def _video_to_row(self, item: dict, collection_method: str, metadata: dict) -> dict:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        metadata = {
            **metadata,
            "channel_id": snippet.get("channelId", ""),
            "view_count": stats.get("viewCount", ""),
            "like_count": stats.get("likeCount", ""),
        }
        return {
            "platform": self.platform,
            "collection_method": collection_method,
            "id": item["id"],
            "text": snippet.get("description", ""),
            "title": snippet.get("title", ""),
            "author": snippet.get("channelTitle", ""),
            "created_at_utc": snippet.get("publishedAt", ""),
            "url": f"https://www.youtube.com/watch?v={item['id']}",
            "comments_json": "[]",
            "metadata_json": json_dumps(metadata, {}),
            "extracted_at_utc": utc_now_iso(),
        }


def json_load(value: str) -> dict:
    import json

    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}

