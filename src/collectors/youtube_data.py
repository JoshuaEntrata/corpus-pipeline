import pandas as pd
import os
import csv
import hashlib
import json
import time
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ModuleNotFoundError:
    build = None

    class HttpError(Exception):
        pass

# Setup
load_dotenv()

# Get project root directory
project_root = Path(__file__).parent.parent.parent
print(f"Project root: {project_root}")

# Setup logging
log_dir = project_root / "logs" / "collectors"
log_dir.mkdir(parents=True, exist_ok=True)
error_log_path = log_dir / "youtube_scrape_errors.log"

logging.basicConfig(
    filename=str(error_log_path),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Setup output CSV path
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "youtube_scraped.csv"


def set_csv_field_size_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


set_csv_field_size_limit()

RAW_FIELDNAMES = [
    "run_id",
    "source_platform",
    "source_item_id",
    "source_url",
    "collection_method",
    "collection_query",
    "collected_at_utc",
    "created_at_utc",
    "author_id_hash",
    "title",
    "body_text",
    "description",
    "transcript",
    "comments_json",
    "engagement_json",
    "raw_json",
    "manual_file_name",
]

print(f"Output CSV: {output_csv}")
print(f"Error log: {error_log_path}")


# YouTube Client
class YouTubeClient:
    def __init__(self, api_key):
        if build is None:
            raise ImportError(
                "google-api-python-client is required for YouTube scraping. "
                "Install it with: pip install google-api-python-client"
            )
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def get_video(self, video_id):
        """Get video details by ID"""
        try:
            request = self.youtube.videos().list(part="snippet,statistics", id=video_id)
            response = request.execute()
            if response["items"]:
                return response["items"][0]
            return None
        except HttpError as e:
            raise Exception(f"YouTube API error: {e}")

    def search_videos(self, keyword, max_results=10):
        """Search videos by keyword"""
        try:
            request = self.youtube.search().list(
                q=keyword,
                part="snippet",
                type="video",
                maxResults=min(max_results, 50),
                order="relevance",
                publishedAfter="2020-01-01T00:00:00Z",
            )
            response = request.execute()
            return response.get("items", [])
        except HttpError as e:
            raise Exception(f"YouTube API error: {e}")

    def get_comments(self, video_id, max_results=100):
        """Get all comments and replies for a video"""
        try:
            comments_list = []
            request = self.youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                textFormat="plainText",
                maxResults=min(max_results, 100),
            )

            while request:
                response = request.execute()

                for item in response.get("items", []):
                    # Top-level comment
                    top_comment = item["snippet"]["topLevelComment"]
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_id = top_comment.get("id")
                    comments_list.append(
                        {
                            "source_item_id": comment_id,
                            "parent_item_id": video_id,
                            "conversation_root_id": video_id,
                            "source_url": f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}",
                            "author_id_hash": hash_identifier(
                                comment.get("authorChannelId", {}).get("value")
                                or comment.get("authorDisplayName")
                            ),
                            "body": comment.get("textDisplay", ""),
                            "created_at_utc": normalize_youtube_timestamp(
                                comment.get("publishedAt", "")
                            ),
                            "updated_at_utc": normalize_youtube_timestamp(
                                comment.get("updatedAt", "")
                            ),
                            "like_count": comment.get("likeCount"),
                            "is_reply": False,
                        }
                    )

                    # Replies to the comment
                    if item["snippet"]["totalReplyCount"] > 0:
                        for reply in item.get("replies", {}).get("comments", []):
                            reply_snippet = reply["snippet"]
                            reply_id = reply.get("id")
                            comments_list.append(
                                {
                                    "source_item_id": reply_id,
                                    "parent_item_id": reply_snippet.get(
                                        "parentId", comment_id
                                    ),
                                    "conversation_root_id": video_id,
                                    "source_url": f"https://www.youtube.com/watch?v={video_id}&lc={reply_id}",
                                    "author_id_hash": hash_identifier(
                                        reply_snippet.get("authorChannelId", {}).get(
                                            "value"
                                        )
                                        or reply_snippet.get("authorDisplayName")
                                    ),
                                    "body": reply_snippet.get("textDisplay", ""),
                                    "created_at_utc": normalize_youtube_timestamp(
                                        reply_snippet.get("publishedAt", "")
                                    ),
                                    "updated_at_utc": normalize_youtube_timestamp(
                                        reply_snippet.get("updatedAt", "")
                                    ),
                                    "like_count": reply_snippet.get("likeCount"),
                                    "is_reply": True,
                                }
                            )

                if len(comments_list) >= max_results:
                    return comments_list[:max_results]

                # Check if there are more pages
                if "nextPageToken" in response:
                    request = self.youtube.commentThreads().list(
                        part="snippet,replies",
                        videoId=video_id,
                        textFormat="plainText",
                        pageToken=response["nextPageToken"],
                        maxResults=min(max_results, 100),
                    )
                else:
                    break

            return comments_list
        except Exception as e:
            logging.error(f"Error getting comments for video {video_id}: {e}")
            return []


_client = None


def get_client():
    """Create the YouTube client only when a YouTube scraper actually runs."""
    global _client
    if _client is None:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable not set")
        _client = YouTubeClient(api_key)

    return _client


# Helper Functions
def clean_text(text):
    """Clean text by replacing newlines and extra spaces with single space."""
    if not text:
        return text
    # Replace newlines and tabs with spaces
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_youtube_timestamp(timestamp):
    """Normalize YouTube ISO timestamps to UTC ISO strings."""
    if not timestamp:
        return None

    try:
        timestamp_text = str(timestamp)
        if isinstance(timestamp, (int, float)) or timestamp_text.replace(
            ".", "", 1
        ).isdigit():
            return (
                datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        return None


def hash_identifier(value):
    """Hash author/channel identifiers instead of storing raw usernames."""
    if not value or value == "[deleted]":
        return None

    salt = os.getenv("AUTHOR_HASH_SALT", "")
    payload = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def migrate_legacy_record(row):
    """Map the first scraper CSV shape into the raw collection schema."""
    author = row.get("author")
    comments = []
    try:
        legacy_comments = json.loads(row.get("comments") or "[]")
    except json.JSONDecodeError:
        legacy_comments = []

    for comment in legacy_comments:
        comments.append(
            {
                "source_item_id": comment.get("id"),
                "parent_item_id": comment.get("parent_id"),
                "conversation_root_id": row.get("id") or None,
                "author_id_hash": hash_identifier(comment.get("author")),
                "body": clean_text(comment.get("body")),
                "created_at_utc": normalize_youtube_timestamp(
                    comment.get("created_utc")
                ),
                "is_reply": comment.get("is_reply"),
            }
        )

    legacy_raw = dict(row)
    legacy_raw.pop("author", None)
    legacy_raw.pop("comments", None)
    legacy_raw["author_id_hash"] = hash_identifier(author)
    legacy_raw["comments_json"] = comments

    return {
        "run_id": "legacy",
        "source_platform": "youtube",
        "source_item_id": row.get("id"),
        "source_url": row.get("url"),
        "collection_method": "legacy",
        "collection_query": None,
        "collected_at_utc": None,
        "created_at_utc": normalize_youtube_timestamp(row.get("created_utc")),
        "author_id_hash": hash_identifier(author),
        "title": clean_text(row.get("title")),
        "body_text": None,
        "description": clean_text(row.get("description")),
        "transcript": None,
        "comments_json": json_dumps(comments),
        "engagement_json": json_dumps({}),
        "raw_json": json_dumps(legacy_raw),
        "manual_file_name": None,
    }


def ensure_output_csv():
    """Create the raw CSV or migrate the legacy header before appending."""
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
            writer.writeheader()
        return

    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])

    if header == RAW_FIELDNAMES:
        return

    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    backup_path = output_csv.with_name(
        f"{output_csv.stem}_legacy_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{output_csv.suffix}"
    )
    output_csv.replace(backup_path)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(migrate_legacy_record(row))

    print(f"Migrated legacy YouTube CSV to schema. Backup: {backup_path}")


def video_exists(video_id):
    """Check if video ID already exists in the CSV."""
    if not output_csv.exists():
        return False

    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_id = row.get("source_item_id") or row.get("id")
                if existing_id == video_id:
                    return True
        return False
    except Exception as e:
        logging.error(f"Error checking if video exists: {e}")
        return False


def save_video(
    video_id, video_data, collection_method, collection_query, run_id, youtube_client
):
    """Save a single video to CSV."""
    try:
        ensure_output_csv()
        snippet = video_data["snippet"]
        created_at_utc = normalize_youtube_timestamp(snippet.get("publishedAt", ""))
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        engagement = video_data.get("statistics", {})

        # Get comments
        comments = youtube_client.get_comments(video_id, max_results=100)
        comments_json = json_dumps(comments)
        raw_item = {
            "id": video_id,
            "title": clean_text(snippet.get("title", "")),
            "description": clean_text(snippet.get("description", "")),
            "channel_id": snippet.get("channelId"),
            "published_at": created_at_utc,
            "url": video_url,
            "engagement": engagement,
        }

        record = {
            "run_id": run_id,
            "source_platform": "youtube",
            "source_item_id": video_id,
            "source_url": video_url,
            "collection_method": collection_method,
            "collection_query": collection_query,
            "collected_at_utc": utc_now_iso(),
            "created_at_utc": created_at_utc,
            "author_id_hash": hash_identifier(
                snippet.get("channelId") or snippet.get("channelTitle")
            ),
            "title": clean_text(snippet.get("title", "")),
            "body_text": None,
            "description": clean_text(snippet.get("description", "")),
            "transcript": None,
            "comments_json": comments_json,
            "engagement_json": json_dumps(engagement),
            "raw_json": json_dumps(raw_item),
            "manual_file_name": None,
        }

        # Append to CSV
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
            writer.writerow(record)

        print(f"Saved: {video_id} ({collection_method})")
        return True

    except Exception as e:
        error_msg = f"Error saving video {video_id}: {str(e)}"
        logging.error(error_msg)
        return False


# Scraping Functions
def scrape_by_video_id(video_ids, rate_limit_sec=1, run_id=None):
    """
    Scrape targeted videos by their IDs.

    Args:
        video_ids: list of YouTube video IDs
        rate_limit_sec: delay between requests in seconds
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING BY VIDEO ID")
    print(f"Total to process: {len(video_ids)}")
    print(f"{'='*60}")

    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    youtube_client = get_client()

    for video_id in tqdm(video_ids, desc="Processing videos", unit="video"):
        # Check if already exists
        if video_exists(video_id):
            skipped += 1
            time.sleep(rate_limit_sec)
            continue

        try:
            video_data = youtube_client.get_video(video_id)
            if video_data and save_video(
                video_id, video_data, "targeted_id", video_id, run_id, youtube_client
            ):
                collected += 1
            else:
                failed += 1

        except Exception as e:
            logging.error(f"Error fetching video {video_id}: {str(e)}")
            failed += 1

        time.sleep(rate_limit_sec)

    print(f"\n{'='*60}")
    print(f"RESULTS - Collected: {collected} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*60}\n")
    return collected, skipped, failed


def scrape_by_keywords(keywords_list, limit_per_query=10, rate_limit_sec=1, run_id=None):
    """
    Search for videos by keywords.

    Args:
        keywords_list: list of keywords to search
        limit_per_query: number of results per keyword search
        rate_limit_sec: delay between requests in seconds
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING BY KEYWORDS")
    print(f"Keywords: {', '.join(keywords_list)}")
    print(f"{'='*60}")

    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    youtube_client = get_client()

    for keyword in tqdm(keywords_list, desc="Processing keywords", unit="kw"):
        try:
            search_results = youtube_client.search_videos(
                keyword, max_results=limit_per_query
            )

            for result in tqdm(
                search_results, desc=f"  ↳ '{keyword}'", leave=False, unit="video"
            ):
                video_id = result["id"]["videoId"]

                if video_exists(video_id):
                    skipped += 1
                else:
                    if save_video(
                        video_id,
                        {"snippet": result["snippet"]},
                        "keyword_search",
                        keyword,
                        run_id,
                        youtube_client,
                    ):
                        collected += 1
                    else:
                        failed += 1

                time.sleep(rate_limit_sec)

        except Exception as e:
            logging.error(f"Error searching '{keyword}': {str(e)}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS - Collected: {collected} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*60}\n")
    return collected, skipped, failed


if __name__ == "__main__":
    # Load input data
    youtube_ids = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "youtube_post_ids.csv"
    )
    keywords = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "keywords.csv"
    )

    # Option 1: Scrape by video IDs
    # video_ids_list = youtube_ids["video_id"].tolist()
    # scrape_by_video_id(video_ids_list, rate_limit_sec=1)

    # Option 2: Scrape by keywords
    keywords_list = keywords["keyword"].tolist()
    scrape_by_keywords(keywords_list, limit_per_query=10, rate_limit_sec=1)
