import pandas as pd
import os
import csv
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
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")

from src.contracts import RAW_COLLECTION_FIELDS

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

RAW_FIELDNAMES = RAW_COLLECTION_FIELDS

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
                            "body": comment.get("textDisplay", ""),
                            "created_at_utc": normalize_youtube_timestamp(
                                comment.get("publishedAt", "")
                            ),
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
                                    "body": reply_snippet.get("textDisplay", ""),
                                    "created_at_utc": normalize_youtube_timestamp(
                                        reply_snippet.get("publishedAt", "")
                                    ),
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


def normalize_comment_record(comment, root_id):
    parent_item_id = comment.get("parent_item_id") or comment.get("parent_id")
    created_at_utc = comment.get("created_at_utc") or normalize_youtube_timestamp(
        comment.get("created_utc")
    )
    is_reply = comment.get("is_reply")
    if is_reply is None:
        is_reply = parent_item_id not in (None, root_id)

    return {
        "source_item_id": comment.get("source_item_id") or comment.get("id"),
        "parent_item_id": parent_item_id,
        "body": clean_text(comment.get("body")),
        "created_at_utc": created_at_utc,
        "is_reply": is_reply,
    }


def load_comment_records(row):
    comments_value = row.get("comments_json") or row.get("comments") or "[]"
    try:
        comments = json.loads(comments_value)
    except (TypeError, json.JSONDecodeError):
        comments = []

    root_id = row.get("source_item_id") or row.get("id")
    return [
        normalize_comment_record(comment, root_id)
        for comment in comments
        if isinstance(comment, dict)
    ]


def migrate_legacy_record(row):
    """Map older YouTube scraper CSV shapes into the compact raw schema."""
    source_item_id = row.get("source_item_id") or row.get("id")
    comments = load_comment_records(row)

    return {
        "source_platform": row.get("source_platform") or "youtube",
        "source_item_id": source_item_id,
        "source_url": row.get("source_url") or row.get("url"),
        "collection_method": row.get("collection_method") or "legacy",
        "created_at_utc": row.get("created_at_utc")
        or normalize_youtube_timestamp(row.get("created_utc")),
        "title": clean_text(row.get("title")),
        "body_text": clean_text(row.get("body_text")),
        "description": clean_text(row.get("description")),
        "transcript": clean_text(row.get("transcript")),
        "comments_json": json_dumps(comments),
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
    video_id, video_data, collection_method, _collection_query, _run_id, youtube_client
):
    """Save a single video to CSV."""
    try:
        ensure_output_csv()
        snippet = video_data["snippet"]
        created_at_utc = normalize_youtube_timestamp(snippet.get("publishedAt", ""))
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Get comments
        comments = youtube_client.get_comments(video_id, max_results=100)
        comments_json = json_dumps(comments)

        record = {
            "source_platform": "youtube",
            "source_item_id": video_id,
            "source_url": video_url,
            "collection_method": collection_method,
            "created_at_utc": created_at_utc,
            "title": clean_text(snippet.get("title", "")),
            "body_text": None,
            "description": clean_text(snippet.get("description", "")),
            "transcript": None,
            "comments_json": comments_json,
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
