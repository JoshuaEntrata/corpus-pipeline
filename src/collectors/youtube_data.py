import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ModuleNotFoundError:
    build = None

    class HttpError(Exception):
        pass


load_dotenv()

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.contracts import RAW_COLLECTION_FIELDS

log_dir = project_root / "logs" / "collectors"
log_dir.mkdir(parents=True, exist_ok=True)
error_log_path = log_dir / "youtube_scrape_errors.log"

logging.basicConfig(
    filename=str(error_log_path),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "youtube_scraped.csv"

RAW_FIELDNAMES = RAW_COLLECTION_FIELDS


def set_csv_field_size_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


set_csv_field_size_limit()


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
        try:
            request = self.youtube.videos().list(part="snippet,statistics", id=video_id)
            response = request.execute()
            if response["items"]:
                return response["items"][0]
            return None
        except HttpError as e:
            raise Exception(f"YouTube API error: {e}")

    def search_videos(self, keyword, max_results=50):
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
                    top_comment = item["snippet"]["topLevelComment"]
                    comment = top_comment["snippet"]
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

                    if item["snippet"]["totalReplyCount"] > 0:
                        for reply in item.get("replies", {}).get("comments", []):
                            reply_snippet = reply["snippet"]
                            comments_list.append(
                                {
                                    "source_item_id": reply.get("id"),
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

                if "nextPageToken" not in response:
                    break

                request = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    textFormat="plainText",
                    pageToken=response["nextPageToken"],
                    maxResults=min(max_results, 100),
                )

            return comments_list
        except Exception as e:
            logging.error(f"youtube comment scrape failed: {e}")
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


def clean_text(text):
    if not text:
        return text
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split())


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_youtube_timestamp(timestamp):
    if not timestamp:
        return None

    try:
        timestamp_text = str(timestamp)
        if (
            isinstance(timestamp, (int, float))
            or timestamp_text.replace(".", "", 1).isdigit()
        ):
            return (
                datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        return None


def log_scrape_start(platform, mode, total, unit):
    tqdm.write(f"{platform} | {mode} | total={total} {unit}")


def log_scrape_done(platform, mode, progress, total, collected, skipped, failed):
    tqdm.write(
        f"{platform} | {mode} | progress={progress}/{total} | "
        f"scraped={collected} | skipped={skipped} | failed={failed}"
    )


def update_progress(progress, collected, skipped, failed):
    progress.set_postfix(
        scraped=collected,
        skipped=skipped,
        failed=failed,
        refresh=False,
    )


def ensure_output_csv():
    """Create the raw CSV if it does not exist before appending."""
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

    raise ValueError(
        f"Unexpected YouTube raw CSV header in {output_csv}: {header}. "
        f"Expected {RAW_FIELDNAMES}."
    )


def video_exists(video_id):
    if not output_csv.exists():
        return False

    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("source_item_id") == video_id:
                    return True
        return False
    except Exception as e:
        logging.error(f"youtube duplicate check failed: {e}")
        return False


def save_video(
    video_id, video_data, collection_method, _collection_query, _run_id, youtube_client
):
    try:
        ensure_output_csv()
        snippet = video_data["snippet"]
        created_at_utc = normalize_youtube_timestamp(snippet.get("publishedAt", ""))
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        comments = youtube_client.get_comments(video_id, max_results=100)

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
            "comments_json": json_dumps(comments),
        }

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
            writer.writerow(record)

        return True
    except Exception as e:
        logging.error(f"youtube {collection_method} save failed: {str(e)}")
        return False


def scrape_by_video_id(video_ids, rate_limit_sec=1, run_id=None):
    """Scrape targeted videos by their IDs."""
    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    youtube_client = get_client()
    total = len(video_ids)
    log_scrape_start("youtube", "targeted_id", total, "videos")

    with tqdm(total=total, desc="youtube targeted_id", unit="video") as progress:
        for video_id in video_ids:
            if video_exists(video_id):
                skipped += 1
            else:
                try:
                    video_data = youtube_client.get_video(video_id)
                    if video_data and save_video(
                        video_id,
                        video_data,
                        "targeted_id",
                        video_id,
                        run_id,
                        youtube_client,
                    ):
                        collected += 1
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(f"youtube targeted_id fetch failed: {str(e)}")
                    failed += 1

            time.sleep(rate_limit_sec)
            progress.update(1)
            update_progress(progress, collected, skipped, failed)

    log_scrape_done("youtube", "targeted_id", total, total, collected, skipped, failed)
    return collected, skipped, failed


def scrape_by_keywords(
    keywords_list, limit_per_query=50, rate_limit_sec=1, run_id=None
):
    """Search for videos by keywords."""
    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    youtube_client = get_client()
    total_queries = len(keywords_list)
    log_scrape_start("youtube", "keyword", total_queries, "queries")

    with tqdm(total=total_queries, desc="youtube keyword", unit="query") as progress:
        for keyword in keywords_list:
            try:
                search_results = youtube_client.search_videos(
                    keyword, max_results=limit_per_query
                )

                for result in search_results:
                    video_id = result["id"]["videoId"]
                    if video_exists(video_id):
                        skipped += 1
                    elif save_video(
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
                logging.error(f"youtube keyword search failed: {str(e)}")
                failed += 1

            progress.update(1)
            update_progress(progress, collected, skipped, failed)

    log_scrape_done(
        "youtube", "keyword", total_queries, total_queries, collected, skipped, failed
    )
    return collected, skipped, failed


if __name__ == "__main__":
    keywords = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "keywords.csv"
    )

    scrape_by_keywords(
        keywords["keyword"].tolist(), limit_per_query=50, rate_limit_sec=1
    )
