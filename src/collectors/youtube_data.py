import pandas as pd
import os
import csv
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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

# Create CSV headers if file doesn't exist
if not output_csv.exists():
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "title",
                "description",
                "author",
                "created_utc",
                "comments",
                "url",
            ],
        )
        writer.writeheader()

print(f"Output CSV: {output_csv}")
print(f"Error log: {error_log_path}")


# YouTube Client
class YouTubeClient:
    def __init__(self, api_key):
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
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments_list.append(
                        {
                            "author": comment.get("authorDisplayName", "[deleted]"),
                            "body": comment.get("textDisplay", ""),
                            "created_utc": comment.get("publishedAt", ""),
                            "is_reply": False,
                        }
                    )

                    # Replies to the comment
                    if item["snippet"]["totalReplyCount"] > 0:
                        for reply in item.get("replies", {}).get("comments", []):
                            reply_snippet = reply["snippet"]
                            comments_list.append(
                                {
                                    "author": reply_snippet.get(
                                        "authorDisplayName", "[deleted]"
                                    ),
                                    "body": reply_snippet.get("textDisplay", ""),
                                    "created_utc": reply_snippet.get("publishedAt", ""),
                                    "is_reply": True,
                                }
                            )

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


# Initialize client
api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("YOUTUBE_API_KEY environment variable not set")

client = YouTubeClient(api_key)


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


def convert_youtube_timestamp(timestamp):
    """Convert YouTube ISO timestamp to Unix timestamp"""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None


def video_exists(video_id):
    """Check if video ID already exists in the CSV."""
    if not output_csv.exists():
        return False

    try:
        df = pd.read_csv(output_csv)
        return video_id in df["id"].values
    except Exception as e:
        logging.error(f"Error checking if video exists: {e}")
        return False


def save_video(video_id, video_data, search_method):
    """Save a single video to CSV."""
    try:
        snippet = video_data["snippet"]
        created_utc = convert_youtube_timestamp(snippet.get("publishedAt", ""))

        # Get comments
        comments = client.get_comments(video_id, max_results=100)
        comments_json = json.dumps(comments)

        record = {
            "id": video_id,
            "title": clean_text(snippet.get("title", "")),
            "description": clean_text(snippet.get("description", "")),
            "author": snippet.get("channelTitle", "[unknown]"),
            "created_utc": created_utc,
            "comments": comments_json,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

        # Append to CSV
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

        print(f"✓ Saved: {video_id} ({search_method})")
        return True

    except Exception as e:
        error_msg = f"Error saving video {video_id}: {str(e)}"
        logging.error(error_msg)
        return False


# Scraping Functions
def scrape_by_video_id(video_ids, rate_limit_sec=1):
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

    for video_id in tqdm(video_ids, desc="Processing videos", unit="video"):
        # Check if already exists
        if video_exists(video_id):
            skipped += 1
            time.sleep(rate_limit_sec)
            continue

        try:
            video_data = client.get_video(video_id)
            if video_data and save_video(video_id, video_data, "targeted_id"):
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


def scrape_by_keywords(keywords_list, limit_per_query=10, rate_limit_sec=1):
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

    for keyword in tqdm(keywords_list, desc="Processing keywords", unit="kw"):
        try:
            search_results = client.search_videos(keyword, max_results=limit_per_query)

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
                        f"keyword_search_{keyword}",
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
