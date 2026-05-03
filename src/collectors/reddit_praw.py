import pandas as pd
import praw
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

# Setup
load_dotenv()

# Get project root directory
project_root = Path(__file__).parent.parent.parent
print(f"Project root: {project_root}")

# Setup logging
log_dir = project_root / "logs" / "collectors"
log_dir.mkdir(parents=True, exist_ok=True)
error_log_path = log_dir / "reddit_scrape_errors.log"

logging.basicConfig(
    filename=str(error_log_path),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Setup output CSV path
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "reddit_scraped.csv"


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


# Reddit Client
class RedditClient:
    def __init__(self, config):
        self.config = config
        self.reddit = praw.Reddit(
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            user_agent=config["user_agent"],
        )
        self.reddit.read_only = True

    def get_subreddit(self, subreddit_name):
        subreddit = self.reddit.subreddit(subreddit_name)
        return subreddit

    def search(self, subreddit, keyword, limit=None):
        return subreddit.search(
            query=keyword, sort="new", time_filter="all", limit=limit
        )

    def get_submission(self, post_id):
        submission = self.reddit.submission(id=post_id)
        return submission


_client = None


def get_client():
    """Create the Reddit client only when a Reddit scraper actually runs."""
    global _client
    if _client is None:
        config = {
            "client_id": os.getenv("REDDIT_CLIENT_ID"),
            "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
            "user_agent": os.getenv("REDDIT_USER_AGENT", "Post Scraper by No-Web7094"),
        }
        missing = [key for key, value in config.items() if not value]
        if missing:
            env_names = {
                "client_id": "REDDIT_CLIENT_ID",
                "client_secret": "REDDIT_CLIENT_SECRET",
                "user_agent": "REDDIT_USER_AGENT",
            }
            missing_env = ", ".join(env_names[key] for key in missing)
            raise ValueError(f"Missing Reddit environment variable(s): {missing_env}")

        _client = RedditClient(config)

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


def epoch_to_utc_iso(value):
    if value in (None, ""):
        return None

    try:
        return (
            datetime.fromtimestamp(float(value), tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OSError):
        return None


def hash_identifier(value):
    """Hash author/channel identifiers instead of storing raw usernames."""
    if not value or value == "[deleted]":
        return None

    salt = os.getenv("AUTHOR_HASH_SALT", "")
    payload = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strip_reddit_kind(value):
    if not value:
        return None
    return value.split("_", 1)[1] if "_" in value else value


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
                "created_at_utc": epoch_to_utc_iso(comment.get("created_utc")),
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
        "source_platform": "reddit",
        "source_item_id": row.get("id"),
        "source_url": row.get("url"),
        "collection_method": "legacy",
        "collection_query": None,
        "collected_at_utc": None,
        "created_at_utc": epoch_to_utc_iso(row.get("created_utc")),
        "author_id_hash": hash_identifier(author),
        "title": clean_text(row.get("title")),
        "body_text": clean_text(row.get("submission")),
        "description": None,
        "transcript": None,
        "comments_json": json_dumps(comments),
        "engagement_json": json_dumps({"subreddit": row.get("subreddit")}),
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

    print(f"Migrated legacy Reddit CSV to schema. Backup: {backup_path}")


def submission_exists(submission_id):
    """Check if submission ID already exists in the CSV."""
    if not output_csv.exists():
        return False

    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_id = row.get("source_item_id") or row.get("id")
                if existing_id == submission_id:
                    return True
        return False
    except Exception as e:
        logging.error(f"Error checking if submission exists: {e}")
        return False


def get_comments_data(submission):
    """Extract all comments from a submission."""
    try:
        submission.comments.replace_more(limit=None)
        comments_list = []

        for comment in submission.comments.list():
            parent_item_id = strip_reddit_kind(comment.parent_id)
            comments_list.append(
                {
                    "source_item_id": comment.id,
                    "parent_item_id": parent_item_id,
                    "conversation_root_id": submission.id,
                    "source_url": f"https://www.reddit.com{comment.permalink}",
                    "author_id_hash": hash_identifier(
                        comment.author.name if comment.author else None
                    ),
                    "body": clean_text(comment.body),
                    "created_at_utc": epoch_to_utc_iso(comment.created_utc),
                    "score": getattr(comment, "score", None),
                    "is_reply": parent_item_id != submission.id,
                }
            )

        return json_dumps(comments_list)
    except Exception as e:
        logging.error(f"Error getting comments for submission {submission.id}: {e}")
        return json_dumps([])


def save_submission(submission, collection_method, collection_query, run_id):
    """Save a single submission to CSV."""
    try:
        ensure_output_csv()
        collected_at = utc_now_iso()
        comments_json = get_comments_data(submission)
        reddit_url = f"https://www.reddit.com{submission.permalink}"
        engagement = {
            "score": getattr(submission, "score", None),
            "upvote_ratio": getattr(submission, "upvote_ratio", None),
            "num_comments": getattr(submission, "num_comments", None),
            "subreddit": submission.subreddit.display_name,
        }
        raw_item = {
            "id": submission.id,
            "name": getattr(submission, "name", None),
            "title": clean_text(submission.title),
            "selftext": clean_text(submission.selftext),
            "created_utc": epoch_to_utc_iso(submission.created_utc),
            "url": getattr(submission, "url", None),
            "permalink": reddit_url,
            "subreddit": submission.subreddit.display_name,
            "engagement": engagement,
        }
        record = {
            "run_id": run_id,
            "source_platform": "reddit",
            "source_item_id": submission.id,
            "source_url": reddit_url,
            "collection_method": collection_method,
            "collection_query": collection_query,
            "collected_at_utc": collected_at,
            "created_at_utc": epoch_to_utc_iso(submission.created_utc),
            "author_id_hash": hash_identifier(
                submission.author.name if submission.author else None
            ),
            "title": clean_text(submission.title),
            "body_text": clean_text(submission.selftext),
            "description": None,
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

        print(f"Saved: {submission.id} ({collection_method})")
        return True
    except Exception as e:
        error_msg = f"Error saving submission {submission.id}: {str(e)}"
        logging.error(error_msg)
        return False


# Scraping Functions
def scrape_by_submission_id(submission_ids, rate_limit_sec=2, run_id=None):
    """
    Scrape targeted submissions by their IDs.

    Args:
        submission_ids: list of Reddit submission IDs
        rate_limit_sec: delay between requests in seconds
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING BY SUBMISSION ID")
    print(f"Total to process: {len(submission_ids)}")
    print(f"{'='*60}")

    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    reddit_client = get_client()

    for sub_id in tqdm(submission_ids, desc="Processing submissions", unit="post"):
        # Check if already exists
        if submission_exists(sub_id):
            skipped += 1
            time.sleep(rate_limit_sec)
            continue

        try:
            submission = reddit_client.get_submission(sub_id)
            if save_submission(submission, "targeted_id", sub_id, run_id):
                collected += 1
            else:
                failed += 1

        except Exception as e:
            logging.error(f"Error fetching submission {sub_id}: {str(e)}")
            failed += 1

        time.sleep(rate_limit_sec)

    print(f"\n{'='*60}")
    print(f"RESULTS - Collected: {collected} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*60}\n")
    return collected, skipped, failed


def scrape_by_subreddits_keywords(
    subreddits_list, keywords_list, limit_per_query=10, rate_limit_sec=2, run_id=None
):
    """
    Search keywords across all subreddits.

    Args:
        subreddits_list: list of subreddit names
        keywords_list: list of keywords to search
        limit_per_query: number of results per keyword search
        rate_limit_sec: delay between requests
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING BY SUBREDDITS + KEYWORDS")
    print(f"Subreddits: {', '.join(subreddits_list)}")
    print(f"Keywords: {', '.join(keywords_list)}")
    print(f"{'='*60}")

    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    reddit_client = get_client()

    for subreddit_name in tqdm(subreddits_list, desc="Subreddits", unit="sub"):
        try:
            subreddit = reddit_client.get_subreddit(subreddit_name)

            for keyword in tqdm(
                keywords_list, desc=f"  ↳ r/{subreddit_name}", leave=False, unit="kw"
            ):
                try:
                    results = reddit_client.search(
                        subreddit, keyword, limit=limit_per_query
                    )

                    for submission in results:
                        if submission_exists(submission.id):
                            skipped += 1
                        else:
                            if save_submission(
                                submission,
                                "keyword_search",
                                f"r/{subreddit_name}:{keyword}",
                                run_id,
                            ):
                                collected += 1
                            else:
                                failed += 1

                        time.sleep(rate_limit_sec)

                except Exception as e:
                    logging.error(
                        f"Error searching '{keyword}' in r/{subreddit_name}: {str(e)}"
                    )
                    failed += 1
                    time.sleep(rate_limit_sec)

        except Exception as e:
            logging.error(f"Error accessing subreddit r/{subreddit_name}: {str(e)}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS - Collected: {collected} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*60}\n")
    return collected, skipped, failed


def scrape_by_keywords(
    keywords_list, limit_per_query=10, rate_limit_sec=2, run_id=None
):
    """
    Search keywords across all of Reddit.

    Args:
        keywords_list: list of keywords to search
        limit_per_query: number of results per keyword search
        rate_limit_sec: delay between requests
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING BY KEYWORDS (ALL SUBREDDITS)")
    print(f"Keywords: {', '.join(keywords_list)}")
    print(f"{'='*60}")

    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    reddit_client = get_client()

    for keyword in tqdm(keywords_list, desc="Processing keywords", unit="kw"):
        try:
            results = reddit_client.reddit.subreddit("all").search(
                query=keyword, sort="new", time_filter="all", limit=limit_per_query
            )

            for submission in tqdm(
                results, desc=f"  ↳ '{keyword}'", leave=False, unit="post"
            ):
                if submission_exists(submission.id):
                    skipped += 1
                else:
                    if save_submission(
                        submission, "keyword_search", keyword, run_id
                    ):
                        collected += 1
                    else:
                        failed += 1

                time.sleep(rate_limit_sec)

        except Exception as e:
            logging.error(
                f"Error searching '{keyword}' across all subreddits: {str(e)}"
            )
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS - Collected: {collected} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*60}\n")
    return collected, skipped, failed


if __name__ == "__main__":
    # Load input data
    keywords = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "keywords.csv"
    )
    subreddits = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "subreddits.csv"
    )
    post_ids = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "reddit_post_ids.csv"
    )

    # Option 1: Scrape by submission IDs
    # submission_ids = post_ids["post_id"].tolist()
    # scrape_by_submission_id(submission_ids, rate_limit_sec=2)

    # Option 2: Scrape by subreddits + keywords
    subreddits_list = subreddits["subreddit"].tolist()
    keywords_list = keywords["keyword"].tolist()
    scrape_by_subreddits_keywords(
        subreddits_list, keywords_list, limit_per_query=10, rate_limit_sec=2
    )

    # Option 3: Scrape by keywords (all of Reddit)
    # scrape_by_keywords(keywords_list, limit_per_query=10, rate_limit_sec=2)
