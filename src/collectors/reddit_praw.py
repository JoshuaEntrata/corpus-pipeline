import pandas as pd
import praw
import os
import csv
import json
import time
import logging
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

# Create CSV headers if file doesn't exist
if not output_csv.exists():
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "title",
                "submission",
                "comments",
                "author",
                "created_utc",
                "url",
                "subreddit",
            ],
        )
        writer.writeheader()

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
        self.reddit_read_only = True

    def get_subreddit(self, subreddit_name):
        subreddit = self.reddit.subreddit(subreddit_name)
        return subreddit

    def search(self, subreddit, keyword):
        return subreddit.search(query=keyword, sort="new", time_filter="all")

    def get_submission(self, post_id):
        submission = self.reddit.submission(id=post_id)
        return submission


# Initialize client
config = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": "Post Scraper by No-Web7094",
}

client = RedditClient(config)


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


def submission_exists(submission_id):
    """Check if submission ID already exists in the CSV."""
    if not output_csv.exists():
        return False

    try:
        df = pd.read_csv(output_csv)
        return submission_id in df["id"].values
    except Exception as e:
        logging.error(f"Error checking if submission exists: {e}")
        return False


def get_comments_data(submission):
    """Extract all comments from a submission."""
    try:
        submission.comments.replace_more(limit=None)
        comments_list = []

        for comment in submission.comments.list():
            comments_list.append(
                {
                    "author": comment.author.name if comment.author else "[deleted]",
                    "body": clean_text(comment.body),
                    "created_utc": comment.created_utc,
                }
            )

        return json.dumps(comments_list)
    except Exception as e:
        logging.error(f"Error getting comments for submission {submission.id}: {e}")
        return json.dumps([])


def save_submission(submission, scrape_method):
    """Save a single submission to CSV."""
    try:
        record = {
            "id": submission.id,
            "title": clean_text(submission.title),
            "submission": clean_text(submission.selftext),
            "comments": get_comments_data(submission),
            "author": submission.author.name if submission.author else "[deleted]",
            "created_utc": submission.created_utc,
            "url": submission.url,
            "subreddit": submission.subreddit.display_name,
        }

        # Append to CSV
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

        print(f"✓ Saved: {submission.id} ({scrape_method})")
        return True
    except Exception as e:
        error_msg = f"Error saving submission {submission.id}: {str(e)}"
        logging.error(error_msg)
        return False


# Scraping Functions
def scrape_by_submission_id(submission_ids, rate_limit_sec=2):
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

    for sub_id in tqdm(submission_ids, desc="Processing submissions", unit="post"):
        # Check if already exists
        if submission_exists(sub_id):
            skipped += 1
            time.sleep(rate_limit_sec)
            continue

        try:
            submission = client.get_submission(sub_id)
            if save_submission(submission, "targeted_id"):
                collected += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1

        time.sleep(rate_limit_sec)

    print(f"\n{'='*60}")
    print(f"RESULTS - Collected: {collected} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*60}\n")
    return collected, skipped, failed


def scrape_by_subreddits_keywords(
    subreddits_list, keywords_list, limit_per_query=10, rate_limit_sec=2
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

    for subreddit_name in tqdm(subreddits_list, desc="Subreddits", unit="sub"):
        try:
            subreddit = client.get_subreddit(subreddit_name)

            for keyword in tqdm(
                keywords_list, desc=f"  ↳ r/{subreddit_name}", leave=False, unit="kw"
            ):
                try:
                    results = client.search(subreddit, keyword)

                    for submission in results:
                        if submission_exists(submission.id):
                            skipped += 1
                        else:
                            if save_submission(submission, f"keyword_search_{keyword}"):
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


def scrape_by_keywords(keywords_list, limit_per_query=10, rate_limit_sec=2):
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

    for keyword in tqdm(keywords_list, desc="Processing keywords", unit="kw"):
        try:
            results = client.reddit.subreddit("all").search(
                query=keyword, sort="new", time_filter="all", limit=limit_per_query
            )

            for submission in tqdm(
                results, desc=f"  ↳ '{keyword}'", leave=False, unit="post"
            ):
                if submission_exists(submission.id):
                    skipped += 1
                else:
                    if save_submission(submission, f"keyword_search_{keyword}"):
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
