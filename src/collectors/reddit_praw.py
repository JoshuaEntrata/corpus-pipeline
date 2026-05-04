import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.contracts import RAW_COLLECTION_FIELDS

log_dir = project_root / "logs" / "collectors"
log_dir.mkdir(parents=True, exist_ok=True)
error_log_path = log_dir / "reddit_scrape_errors.log"

logging.basicConfig(
    filename=str(error_log_path),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "reddit_scraped.csv"

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
        return self.reddit.subreddit(subreddit_name)

    def search(self, subreddit, keyword, limit=None):
        return subreddit.search(
            query=keyword, sort="new", time_filter="all", limit=limit
        )

    def get_submission(self, post_id):
        return self.reddit.submission(id=post_id)


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


def clean_text(text):
    if not text:
        return text
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split())


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


def strip_reddit_kind(value):
    if not value:
        return None
    return value.split("_", 1)[1] if "_" in value else value


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


def normalize_comment_record(comment, root_id):
    parent_item_id = comment.get("parent_item_id") or comment.get("parent_id")
    created_at_utc = comment.get("created_at_utc") or epoch_to_utc_iso(
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
    """Map older Reddit scraper CSV shapes into the compact raw schema."""
    source_item_id = row.get("source_item_id") or row.get("id")
    comments = load_comment_records(row)

    return {
        "source_platform": row.get("source_platform") or "reddit",
        "source_item_id": source_item_id,
        "source_url": row.get("source_url") or row.get("url"),
        "collection_method": row.get("collection_method") or "legacy",
        "created_at_utc": row.get("created_at_utc")
        or epoch_to_utc_iso(row.get("created_utc")),
        "title": clean_text(row.get("title")),
        "body_text": clean_text(
            row.get("body_text") or row.get("submission") or row.get("selftext")
        ),
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
        f"{output_csv.stem}_legacy_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{output_csv.suffix}"
    )
    output_csv.replace(backup_path)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(migrate_legacy_record(row))

    tqdm.write(f"reddit | migrate_schema | backup={backup_path}")


def submission_exists(submission_id):
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
        logging.error(f"reddit duplicate check failed: {e}")
        return False


def get_comments_data(submission):
    try:
        submission.comments.replace_more(limit=None)
        comments_list = []

        for comment in submission.comments.list():
            parent_item_id = strip_reddit_kind(comment.parent_id)
            comments_list.append(
                {
                    "source_item_id": comment.id,
                    "parent_item_id": parent_item_id,
                    "body": clean_text(comment.body),
                    "created_at_utc": epoch_to_utc_iso(comment.created_utc),
                    "is_reply": parent_item_id != submission.id,
                }
            )

        return json_dumps(comments_list)
    except Exception as e:
        logging.error(f"reddit comment scrape failed: {e}")
        return json_dumps([])


def save_submission(
    submission, collection_method, _collection_query=None, _run_id=None
):
    try:
        ensure_output_csv()
        reddit_url = f"https://www.reddit.com{submission.permalink}"
        record = {
            "source_platform": "reddit",
            "source_item_id": submission.id,
            "source_url": reddit_url,
            "collection_method": collection_method,
            "created_at_utc": epoch_to_utc_iso(submission.created_utc),
            "title": clean_text(submission.title),
            "body_text": clean_text(submission.selftext),
            "description": None,
            "transcript": None,
            "comments_json": get_comments_data(submission),
        }

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
            writer.writerow(record)

        return True
    except Exception as e:
        logging.error(f"reddit {collection_method} save failed: {str(e)}")
        return False


def scrape_by_submission_id(submission_ids, rate_limit_sec=2, run_id=None):
    """Scrape targeted submissions by their IDs."""
    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    reddit_client = get_client()
    total = len(submission_ids)
    log_scrape_start("reddit", "targeted_id", total, "posts")

    with tqdm(total=total, desc="reddit targeted_id", unit="post") as progress:
        for sub_id in submission_ids:
            if submission_exists(sub_id):
                skipped += 1
            else:
                try:
                    submission = reddit_client.get_submission(sub_id)
                    if save_submission(submission, "targeted_id", sub_id, run_id):
                        collected += 1
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(f"reddit targeted_id fetch failed: {str(e)}")
                    failed += 1

            time.sleep(rate_limit_sec)
            progress.update(1)
            update_progress(progress, collected, skipped, failed)

    log_scrape_done("reddit", "targeted_id", total, total, collected, skipped, failed)
    return collected, skipped, failed


def scrape_by_subreddits_keywords(
    subreddits_list, keywords_list, limit_per_query=100, rate_limit_sec=2, run_id=None
):
    """Search keywords across configured subreddits."""
    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    reddit_client = get_client()
    total_queries = len(subreddits_list) * len(keywords_list)
    log_scrape_start("reddit", "subreddit_keyword", total_queries, "queries")

    with tqdm(
        total=total_queries,
        desc="reddit subreddit_keyword",
        unit="query",
    ) as progress:
        for subreddit_name in subreddits_list:
            try:
                subreddit = reddit_client.get_subreddit(subreddit_name)
            except Exception as e:
                logging.error(f"reddit subreddit_keyword access failed: {str(e)}")
                failed += len(keywords_list)
                progress.update(len(keywords_list))
                update_progress(progress, collected, skipped, failed)
                continue

            for keyword in keywords_list:
                try:
                    results = reddit_client.search(
                        subreddit, keyword, limit=limit_per_query
                    )

                    for submission in results:
                        if submission_exists(submission.id):
                            skipped += 1
                        elif save_submission(
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
                    logging.error(f"reddit subreddit_keyword search failed: {str(e)}")
                    failed += 1

                progress.update(1)
                update_progress(progress, collected, skipped, failed)

    log_scrape_done(
        "reddit",
        "subreddit_keyword",
        total_queries,
        total_queries,
        collected,
        skipped,
        failed,
    )
    return collected, skipped, failed


def scrape_by_keywords(
    keywords_list, limit_per_query=100, rate_limit_sec=2, run_id=None
):
    """Search keywords across all of Reddit."""
    collected = 0
    skipped = 0
    failed = 0
    run_id = run_id or utc_now_iso()
    ensure_output_csv()
    reddit_client = get_client()
    total_queries = len(keywords_list)
    log_scrape_start("reddit", "keyword", total_queries, "queries")

    with tqdm(total=total_queries, desc="reddit keyword", unit="query") as progress:
        for keyword in keywords_list:
            try:
                results = reddit_client.reddit.subreddit("all").search(
                    query=keyword, sort="new", time_filter="all", limit=limit_per_query
                )

                for submission in results:
                    if submission_exists(submission.id):
                        skipped += 1
                    elif save_submission(
                        submission,
                        "keyword_search",
                        keyword,
                        run_id,
                    ):
                        collected += 1
                    else:
                        failed += 1

                    time.sleep(rate_limit_sec)
            except Exception as e:
                logging.error(f"reddit keyword search failed: {str(e)}")
                failed += 1

            progress.update(1)
            update_progress(progress, collected, skipped, failed)

    log_scrape_done(
        "reddit", "keyword", total_queries, total_queries, collected, skipped, failed
    )
    return collected, skipped, failed


if __name__ == "__main__":
    keywords = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "keywords.csv"
    )
    subreddits = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "subreddits.csv"
    )

    scrape_by_subreddits_keywords(
        subreddits["subreddit"].tolist(),
        keywords["keyword"].tolist(),
        limit_per_query=100,
        rate_limit_sec=2,
    )
