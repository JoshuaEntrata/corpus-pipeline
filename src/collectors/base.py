import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from reddit_praw import (
    scrape_by_submission_id,
    scrape_by_subreddits_keywords,
    scrape_by_keywords as scrape_reddit_by_keywords,
)
from youtube_data import (
    scrape_by_video_id,
    scrape_by_keywords as scrape_youtube_by_keywords,
)

# Setup
load_dotenv()

# Get project root directory
project_root = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run Reddit PRAW scraper with different collection methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--reddit_scrape_by_id",
        action="store_true",
        help="Scrape Reddit submissions by their IDs",
    )

    parser.add_argument(
        "--reddit_scrape_by_subreddits_and_keywords",
        action="store_true",
        help="Scrape by searching keywords across specific subreddits",
    )

    parser.add_argument(
        "--reddit_scrape_by_keywords",
        action="store_true",
        help="Scrape by searching keywords across all subreddits",
    )

    parser.add_argument(
        "--youtube_scrape_by_id",
        action="store_true",
        help="Scrape Youtube videos by their IDs",
    )

    parser.add_argument(
        "--youtube_scrape_by_keywords",
        action="store_true",
        help="Scrape by searching keywords across all Youtube videos",
    )

    parser.add_argument("--all", action="store_true", help="Run all scraping methods")

    parser.add_argument(
        "--rate-limit",
        type=int,
        default=2,
        help="Rate limit delay between requests in seconds (default: 2)",
    )

    parser.add_argument(
        "--limit", type=int, default=10, help="Limit of results per query (default: 10)"
    )

    args = parser.parse_args()

    # Load input data
    print(f"\nLoading input data from {project_root}...\n")

    keywords_df = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "keywords.csv"
    )
    subreddits_df = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "subreddits.csv"
    )
    reddit_post_ids_df = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "reddit_post_ids.csv"
    )
    youtube_post_ids_df = pd.read_csv(
        project_root / "src" / "collectors" / "inputs" / "youtube_post_ids.csv"
    )

    keywords = keywords_df["keyword"].tolist()
    subreddits = subreddits_df["subreddit"].tolist()
    reddit_post_ids = reddit_post_ids_df["post_id"].tolist()
    youtube_post_ids = youtube_post_ids_df["video_id"].tolist()

    print(f"Loaded {len(keywords)} keywords")
    print(f"Loaded {len(subreddits)} subreddits")
    print(f"Loaded {len(reddit_post_ids)} Reddit post IDs\n")
    print(f"Loaded {len(youtube_post_ids)} Youtube post IDs\n")

    # Track total results
    total_collected = 0
    total_skipped = 0
    total_failed = 0

    # If no options specified, show help
    if not any(
        [
            args.reddit_scrape_by_id,
            args.reddit_scrape_by_subreddits_and_keywords,
            args.reddit_scrape_by_keywords,
            args.youtube_scrape_by_id,
            args.youtube_scrape_by_keywords,
            args.all,
        ]
    ):
        parser.print_help()
        return

    # Run selected scraping methods
    if args.all:
        args.reddit_scrape_by_id = True
        args.reddit_scrape_by_subreddits_and_keywords = True
        args.reddit_scrape_by_keywords = True
        args.youtube_scrape_by_id = True
        args.youtube_scrape_by_keywords = True

    # Scrape by Reddit submission IDs
    if args.youtube_scrape_by_keywords:
        collected, skipped, failed = scrape_by_submission_id(
            reddit_post_ids, rate_limit_sec=args.rate_limit
        )
        total_collected += collected
        total_skipped += skipped
        total_failed += failed

    # Scrape by subreddits + keywords
    if args.reddit_scrape_by_subreddits_and_keywords:
        collected, skipped, failed = scrape_by_subreddits_keywords(
            subreddits,
            keywords,
            limit_per_query=args.limit,
            rate_limit_sec=args.rate_limit,
        )
        total_collected += collected
        total_skipped += skipped
        total_failed += failed

    # Scrape by keywords (all subreddits)
    if args.reddit_scrape_by_keywords:
        collected, skipped, failed = scrape_reddit_by_keywords(
            keywords, limit_per_query=args.limit, rate_limit_sec=args.rate_limit
        )
        total_collected += collected
        total_skipped += skipped
        total_failed += failed

    if args.youtube_scrape_by_id:
        collected, skipped, failed = scrape_by_video_id(
            youtube_post_ids, rate_limit_sec=args.rate_limit
        )
        total_collected += collected
        total_skipped += skipped
        total_failed += failed

    # Scrape by keywords (all Youtube videos)
    if args.youtube_scrape_by_keywords:
        collected, skipped, failed = scrape_youtube_by_keywords(
            keywords, limit_per_query=args.limit, rate_limit_sec=args.rate_limit
        )
        total_collected += collected
        total_skipped += skipped
        total_failed += failed

    # Final summary
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total Collected: {total_collected}")
    print(f"Total Skipped:   {total_skipped}")
    print(f"Total Failed:    {total_failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
