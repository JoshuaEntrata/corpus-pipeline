# Extraction Inputs

Extraction reads these CSV files by default. Keep headers even when a file has no rows.

- `reddit_ids.csv`: `id`
- `youtube_ids.csv`: `id`
- `twitter_ids.csv`: `id`
- `subreddits.csv`: `subreddit`
- `keywords.csv`: `keyword`

Keywords are applied to Reddit, YouTube, and Twitter/X keyword searches. Reddit subreddit searches are created from every `subreddit` and `keyword` combination.
