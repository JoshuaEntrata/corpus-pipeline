# Corpus Pipeline

Readable, modular ETL pipeline for collecting social-platform posts/comments, standardizing them, classifying AI-in-healthcare relevance, and detecting language only for valid AI-and-healthcare rows.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Optional live extraction dependencies:

```bash
pip install -e ".[reddit,youtube,language]"
```

Twitter/X extraction uses `requests`.

## Commands

```bash
python -m etl_pipeline.cli extract --config config/pipeline.yaml --inputs inputs
python -m etl_pipeline.cli manual-upload --config config/pipeline.yaml
python -m etl_pipeline.cli preprocess --config config/pipeline.yaml
python -m etl_pipeline.cli classify --config config/pipeline.yaml
python -m etl_pipeline.cli detect-language --config config/pipeline.yaml
python -m etl_pipeline.cli run-all --config config/pipeline.yaml
```

Useful flags:

```bash
--run-id 20260524T000000Z
--input local_data/master/standardized.csv
--output local_data/manual/manual_extraction_raw.csv
--limit 100
--force
```

## Extraction Inputs

Extraction is driven by CSV files in `inputs/`:

- `reddit_ids.csv` with one `id` column
- `youtube_ids.csv` with one `id` column
- `twitter_ids.csv` with one `id` column
- `subreddits.csv` with one `subreddit` column
- `keywords.csv` with one `keyword` column

Keywords are used for platform keyword searches. Reddit subreddit searches are created from every subreddit/keyword combination.

Enable or disable collectors in `config/pipeline.yaml`:

```yaml
stages:
  extraction:
    platforms:
      reddit: true
      youtube: true
      twitter: false
```

Manual uploads can be normalized from `local_data/manual_upload/*.csv`:

```bash
python -m etl_pipeline.cli manual-upload --config config/pipeline.yaml
python -m etl_pipeline.cli preprocess --config config/pipeline.yaml --input local_data/manual/manual_extraction_raw.csv
```

If an input row includes `provided_classification_label` or `provided_language_label`, the corresponding stage uses that value directly and skips model labeling for that row.

## Data Flow

Outputs are written to timestamped run folders under `local_data/<stage>/<run_id>/` and cumulative master files under `local_data/master/`. State files under `local_data/state/` let stages skip already-processed rows.

Language detection intentionally reads `local_data/master/classification_valid_only.csv`, so only `valid_ai_healthcare` rows are sent through language detection.

## Notes

Secrets belong in `.env`. Generated data, logs, local models, caches, and virtual environments are ignored by git.

The default OpenAI model, batch sizes, and pricing live in `config/models.yaml`. Verify pricing against official OpenAI documentation before production runs.
