# corpus-pipeline

## Collector Runner

Manual CSV files are ingested by dropping `.csv` files into:

```text
data/manual_uploads/
```

The importer accepts flexible text columns such as `text`, `post_text`,
`comment_text`, `body_text`, `content`, or `message`. If a row does not include
an ID column, the pipeline generates a stable ID from the file name, row number,
and text.

Run collectors:

```bash
python -m src.orchestration.run_collectors --source manual_csv
python -m src.orchestration.run_collectors --source reddit --mode targeted
python -m src.orchestration.run_collectors --source youtube --mode keyword
python -m src.orchestration.run_collectors --all
```

Config lives in `configs/collectors.yaml`. Raw schema fields are documented in
`configs/schema.yaml`, and collected IDs are tracked in
`data/registry/collected_ids.csv`.
Collector outputs keep only extracted text/source attributes plus
`source_platform` and `collection_method`.

## Preprocessing

Normalize raw collector outputs into one standalone text row per title, body,
description, comment, reply, transcript, or manual CSV text:

```bash
python -m src.orchestration.run_preprocessing
python -m src.orchestration.run_preprocessing --source reddit
python -m src.orchestration.run_preprocessing --source youtube
python -m src.orchestration.run_preprocessing --source manual_csv
```

Outputs are written to `data/processed/normalized_text_rows_<run_id>.csv`.
The preprocessing CSV is intentionally compact: source platform, text-row ID,
preprocessed text, collection method, keyword flags, and classification/language
detection flags.

## Classification

Classify normalized rows for AI-in-healthcare relevance. By default this uses
the local multilingual prefilter only and does not spend API calls:

```bash
python -m src.orchestration.run_classification
```

To call OpenAI for likely relevant rows, opt in explicitly:

```bash
python -m src.orchestration.run_classification --use-model --max-model-calls 50
```

Outputs are written to `data/classified/ai_healthcare_classified_<run_id>.csv`.
The classified CSV keeps only the text, label, confidence, short reason,
timestamp, and prefilter fields.
