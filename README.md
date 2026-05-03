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
