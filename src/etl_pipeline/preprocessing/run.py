from __future__ import annotations

from pathlib import Path
from typing import Any

from etl_pipeline.common.ids import row_key_from_record
from etl_pipeline.common.io import append_csv, read_csv, write_csv, write_json
from etl_pipeline.common.paths import master_dir, stage_run_dir, state_dir
from etl_pipeline.common.state import append_state_rows, read_state_keys, row_master_keys
from etl_pipeline.common.summaries import category_counts, collection_method_counts, platform_counts
from etl_pipeline.common.config import should_skip_seen
from etl_pipeline.preprocessing.schema import STANDARDIZED_FIELDS
from etl_pipeline.preprocessing.transform import transform_extraction_rows
from tqdm import tqdm


def run_preprocessing(
    pipeline_config: dict[str, Any],
    *,
    run_id: str,
    input_path: str | Path | None = None,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    source_path = Path(input_path) if input_path else master_dir(pipeline_config) / "extraction_raw.csv"
    raw_rows = read_csv(source_path)
    if limit is not None:
        raw_rows = raw_rows[:limit]
    transformed_rows, stats = transform_extraction_rows(raw_rows)

    master_path = master_dir(pipeline_config) / "standardized.csv"
    state_path = state_dir(pipeline_config) / "preprocessing_seen_ids.csv"
    seen = set()
    if should_skip_seen(pipeline_config, "preprocessing", force):
        seen |= row_master_keys(master_path)
        seen |= read_state_keys(state_path)

    new_rows = []
    skipped_existing = 0
    for row in tqdm(transformed_rows, desc="preprocessing", unit="row"):
        key = row_key_from_record(row)
        if key in seen:
            skipped_existing += 1
            continue
        seen.add(key)
        new_rows.append(row)

    run_dir = stage_run_dir(pipeline_config, "preprocessing", run_id)
    write_csv(run_dir / "standardized.csv", new_rows, STANDARDIZED_FIELDS)
    append_csv(master_path, new_rows, STANDARDIZED_FIELDS)
    append_state_rows(state_path, "preprocessing", new_rows, row_key_from_record)

    summary = {
        "stage": "preprocessing",
        "run_id": run_id,
        "input_rows": stats.input_rows,
        "output_rows": len(transformed_rows),
        "new_rows_added": len(new_rows),
        "skipped_existing_rows": skipped_existing,
        "dropped_empty_text_rows": stats.dropped_empty_text_rows,
        "deduplicated_rows": stats.deduplicated_rows,
        "comments_exploded": stats.comments_exploded,
        "replies_exploded": stats.replies_exploded,
        "invalid_comments_json_rows": stats.invalid_comments_json_rows,
        "num_platforms": len({row["platform"] for row in new_rows}),
        "platform_distribution": platform_counts(new_rows),
        "collection_method_distribution": collection_method_counts(new_rows),
        "category_distribution": category_counts(new_rows),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
