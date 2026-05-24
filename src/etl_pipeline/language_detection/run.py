from __future__ import annotations

from pathlib import Path
from typing import Any

from etl_pipeline.common.config import load_yaml, should_skip_seen
from etl_pipeline.common.costs import TokenUsage, add_usage, calculate_cost
from etl_pipeline.common.ids import row_key_from_record
from etl_pipeline.common.io import append_csv, json_dumps, read_csv, write_csv, write_json
from etl_pipeline.common.logging import log_stage_error
from etl_pipeline.common.paths import logs_dir, master_dir, stage_run_dir, state_dir
from etl_pipeline.common.state import append_state_rows, read_state_keys, row_master_keys
from etl_pipeline.common.summaries import collection_method_counts, distribution, platform_counts
from etl_pipeline.common.time import utc_now_iso
from etl_pipeline.language_detection.fasttext_detector import FastTextDetector
from etl_pipeline.language_detection.gpt_detector import GPTLanguageDetector
from etl_pipeline.language_detection.schema import LANGUAGE_DETECTION_FIELDS
from tqdm import tqdm


def run_language_detection(
    pipeline_config: dict[str, Any],
    *,
    run_id: str,
    input_path: str | Path | None = None,
    language_path: str | Path = "config/language.yaml",
    models_path: str | Path = "config/models.yaml",
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    source_path = Path(input_path) if input_path else master_dir(pipeline_config) / "classification_valid_only.csv"
    input_rows = read_csv(source_path)
    if limit is not None:
        input_rows = input_rows[:limit]

    master_path = master_dir(pipeline_config) / "language_detection.csv"
    state_path = state_dir(pipeline_config) / "language_detection_seen_ids.csv"
    seen = set()
    if should_skip_seen(pipeline_config, "language_detection", force):
        seen |= row_master_keys(master_path)
        seen |= read_state_keys(state_path)

    language_config = load_yaml(language_path)
    models_config = load_yaml(models_path)
    model = models_config["openai"]["language_model"]
    batch_size = max(int(models_config.get("openai", {}).get("language_batch_size", 20)), 1)
    fasttext_detector = FastTextDetector(language_config)
    gpt_detector: GPTLanguageDetector | None = None
    total_usage = TokenUsage()
    rows_sent_to_gpt = 0
    skipped_existing = 0
    output_rows: list[dict[str, Any]] = []

    pending_gpt: list[dict[str, Any]] = []

    with tqdm(total=len(input_rows), desc="language_detection", unit="row") as pbar:
        for row in input_rows:
            key = row_key_from_record(row)
            if key in seen:
                skipped_existing += 1
                pbar.update(1)
                continue

            result = fasttext_detector.detect(row.get("text", ""))
            if result.needs_gpt_fallback and language_config.get("fallback", {}).get("use_gpt_for_low_confidence", True):
                pending_gpt.append({"key": key, "row": row})
                if len(pending_gpt) >= batch_size:
                    usage = _process_gpt_language_batch(
                        pending_gpt,
                        models_config,
                        language_config,
                        model,
                        pipeline_config,
                        output_rows,
                        seen,
                        gpt_detector,
                    )
                    gpt_detector = usage["detector"]
                    total_usage = add_usage(total_usage, usage["usage"])
                    rows_sent_to_gpt += usage["rows_sent"]
                    pending_gpt.clear()
                    pbar.update(usage["rows_sent"])
                    pbar.set_postfix(estimated_cost_usd=_cost_so_far(total_usage, models_config, model))
                continue

            output_rows.append(_language_output(row, result.languages, result.label, result.detector))
            seen.add(key)
            pbar.update(1)
            pbar.set_postfix(estimated_cost_usd=_cost_so_far(total_usage, models_config, model))

        if pending_gpt:
            usage = _process_gpt_language_batch(
                pending_gpt,
                models_config,
                language_config,
                model,
                pipeline_config,
                output_rows,
                seen,
                gpt_detector,
            )
            gpt_detector = usage["detector"]
            total_usage = add_usage(total_usage, usage["usage"])
            rows_sent_to_gpt += usage["rows_sent"]
            pbar.update(usage["rows_sent"])
            pbar.set_postfix(estimated_cost_usd=_cost_so_far(total_usage, models_config, model))

    run_dir = stage_run_dir(pipeline_config, "language_detection", run_id)
    write_csv(run_dir / "language_detection.csv", output_rows, LANGUAGE_DETECTION_FIELDS)
    append_csv(master_path, output_rows, LANGUAGE_DETECTION_FIELDS)
    append_state_rows(state_path, "language_detection", output_rows, row_key_from_record)

    gpt_usage = calculate_cost(total_usage, models_config, model)
    gpt_usage["rows_sent_to_gpt"] = rows_sent_to_gpt
    summary = {
        "stage": "language_detection",
        "run_id": run_id,
        "input_file": str(source_path),
        "input_scope": "valid_ai_healthcare_only",
        "input_rows": len(input_rows),
        "new_rows_processed": len(output_rows),
        "skipped_existing_rows": skipped_existing,
        "num_platforms": len({row["platform"] for row in output_rows}),
        "platform_distribution": platform_counts(output_rows),
        "collection_method_distribution": collection_method_counts(output_rows),
        "language_label_distribution": distribution(output_rows, "language_label"),
        "detector_distribution": distribution(output_rows, "model_classification"),
        "gpt_usage": gpt_usage,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def _process_gpt_language_batch(
    pending_gpt: list[dict[str, Any]],
    models_config: dict[str, Any],
    language_config: dict[str, Any],
    model: str,
    pipeline_config: dict[str, Any],
    output_rows: list[dict[str, Any]],
    seen: set[str],
    detector: GPTLanguageDetector | None,
) -> dict[str, Any]:
    rows_sent = len(pending_gpt)
    usage = TokenUsage()
    batch_failed = False
    try:
        detector = detector or GPTLanguageDetector(models_config, language_config)
        batch_payload = [
            {"row_id": item["key"], "text": item["row"].get("text", "")}
            for item in pending_gpt
        ]
        results, usage = detector.detect_batch(batch_payload)
    except Exception as exc:
        batch_failed = True
        results = {item["key"]: ([], language_config.get("labels", {}).get("out_of_scope", "out_of_scope")) for item in pending_gpt}
        for item in pending_gpt:
            row = item["row"]
            log_stage_error(
                logs_dir(pipeline_config),
                stage="language_detection",
                platform=row.get("platform", ""),
                collection_method=row.get("collection_method", ""),
                id_or_query=row.get("id", ""),
                error=exc,
            )

    for item in pending_gpt:
        languages, label = results.get(
            item["key"],
            ([], language_config.get("labels", {}).get("out_of_scope", "out_of_scope")),
        )
        detector_used = "gpt_error_fallback" if batch_failed else model
        output_rows.append(_language_output(item["row"], languages, label, detector_used))
        seen.add(item["key"])

    return {"detector": detector, "usage": usage, "rows_sent": rows_sent}


def _language_output(row: dict[str, Any], languages: list[str], label: str, detector_used: str) -> dict[str, Any]:
    return {
        "platform": row.get("platform", ""),
        "collection_method": row.get("collection_method", ""),
        "id": row.get("id", ""),
        "text": row.get("text", ""),
        "category": row.get("category", ""),
        "associated_id": row.get("associated_id", ""),
        "language_detected": json_dumps(languages, []),
        "language_label": label,
        "language_detected_at_utc": utc_now_iso(),
        "model_classification": detector_used,
    }


def _cost_so_far(usage: TokenUsage, models_config: dict[str, Any], model: str) -> float:
    return calculate_cost(usage, models_config, model)["estimated_cost_usd"]
