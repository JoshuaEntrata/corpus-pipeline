from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from etl_pipeline.classification.gpt_classifier import (
    GPTClassifier,
    INVALID_AI_ONLY,
    INVALID_CONFUSING_OR_INSUFFICIENT,
    INVALID_HEALTH_ONLY,
    INVALID_NEITHER,
    VALID_AI_HEALTHCARE,
)
from etl_pipeline.classification.schema import CLASSIFICATION_FIELDS
from etl_pipeline.classification.terms import TermMatcher, WITH_AI_AND_HEALTH_TERMS
from etl_pipeline.common.config import load_yaml, should_skip_seen
from etl_pipeline.common.costs import TokenUsage, add_usage, calculate_cost
from etl_pipeline.common.ids import row_key_from_record
from etl_pipeline.common.io import append_csv, read_csv, write_csv, write_json
from etl_pipeline.common.logging import log_stage_error
from etl_pipeline.common.paths import logs_dir, master_dir, stage_run_dir, state_dir
from etl_pipeline.common.state import append_state_rows, read_state_keys, row_master_keys
from etl_pipeline.common.summaries import collection_method_counts, distribution, platform_counts
from etl_pipeline.common.time import utc_now_iso
from tqdm import tqdm

WriteBuffer = list[dict[str, Any]]


def run_classification(
    pipeline_config: dict[str, Any],
    *,
    run_id: str,
    input_path: str | Path | None = None,
    terms_path: str | Path = "config/terms.yaml",
    models_path: str | Path = "config/models.yaml",
    force: bool = False,
    limit: int | None = None,
    prefilter_only: bool = False,
) -> dict[str, Any]:
    source_path = Path(input_path) if input_path else master_dir(pipeline_config) / "standardized.csv"
    input_rows = read_csv(source_path)
    if limit is not None:
        input_rows = input_rows[:limit]

    all_master_path = master_dir(pipeline_config) / "classification_all.csv"
    valid_master_path = master_dir(pipeline_config) / "classification_valid_only.csv"
    state_path = state_dir(pipeline_config) / "classification_seen_ids.csv"
    seen = set()
    if should_skip_seen(pipeline_config, "classification", force):
        seen |= row_master_keys(all_master_path)
        seen |= read_state_keys(state_path)

    matcher = TermMatcher.from_config_path(str(terms_path))
    models_config = load_yaml(models_path)
    model = models_config["openai"]["classification_model"]
    batch_size = max(int(models_config.get("openai", {}).get("classification_batch_size", 20)), 1)
    max_inflight_batches = max(int(models_config.get("openai", {}).get("classification_max_inflight_batches", 1)), 1)
    write_buffer_size = max(int(models_config.get("openai", {}).get("classification_write_buffer_size", 200)), 1)
    total_usage = TokenUsage()
    rows_sent_to_gpt = 0
    gpt_candidate_count = 0
    provided_label_count = 0
    rule_filtered_count = 0
    skipped_existing = 0
    output_rows: list[dict[str, Any]] = []
    pending_writes: WriteBuffer = []

    run_dir = stage_run_dir(pipeline_config, "classification", run_id)
    all_run_path = run_dir / "classification_all.csv"
    valid_run_path = run_dir / "classification_valid_only.csv"

    if prefilter_only:
        with tqdm(total=len(input_rows), desc="classification_prefilter", unit="row") as pbar:
            for row in input_rows:
                key = row_key_from_record(row)
                if key in seen:
                    skipped_existing += 1
                    pbar.update(1)
                    continue
                if _provided_label(row, "provided_classification_label", "classification_label"):
                    provided_label_count += 1
                    output = _classification_output(row, "provided_label", "prefilter_only", "provided_label")
                    output_rows.append(output)
                    seen.add(key)
                    pending_writes.append(output)
                    if len(pending_writes) >= write_buffer_size:
                        _flush_classification_outputs(
                            pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path
                        )
                    pbar.update(1)
                    continue
                available_terms = matcher.classify_available_terms(row.get("text", ""))
                if available_terms == WITH_AI_AND_HEALTH_TERMS:
                    gpt_candidate_count += 1
                    # In prefilter-only mode, treat AI+health term matches as valid rows.
                    model_classification = VALID_AI_HEALTHCARE
                else:
                    rule_filtered_count += 1
                    model_classification = _rule_label_for_terms(available_terms)
                output = _classification_output(row, available_terms, "prefilter_only", model_classification)
                output_rows.append(output)
                seen.add(key)
                pending_writes.append(output)
                if len(pending_writes) >= write_buffer_size:
                    _flush_classification_outputs(
                        pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path
                    )
                pbar.update(1)

        if pending_writes:
            _flush_classification_outputs(pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path)
        write_csv(all_run_path, read_csv(all_run_path), CLASSIFICATION_FIELDS)
        write_csv(valid_run_path, read_csv(valid_run_path), CLASSIFICATION_FIELDS)
        write_csv(valid_master_path, read_csv(valid_master_path), CLASSIFICATION_FIELDS)
        summary = {
            "stage": "classification",
            "run_id": run_id,
            "mode": "prefilter_only",
            "input_rows": len(input_rows),
            "new_rows_processed": len(output_rows),
            "skipped_existing_rows": skipped_existing,
            "provided_label_rows": provided_label_count,
            "rule_filtered_rows": rule_filtered_count,
            "gpt_candidate_rows": gpt_candidate_count,
            "gpt_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "rows_sent_to_gpt": 0,
            },
        }
        write_json(run_dir / "summary.json", summary)
        return summary

    pending_gpt: list[dict[str, Any]] = []
    futures: set[Future[dict[str, Any]]] = set()

    with ThreadPoolExecutor(max_workers=max_inflight_batches) as executor:
        with tqdm(total=len(input_rows), desc="classification", unit="row") as pbar:
            for row in input_rows:
                key = row_key_from_record(row)
                if key in seen:
                    skipped_existing += 1
                    pbar.update(1)
                    continue
                provided_classification = _provided_label(row, "provided_classification_label", "classification_label")
                if provided_classification:
                    output = _classification_output(row, "provided_label", "provided_label", provided_classification)
                    output_rows.append(output)
                    pending_writes.append(output)
                    seen.add(key)
                    if len(pending_writes) >= write_buffer_size:
                        _flush_classification_outputs(
                            pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path
                        )
                    pbar.update(1)
                    pbar.set_postfix(estimated_cost_usd=_cost_so_far(total_usage, models_config, model))
                    continue

                available_terms = matcher.classify_available_terms(row.get("text", ""))
                if available_terms == WITH_AI_AND_HEALTH_TERMS:
                    pending_gpt.append({"key": key, "row": row, "available_terms": available_terms})
                    if len(pending_gpt) >= batch_size:
                        batch = pending_gpt[:]
                        pending_gpt.clear()
                        future = executor.submit(_process_gpt_batch, batch, models_config, model, pipeline_config)
                        futures.add(future)
                    while len(futures) >= max_inflight_batches:
                        done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            futures.remove(future)
                            usage = future.result()
                            total_usage = add_usage(total_usage, usage["usage"])
                            rows_sent_to_gpt += usage["rows_sent"]
                            for output in usage["outputs"]:
                                output_rows.append(output)
                                pending_writes.append(output)
                                seen.add(row_key_from_record(output))
                            if len(pending_writes) >= write_buffer_size:
                                _flush_classification_outputs(
                                    pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path
                                )
                            pbar.update(usage["rows_sent"])
                            pbar.set_postfix(estimated_cost_usd=_cost_so_far(total_usage, models_config, model))
                    continue

                model_classification = _rule_label_for_terms(available_terms)
                output = _classification_output(row, available_terms, "rules_terms_filter", model_classification)
                output_rows.append(output)
                pending_writes.append(output)
                seen.add(key)
                if len(pending_writes) >= write_buffer_size:
                    _flush_classification_outputs(
                        pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path
                    )
                pbar.update(1)
                pbar.set_postfix(estimated_cost_usd=_cost_so_far(total_usage, models_config, model))

            if pending_gpt:
                future = executor.submit(_process_gpt_batch, pending_gpt[:], models_config, model, pipeline_config)
                futures.add(future)

            for future in list(futures):
                usage = future.result()
                total_usage = add_usage(total_usage, usage["usage"])
                rows_sent_to_gpt += usage["rows_sent"]
                for output in usage["outputs"]:
                    output_rows.append(output)
                    pending_writes.append(output)
                    seen.add(row_key_from_record(output))
                if len(pending_writes) >= write_buffer_size:
                    _flush_classification_outputs(
                        pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path
                    )
                pbar.update(usage["rows_sent"])
                pbar.set_postfix(estimated_cost_usd=_cost_so_far(total_usage, models_config, model))

    if pending_writes:
        _flush_classification_outputs(pending_writes, all_run_path, valid_run_path, all_master_path, valid_master_path, state_path)

    write_csv(all_run_path, read_csv(all_run_path), CLASSIFICATION_FIELDS)
    write_csv(valid_run_path, read_csv(valid_run_path), CLASSIFICATION_FIELDS)
    write_csv(valid_master_path, read_csv(valid_master_path), CLASSIFICATION_FIELDS)

    gpt_usage = calculate_cost(total_usage, models_config, model)
    gpt_usage["rows_sent_to_gpt"] = rows_sent_to_gpt
    summary = {
        "stage": "classification",
        "run_id": run_id,
        "input_rows": len(input_rows),
        "new_rows_processed": len(output_rows),
        "skipped_existing_rows": skipped_existing,
        "num_platforms": len({row["platform"] for row in output_rows}),
        "platform_distribution": platform_counts(output_rows),
        "collection_method_distribution": collection_method_counts(output_rows),
        "available_terms_distribution": distribution(output_rows, "available_terms"),
        "model_classification_distribution": distribution(output_rows, "model_classification"),
        "gpt_usage": gpt_usage,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def _process_gpt_batch(
    pending_gpt: list[dict[str, Any]],
    models_config: dict[str, Any],
    model: str,
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    rows_sent = len(pending_gpt)
    usage = TokenUsage()
    batch_failed = False
    try:
        classifier = GPTClassifier(models_config)
        batch_payload = [
            {"row_id": item["key"], "text": item["row"].get("text", "")}
            for item in pending_gpt
        ]
        results, usage = classifier.classify_batch(batch_payload)
    except Exception as exc:
        batch_failed = True
        results = {
            item["key"]: (INVALID_CONFUSING_OR_INSUFFICIENT, "GPT batch failed; fallback label applied.")
            for item in pending_gpt
        }
        for item in pending_gpt:
            row = item["row"]
            log_stage_error(
                logs_dir(pipeline_config),
                stage="classification",
                platform=row.get("platform", ""),
                collection_method=row.get("collection_method", ""),
                id_or_query=row.get("id", ""),
                error=exc,
            )

    outputs: list[dict[str, Any]] = []
    for item in pending_gpt:
        row = item["row"]
        key = item["key"]
        model_classification, _reason = results.get(
            key,
            (INVALID_CONFUSING_OR_INSUFFICIENT, "Missing batch result."),
        )
        model_used = "gpt_error_fallback" if batch_failed else model
        output = _classification_output(row, item["available_terms"], model_used, model_classification)
        outputs.append(output)

    return {"usage": usage, "rows_sent": rows_sent, "outputs": outputs}


def _classification_output(
    row: dict[str, Any],
    available_terms: str,
    model_used: str,
    model_classification: str,
) -> dict[str, Any]:
    return {
        **{field: row.get(field, "") for field in CLASSIFICATION_FIELDS},
        "available_terms": available_terms,
        "classified_at_utc": utc_now_iso(),
        "model_used": model_used,
        "model_classification": model_classification,
    }


def _provided_label(row: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = str(row.get(field, "") or "").strip()
        if value:
            return value
    return ""


def _flush_classification_outputs(
    outputs: list[dict[str, Any]],
    all_run_path: Path,
    valid_run_path: Path,
    all_master_path: Path,
    valid_master_path: Path,
    state_path: Path,
) -> None:
    if not outputs:
        return
    append_csv(all_run_path, outputs, CLASSIFICATION_FIELDS)
    append_csv(all_master_path, outputs, CLASSIFICATION_FIELDS)
    append_state_rows(state_path, "classification", outputs, row_key_from_record)
    valid_outputs = [output for output in outputs if output.get("model_classification") == VALID_AI_HEALTHCARE]
    if valid_outputs:
        append_csv(valid_run_path, valid_outputs, CLASSIFICATION_FIELDS)
        append_csv(valid_master_path, valid_outputs, CLASSIFICATION_FIELDS)
    outputs.clear()


def _rule_label_for_terms(available_terms: str) -> str:
    if available_terms == "with_ai_term_only":
        return INVALID_AI_ONLY
    if available_terms == "with_health_term_only":
        return INVALID_HEALTH_ONLY
    return INVALID_NEITHER


def _cost_so_far(usage: TokenUsage, models_config: dict[str, Any], model: str) -> float:
    return calculate_cost(usage, models_config, model)["estimated_cost_usd"]
