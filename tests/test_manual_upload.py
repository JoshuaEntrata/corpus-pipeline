import csv
import json
from pathlib import Path

from etl_pipeline.classification.run import run_classification
from etl_pipeline.common.io import read_csv, write_csv
from etl_pipeline.extraction.manual_upload import load_manual_upload_rows, run_manual_upload_extraction
from etl_pipeline.language_detection.run import run_language_detection
from etl_pipeline.preprocessing.schema import STANDARDIZED_FIELDS
from etl_pipeline.preprocessing.transform import transform_extraction_rows


def test_manual_upload_normalizes_flat_and_json_rows(tmp_path: Path) -> None:
    manual_dir = tmp_path / "manual_upload"
    manual_dir.mkdir()
    (manual_dir / "manual01.csv").write_text(
        "source_platform,text,url\nfacebook,AI health text,https://example.test/a\n",
        encoding="utf-8",
    )
    payload = {
        "results": [
            {
                "post_id": "p1",
                "url": "https://facebook.test/p1",
                "message": "ChatGPT doctor post",
                "timestamp": 1778839239,
                "author": {"name": "Example Author"},
            }
        ]
    }
    with (manual_dir / "manual04.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["platform", "query", "json", "collection_method"])
        writer.writeheader()
        writer.writerow(
            {
                "platform": "facebook",
                "query": "ai healthcare philippines",
                "json": json.dumps(payload),
                "collection_method": "rapidapi_facebook_scraper",
            }
        )

    rows, stats = load_manual_upload_rows(manual_dir)

    assert stats.output_rows == 2
    assert rows[0]["platform"] == "facebook"
    assert rows[0]["collection_method"] == "manual_upload"
    assert rows[1]["id"] == "p1"
    assert rows[1]["author"] == "Example Author"
    assert rows[1]["created_at_utc"].endswith("Z")


def test_manual_upload_preserves_provided_language_label(tmp_path: Path) -> None:
    manual_dir = tmp_path / "manual_upload"
    manual_dir.mkdir()
    (manual_dir / "manual03.csv").write_text(
        "source,text,url,provided_language_label\nfacebook,AI tambal,https://example.test/a,cebuano\n",
        encoding="utf-8",
    )
    config = {
        "paths": {
            "local_data_dir": str(tmp_path / "local_data"),
            "master_dir": str(tmp_path / "local_data" / "master"),
        }
    }

    summary = run_manual_upload_extraction(config, run_id="manual-test", input_dir=manual_dir)
    raw_rows = read_csv(summary["output_file"])
    standardized, _stats = transform_extraction_rows(raw_rows)

    assert raw_rows[0]["provided_language_label"] == "cebuano"
    assert standardized[0]["provided_language_label"] == "cebuano"


def test_classification_uses_provided_label_without_gpt(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    source = tmp_path / "standardized.csv"
    write_csv(
        source,
        [
            {
                "platform": "facebook",
                "collection_method": "manual_upload",
                "id": "m1",
                "text": "short text",
                "category": "post",
                "associated_id": "m1",
                "provided_language_label": "cebuano",
                "provided_classification_label": "valid_ai_healthcare",
            }
        ],
        STANDARDIZED_FIELDS,
    )

    summary = run_classification(config, run_id="provided-class", input_path=source)
    output = read_csv(Path(config["paths"]["master_dir"]) / "classification_all.csv")
    valid = read_csv(Path(config["paths"]["master_dir"]) / "classification_valid_only.csv")

    assert summary["gpt_usage"]["rows_sent_to_gpt"] == 0
    assert output[0]["model_used"] == "provided_label"
    assert output[0]["model_classification"] == "valid_ai_healthcare"
    assert output[0]["provided_language_label"] == "cebuano"
    assert len(valid) == 1


def test_language_detection_uses_provided_label_without_gpt(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    source = tmp_path / "classification_valid_only.csv"
    write_csv(
        source,
        [
            {
                "platform": "facebook",
                "collection_method": "manual_upload",
                "id": "m1",
                "text": "short text",
                "category": "post",
                "associated_id": "m1",
                "provided_language_label": "cebuano",
                "model_classification": "valid_ai_healthcare",
            }
        ],
        [
            "platform",
            "collection_method",
            "id",
            "text",
            "category",
            "associated_id",
            "provided_language_label",
            "model_classification",
        ],
    )

    summary = run_language_detection(config, run_id="provided-language", input_path=source)
    output = read_csv(Path(config["paths"]["master_dir"]) / "language_detection.csv")

    assert summary["gpt_usage"]["rows_sent_to_gpt"] == 0
    assert output[0]["language_label"] == "cebuano"
    assert output[0]["model_classification"] == "provided_label"
    assert output[0]["language_detected"] == '["cebuano"]'


def _pipeline_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "local_data_dir": str(tmp_path / "local_data"),
            "master_dir": str(tmp_path / "local_data" / "master"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "stages": {
            "classification": {"skip_seen_ids": True},
            "language_detection": {"skip_seen_ids": True},
        },
    }
