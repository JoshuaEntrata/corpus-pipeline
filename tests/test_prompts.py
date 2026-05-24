from etl_pipeline.classification.prompts import build_batch_classification_prompt
from etl_pipeline.language_detection.prompts import build_batch_language_prompt


def test_batch_classification_prompt_includes_row_ids() -> None:
    prompt = build_batch_classification_prompt([{"row_id": "row-1", "text": "AI in hospitals"}])
    assert "valid_ai_healthcare" in prompt
    assert '"row_id": "row-1"' in prompt


def test_batch_language_prompt_includes_batch_schema() -> None:
    prompt = build_batch_language_prompt([{"row_id": "row-1", "text": "AI sa hospital"}])
    assert "language_detected" in prompt
    assert '"row_id": "row-1"' in prompt
