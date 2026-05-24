from __future__ import annotations

import json
import os
import time
from typing import Any

from etl_pipeline.classification.prompts import build_batch_classification_prompt, build_classification_prompt
from etl_pipeline.common.config import ConfigError
from etl_pipeline.common.costs import TokenUsage, estimate_usage, extract_usage

VALID_AI_HEALTHCARE = "valid_ai_healthcare"
INVALID_AI_ONLY = "invalid_ai_only"
INVALID_HEALTH_ONLY = "invalid_health_only"
INVALID_NEITHER = "invalid_neither"
INVALID_CONFUSING_OR_INSUFFICIENT = "invalid_confusing_or_insufficient"
UNSURE = "unsure"
CLASSIFICATION_LABELS = {
    VALID_AI_HEALTHCARE,
    INVALID_AI_ONLY,
    INVALID_HEALTH_ONLY,
    INVALID_NEITHER,
    INVALID_CONFUSING_OR_INSUFFICIENT,
    UNSURE,
}


class GPTClassifier:
    def __init__(self, models_config: dict[str, Any]) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigError("Missing OPENAI_API_KEY in environment.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigError("Install the openai package to use GPT classification.") from exc
        self.config = models_config
        self.model = models_config["openai"]["classification_model"]
        self.max_retries = int(models_config.get("openai", {}).get("max_retries", 3))
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=models_config["openai"].get("timeout_seconds", 60))

    def classify(self, text: str) -> tuple[str, str, TokenUsage]:
        results, usage = self.classify_batch([{"row_id": "row_0", "text": text}])
        label, reason = results["row_0"]
        return label, reason, usage

    def classify_batch(self, rows: list[dict[str, str]]) -> tuple[dict[str, tuple[str, str]], TokenUsage]:
        prompt = build_batch_classification_prompt(rows)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=self.config["openai"].get("temperature", 0),
                )
                response_text = _response_text(response)
                parsed = json.loads(response_text)
                results = _parse_batch_results(parsed, rows)
                return results, extract_usage(response)
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error


def classify_with_estimate(
    text: str,
    model: str,
    label: str = INVALID_CONFUSING_OR_INSUFFICIENT,
) -> tuple[str, str, TokenUsage]:
    prompt = build_classification_prompt(text)
    return label, "Estimated fallback after GPT error.", estimate_usage(prompt, '{"model_classification": "..."}')


def _parse_batch_results(parsed: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    raw_results = parsed.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    by_id: dict[str, tuple[str, str]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("row_id", ""))
        label = str(item.get("model_classification", INVALID_CONFUSING_OR_INSUFFICIENT))
        if label not in CLASSIFICATION_LABELS:
            label = INVALID_CONFUSING_OR_INSUFFICIENT
        by_id[row_id] = (label, str(item.get("reason", "")))
    for row in rows:
        by_id.setdefault(row["row_id"], (INVALID_CONFUSING_OR_INSUFFICIENT, "Missing batch result."))
    return by_id


def _response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    if isinstance(response, dict) and response.get("output_text"):
        return str(response["output_text"])
    output = getattr(response, "output", None) or (response.get("output") if isinstance(response, dict) else None)
    if output:
        first = output[0]
        content = getattr(first, "content", None) or first.get("content", [])
        if content:
            item = content[0]
            text = getattr(item, "text", None) or item.get("text")
            if text:
                return str(text)
    return str(response)
