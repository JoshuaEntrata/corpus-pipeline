from __future__ import annotations

import json
import os
import time
from typing import Any

from etl_pipeline.common.config import ConfigError
from etl_pipeline.common.costs import TokenUsage, extract_usage
from etl_pipeline.language_detection.fasttext_detector import normalize_language_label
from etl_pipeline.language_detection.prompts import build_batch_language_prompt, build_language_prompt


class GPTLanguageDetector:
    def __init__(self, models_config: dict[str, Any], language_config: dict[str, Any]) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigError("Missing OPENAI_API_KEY in environment.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigError("Install the openai package to use GPT language detection.") from exc
        self.models_config = models_config
        self.language_config = language_config
        self.model = models_config["openai"]["language_model"]
        self.max_retries = int(models_config.get("openai", {}).get("max_retries", 3))
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=models_config["openai"].get("timeout_seconds", 60))

    def detect(self, text: str) -> tuple[list[str], str, TokenUsage]:
        results, usage = self.detect_batch([{"row_id": "row_0", "text": text}])
        languages, label = results["row_0"]
        return languages, label, usage

    def detect_batch(self, rows: list[dict[str, str]]) -> tuple[dict[str, tuple[list[str], str]], TokenUsage]:
        prompt = build_batch_language_prompt(rows)
        last_error: Exception | None = None
        supported = set(self.language_config.get("supported_languages", []))
        order = self.language_config.get("language_order", [])
        max_languages = int(self.language_config.get("fallback", {}).get("max_languages_per_row", 3))
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=self.models_config["openai"].get("temperature", 0),
                )
                parsed = json.loads(_response_text(response))
                results = _parse_batch_results(parsed, rows, supported, order, max_languages)
                return results, extract_usage(response)
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error


def _parse_batch_results(
    parsed: dict[str, Any],
    rows: list[dict[str, str]],
    supported: set[str],
    order: list[str],
    max_languages: int,
) -> dict[str, tuple[list[str], str]]:
    raw_results = parsed.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    by_id: dict[str, tuple[list[str], str]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("row_id", ""))
        raw_languages = item.get("language_detected", [])
        if not isinstance(raw_languages, list):
            raw_languages = []
        normalized_raw = [str(language) for language in raw_languages]
        has_unsupported = any(language not in supported for language in normalized_raw)
        languages = [language for language in normalized_raw if language in supported][:max_languages]
        if has_unsupported:
            label = "out_of_scope"
        else:
            label = normalize_language_label(languages, order)
        by_id[row_id] = (languages, label)
    for row in rows:
        by_id.setdefault(row["row_id"], ([], "out_of_scope"))
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
