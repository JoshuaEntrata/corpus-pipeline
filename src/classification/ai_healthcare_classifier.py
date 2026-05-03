import json
import os

from src.classification.prompts import (
    CLASSIFICATION_SCHEMA,
    CLASSIFIER_VERSION,
    SYSTEM_PROMPT,
    user_prompt,
)


DEFAULT_MODEL = "gpt-4o-mini"


def extract_output_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output = getattr(response, "output", None) or []
    parts = []
    for item in output:
        content = getattr(item, "content", None) or []
        for content_item in content:
            text = getattr(content_item, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def usage_to_dict(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def estimate_tokens(text):
    return max(1, len(text or "") // 4)


class AIHealthcareClassifier:
    def __init__(self, model=None):
        self.model = model or os.getenv("OPENAI_CLASSIFIER_MODEL") or DEFAULT_MODEL
        self.client = None

    def _client(self):
        if self.client is not None:
            return self.client

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when --use-model is enabled.")

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ImportError("Install the OpenAI SDK with: pip install openai") from exc

        self.client = OpenAI()
        return self.client

    def classify(self, text):
        client = self._client()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(text)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ai_healthcare_classification",
                    "strict": True,
                    "schema": CLASSIFICATION_SCHEMA,
                }
            },
        )
        output_text = extract_output_text(response)
        result = json.loads(output_text)
        result["model_name"] = self.model
        result["classifier_version"] = CLASSIFIER_VERSION
        result["usage"] = usage_to_dict(response)
        return result

