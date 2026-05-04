import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You detect language for short social-media text about AI in healthcare.
Use only the provided text. Do not infer from source, topic, author, or missing context.
Return JSON only.

Allowed base languages are English, Tagalog, Cebuano, Ilocano, and Hiligaynon.
If exactly two allowed languages are clearly present, use the matching mixed_<language>_<language> label.
If three or more allowed languages are clearly present, use mixed_multiple.
If the text mixes an allowed language with another unsupported language, use mixed_other.
If the text is too short to judge, use too_short.
If there is enough text but the language remains unclear, use unknown."""


def output_schema(target_labels):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "confidence", "reason_short"],
        "properties": {
            "label": {"type": "string", "enum": list(target_labels)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_short": {"type": "string"},
        },
    }


def user_prompt(text, target_labels):
    labels = ", ".join(target_labels)
    return (
        "Detect the language label for this text.\n\n"
        f"Allowed labels: {labels}\n\n"
        f"Text:\n{text}"
    )


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


class OpenAILanguageFallback:
    def __init__(self, model=None):
        self.model = model or os.getenv("OPENAI_LANGUAGE_MODEL") or DEFAULT_MODEL
        self.client = None

    def is_available(self):
        return bool(os.getenv("OPENAI_API_KEY"))

    def _client(self):
        if self.client is not None:
            return self.client

        if not self.is_available():
            raise ValueError("OPENAI_API_KEY is required for language fallback.")

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ImportError("Install the OpenAI SDK with: pip install openai") from exc

        self.client = OpenAI()
        return self.client

    def detect(self, text, target_labels):
        client = self._client()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(text, target_labels)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "language_detection",
                    "strict": True,
                    "schema": output_schema(target_labels),
                }
            },
        )
        result = json.loads(extract_output_text(response))
        result["model_name"] = self.model
        result["usage"] = usage_to_dict(response)
        return result

