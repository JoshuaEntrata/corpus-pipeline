from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    usage_source: str = "openai_api_usage"

    @property
    def uncached_input_tokens(self) -> int:
        return max(self.input_tokens - self.cached_input_tokens, 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_usage(api_response: Any) -> TokenUsage:
    usage = _get(api_response, "usage", api_response)
    input_tokens = int(_get(usage, "input_tokens", _get(usage, "prompt_tokens", 0)) or 0)
    output_tokens = int(_get(usage, "output_tokens", _get(usage, "completion_tokens", 0)) or 0)
    details = _get(usage, "input_tokens_details", _get(usage, "prompt_tokens_details", {})) or {}
    cached = int(_get(details, "cached_tokens", 0) or 0)
    return TokenUsage(input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output_tokens)


def estimate_usage(prompt: str, response_text: str = "") -> TokenUsage:
    # A conservative rough estimate when exact tokenizer usage is unavailable.
    input_tokens = max(len(prompt) // 4, 1) if prompt else 0
    output_tokens = max(len(response_text) // 4, 1) if response_text else 0
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, usage_source="estimated_tokenizer")


def add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    source = left.usage_source if left.usage_source == right.usage_source else "mixed"
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        usage_source=source,
    )


def calculate_cost(usage: TokenUsage, pricing_config: dict[str, Any], model: str) -> dict[str, Any]:
    model_prices = pricing_config.get("pricing", {}).get("models", {}).get(model)
    if not model_prices:
        raise KeyError(f"Pricing config missing for model: {model}")
    input_cost = usage.uncached_input_tokens / 1_000_000 * float(model_prices["input_usd_per_1m_tokens"])
    cached_input_cost = usage.cached_input_tokens / 1_000_000 * float(
        model_prices["cached_input_usd_per_1m_tokens"]
    )
    output_cost = usage.output_tokens / 1_000_000 * float(model_prices["output_usd_per_1m_tokens"])
    estimated = input_cost + cached_input_cost + output_cost
    return {
        "model_used": model,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "uncached_input_tokens": usage.uncached_input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_cost_usd": round(input_cost, 10),
        "cached_input_cost_usd": round(cached_input_cost, 10),
        "output_cost_usd": round(output_cost, 10),
        "estimated_cost_usd": round(estimated, 10),
        "pricing_source": "config/models.yaml",
        "usage_source": usage.usage_source,
    }

