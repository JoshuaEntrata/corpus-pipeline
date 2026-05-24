from etl_pipeline.common.costs import TokenUsage, calculate_cost, extract_usage


MODELS_CONFIG = {
    "pricing": {
        "models": {
            "gpt-5.4-mini": {
                "input_usd_per_1m_tokens": 0.75,
                "cached_input_usd_per_1m_tokens": 0.075,
                "output_usd_per_1m_tokens": 4.50,
            }
        }
    }
}


def test_calculate_cost_uses_cached_and_uncached_prices() -> None:
    cost = calculate_cost(
        TokenUsage(input_tokens=100_000, cached_input_tokens=60_000, output_tokens=10_000),
        MODELS_CONFIG,
        "gpt-5.4-mini",
    )

    assert cost["uncached_input_tokens"] == 40_000
    assert cost["input_cost_usd"] == 0.03
    assert cost["cached_input_cost_usd"] == 0.0045
    assert cost["output_cost_usd"] == 0.045
    assert cost["estimated_cost_usd"] == 0.0795


def test_extract_usage_supports_prompt_token_shape() -> None:
    usage = extract_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "prompt_tokens_details": {"cached_tokens": 40},
            }
        }
    )

    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 40
    assert usage.output_tokens == 25

