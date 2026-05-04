from src.config.keywords import contains_term, load_keyword_terms


def parse_keyword_config(path):
    return load_keyword_terms(path)


def prefilter_text(
    text, is_empty=False, is_too_short=False, ai_terms=None, health_terms=None
):
    if ai_terms is None or health_terms is None:
        config_ai_terms, config_health_terms = load_keyword_terms()
        ai_terms = config_ai_terms if ai_terms is None else ai_terms
        health_terms = config_health_terms if health_terms is None else health_terms

    text = text or ""

    if is_empty:
        return {
            "prefilter_result": "empty_text",
            "label": "invalid_confusing_or_insufficient",
            "confidence": 0.95,
            "reason_short": "Text is empty.",
            "needs_model": False,
        }

    if is_too_short:
        return {
            "prefilter_result": "too_short",
            "label": "invalid_confusing_or_insufficient",
            "confidence": 0.85,
            "reason_short": "Text is too short or context-dependent.",
            "needs_model": False,
        }

    has_ai = contains_term(text, ai_terms)
    has_health = contains_term(text, health_terms)

    if has_ai and has_health:
        return {
            "prefilter_result": "likely_relevant",
            "label": "valid_ai_healthcare",
            "confidence": 0.72,
            "reason_short": "Prefilter found both AI and healthcare terms in the standalone text.",
            "needs_model": True,
        }

    if has_ai:
        return {
            "prefilter_result": "likely_ai_only",
            "label": "invalid_ai_only",
            "confidence": 0.78,
            "reason_short": "Prefilter found AI terms but no healthcare terms.",
            "needs_model": False,
        }

    if has_health:
        return {
            "prefilter_result": "likely_health_only",
            "label": "invalid_health_only",
            "confidence": 0.78,
            "reason_short": "Prefilter found healthcare terms but no AI terms.",
            "needs_model": False,
        }

    return {
        "prefilter_result": "likely_neither",
        "label": "invalid_neither",
        "confidence": 0.82,
        "reason_short": "Prefilter found neither AI nor healthcare terms.",
        "needs_model": False,
    }
