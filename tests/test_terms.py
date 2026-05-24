from etl_pipeline.classification.terms import (
    NO_VALID_TERMS,
    WITH_AI_AND_HEALTH_TERMS,
    WITH_AI_TERM_ONLY,
    WITH_HEALTH_TERM_ONLY,
    TermMatcher,
    flatten_terms,
)


def matcher() -> TermMatcher:
    return TermMatcher(ai_terms=["ai", "machine learning"], health_terms=["health", "diagnosis"])


def test_detects_ai_and_health_terms() -> None:
    assert matcher().classify_available_terms("AI could help diagnosis workflows.") == WITH_AI_AND_HEALTH_TERMS


def test_detects_single_term_groups() -> None:
    assert matcher().classify_available_terms("machine learning system") == WITH_AI_TERM_ONLY
    assert matcher().classify_available_terms("community health program") == WITH_HEALTH_TERM_ONLY


def test_word_boundaries_avoid_false_ai_match() -> None:
    assert matcher().classify_available_terms("plain words only") == NO_VALID_TERMS


def test_wildcard_terms_match_suffixes() -> None:
    wildcard_matcher = TermMatcher(ai_terms=["chatbot*"], health_terms=["diagnos*"])
    assert wildcard_matcher.classify_available_terms("Chatbots for diagnosis") == WITH_AI_AND_HEALTH_TERMS


def test_flatten_terms_supports_language_grouped_config() -> None:
    terms = flatten_terms({"english": ["ai"], "cebuano": ["modelo"]})
    assert terms == ["ai", "modelo"]
