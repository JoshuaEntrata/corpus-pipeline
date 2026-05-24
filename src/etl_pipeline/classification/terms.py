from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from etl_pipeline.common.config import load_yaml
from etl_pipeline.common.text import normalize_for_matching

WITH_AI_AND_HEALTH_TERMS = "with_ai_and_health_terms"
WITH_AI_TERM_ONLY = "with_ai_term_only"
WITH_HEALTH_TERM_ONLY = "with_health_term_only"
NO_VALID_TERMS = "no_valid_terms"


@dataclass
class TermMatcher:
    ai_terms: list[str]
    health_terms: list[str]
    case_sensitive: bool = False
    use_word_boundaries: bool = True

    @classmethod
    def from_config_path(cls, path: str = "config/terms.yaml") -> "TermMatcher":
        config = load_yaml(path)
        matching = config.get("matching", {})
        return cls(
            ai_terms=flatten_terms(config.get("ai_terms", [])),
            health_terms=flatten_terms(config.get("health_terms", [])),
            case_sensitive=bool(matching.get("case_sensitive", False)),
            use_word_boundaries=bool(matching.get("use_word_boundaries", True)),
        )

    def classify_available_terms(self, text: str) -> str:
        normalized = normalize_for_matching(text, case_sensitive=self.case_sensitive)
        has_ai = self._has_any(normalized, self.ai_terms)
        has_health = self._has_any(normalized, self.health_terms)
        if has_ai and has_health:
            return WITH_AI_AND_HEALTH_TERMS
        if has_ai:
            return WITH_AI_TERM_ONLY
        if has_health:
            return WITH_HEALTH_TERM_ONLY
        return NO_VALID_TERMS

    def _has_any(self, normalized_text: str, terms: list[str]) -> bool:
        for term in terms:
            normalized_term = normalize_for_matching(term, case_sensitive=self.case_sensitive)
            pattern = term_pattern(normalized_term, use_word_boundaries=self.use_word_boundaries)
            if re.search(pattern, normalized_text):
                return True
        return False


def flatten_terms(value: Any) -> list[str]:
    if isinstance(value, dict):
        terms: list[str] = []
        for nested in value.values():
            terms.extend(flatten_terms(nested))
        return terms
    if isinstance(value, list):
        terms = []
        for item in value:
            terms.extend(flatten_terms(item))
        return terms
    if value in (None, ""):
        return []
    return [str(value).strip()]


def term_pattern(term: str, *, use_word_boundaries: bool = True) -> str:
    escaped = re.escape(term).replace(r"\*", r"[\w-]*")
    if not use_word_boundaries:
        return escaped
    return rf"(?<!\w){escaped}(?!\w)"
