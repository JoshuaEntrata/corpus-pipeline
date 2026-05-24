from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from etl_pipeline.common.text import clean_text


@dataclass
class DetectionResult:
    languages: list[str]
    label: str
    detector: str
    confidence: float = 0.0
    needs_gpt_fallback: bool = False
    reason: str = ""


class FastTextDetector:
    def __init__(self, language_config: dict[str, Any]) -> None:
        self.config = language_config
        fasttext_config = language_config.get("fasttext", {})
        self.label_map = fasttext_config.get("label_map", {})
        self.supported = set(language_config.get("supported_languages", []))
        self.language_order = language_config.get("language_order", [])
        self.confidence_threshold = float(fasttext_config.get("confidence_threshold", 0.75))
        self.mixed_confidence_threshold = float(fasttext_config.get("mixed_confidence_threshold", 0.30))
        self.top_k = int(fasttext_config.get("top_k", 3))
        self.model = None
        model_path = Path(fasttext_config.get("model_path", ""))
        if model_path.exists():
            try:
                import fasttext

                self.model = fasttext.load_model(str(model_path))
            except ImportError:
                self.model = None

    def detect(self, text: str) -> DetectionResult:
        cleaned = clean_text(text)
        if len(cleaned) < 12:
            return self._fallback("too_short")
        if self._possible_hiligaynon(cleaned) and self.config.get("fallback", {}).get("use_gpt_for_possible_hiligaynon", True):
            return self._fallback("possible_hiligaynon")
        if self.model is None:
            return self._fallback("fasttext_model_unavailable")

        labels, scores = self.model.predict(cleaned.replace("\n", " "), k=self.top_k)
        mapped = []
        for label, score in zip(labels, scores):
            code = str(label).replace("__label__", "")
            language = self.label_map.get(code)
            if language:
                mapped.append((language, float(score)))

        if not mapped:
            return self._fallback("unsupported_language")
        top_language, top_score = mapped[0]
        if top_language not in self.supported:
            return self._fallback("unsupported_language")
        mixed = [language for language, score in mapped if score >= self.mixed_confidence_threshold and language in self.supported]
        if len(set(mixed)) > 1:
            return self._fallback("possible_mixed_language")
        if top_score < self.confidence_threshold:
            return self._fallback("low_confidence")
        return DetectionResult(
            languages=[top_language],
            label=normalize_language_label([top_language], self.language_order),
            detector="fasttext",
            confidence=top_score,
        )

    def _fallback(self, reason: str) -> DetectionResult:
        return DetectionResult([], "out_of_scope", "fasttext", needs_gpt_fallback=True, reason=reason)

    def _possible_hiligaynon(self, text: str) -> bool:
        lowered = text.casefold()
        cues = {"gid", "indi", "sang", "sa akon", "nga", "subong"}
        return any(cue in lowered for cue in cues)


def normalize_language_label(languages: list[str], language_order: list[str]) -> str:
    ordered = [language for language in language_order if language in set(languages)]
    if not ordered:
        return "out_of_scope"
    if len(ordered) == 1:
        return ordered[0]
    return "mixed_" + "_".join(ordered[:3])

