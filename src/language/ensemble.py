import re
from itertools import combinations
from pathlib import Path

from src.config.keywords import keyword_pattern
from src.language.openai_fallback import OpenAILanguageFallback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_DETECTOR_VERSION = "language_ensemble_v1"

BASE_LANGUAGE_ORDER = [
    "english",
    "tagalog",
    "cebuano",
    "ilocano",
    "hiligaynon",
]
DEFAULT_LANGUAGE_TERMS = {
    "english": [
        "the",
        "and",
        "is",
        "are",
        "this",
        "that",
        "with",
        "for",
        "healthcare",
        "medical",
        "patient",
        "doctor",
        "hospital",
        "artificial intelligence",
        "machine learning",
    ],
    "tagalog": ["hindi", "kasi", "naman", "yung", "kalusugan", "gamot"],
    "cebuano": ["dili", "kaayo", "unsa", "kahimsog", "tambal"],
    "ilocano": ["adda", "awan", "saan", "salun-at", "agas"],
    "hiligaynon": ["indi", "gid", "sang", "bulong", "balatian"],
}

LANGUAGE_CODE_MAP = {
    "en": "english",
    "tl": "tagalog",
    "fil": "tagalog",
    "ceb": "cebuano",
    "ilo": "ilocano",
    "hil": "hiligaynon",
}


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalized_terms(terms):
    values = []
    seen = set()
    for term in terms or []:
        value = str(term or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def merge_language_terms(config_terms):
    terms = {language: list(values) for language, values in DEFAULT_LANGUAGE_TERMS.items()}
    for language, values in (config_terms or {}).items():
        terms[language] = normalized_terms(values)
    return {language: normalized_terms(values) for language, values in terms.items()}


def configured_base_languages(target_labels):
    return [
        language
        for language in BASE_LANGUAGE_ORDER
        if language in target_labels
    ]


def mixed_label(languages, target_labels):
    ordered = [
        language
        for language in BASE_LANGUAGE_ORDER
        if language in set(languages)
    ]
    if len(ordered) >= 3:
        return "mixed_multiple" if "mixed_multiple" in target_labels else "mixed_other"
    if len(ordered) != 2:
        return "unknown"

    label = f"mixed_{ordered[0]}_{ordered[1]}"
    return label if label in target_labels else "mixed_other"


def label_languages(label):
    if label in BASE_LANGUAGE_ORDER:
        return {label}
    if not label.startswith("mixed_"):
        return set()

    parts = label.removeprefix("mixed_").split("_")
    return {part for part in parts if part in BASE_LANGUAGE_ORDER}


def ordered_languages(languages):
    return [
        language
        for language in BASE_LANGUAGE_ORDER
        if language in set(languages or [])
    ]


def languages_from_scores(scores, min_rule_score, mixed_score_ratio):
    if not scores:
        return []

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1]
    if top_score < min_rule_score:
        return []

    languages = [
        language
        for language, score in sorted_scores
        if score >= min_rule_score and score / top_score >= mixed_score_ratio
    ]
    return ordered_languages(languages)


def labels_compatible(left, right):
    if left == right:
        return True
    left_languages = label_languages(left)
    right_languages = label_languages(right)
    return bool(left_languages and right_languages and left_languages & right_languages)


def term_weight(term):
    clean = term.rstrip("*")
    word_count = len(clean.split())
    if word_count > 1:
        return 2.0
    if len(clean) <= 2:
        return 0.35
    if len(clean) <= 3:
        return 0.6
    return 1.0


def build_term_language_counts(language_terms):
    counts = {}
    for terms in language_terms.values():
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
    return counts


def count_term_matches(text, term):
    pattern = keyword_pattern(term)
    if not pattern:
        return 0
    return len(re.findall(pattern, text))


class LanguageEnsembleDetector:
    def __init__(self, config):
        self.config = config or {}
        self.target_labels = list(self.config.get("target_languages") or [])
        if not self.target_labels:
            self.target_labels = self.default_target_labels()
        self.target_label_set = set(self.target_labels)
        self.base_languages = configured_base_languages(self.target_label_set)
        self.thresholds = self.config.get("thresholds") or {}
        self.detector_config = self.config.get("detectors") or {}
        self.fallback_config = self.config.get("fallback") or {}
        self.language_terms = merge_language_terms(self.config.get("language_terms"))
        self.term_language_counts = build_term_language_counts(self.language_terms)
        self.fasttext_model = None
        self.fasttext_status = None
        self.openai_fallback = OpenAILanguageFallback(
            model=self.fallback_config.get("model")
        )

    @staticmethod
    def default_target_labels():
        labels = list(BASE_LANGUAGE_ORDER)
        labels.extend(
            f"mixed_{left}_{right}"
            for left, right in combinations(BASE_LANGUAGE_ORDER, 2)
        )
        labels.extend(["mixed_multiple", "mixed_other", "unknown", "too_short"])
        return labels

    def min_chars(self):
        return int(self.thresholds.get("min_chars_for_detection", 15))

    def strong_confidence(self):
        return float(self.thresholds.get("strong_confidence", 0.80))

    def fallback_confidence(self):
        return float(self.thresholds.get("fallback_confidence", 0.60))

    def min_rule_score(self):
        return float(self.thresholds.get("min_rule_score", 1.0))

    def mixed_score_ratio(self):
        return float(self.thresholds.get("mixed_score_ratio", 0.35))

    def use_openai_when_uncertain(self):
        return bool(self.fallback_config.get("use_openai_when_uncertain", True))

    def use_openai_for_short_text(self):
        return bool(self.fallback_config.get("use_openai_for_short_text", False))

    def rule_vote(self, text):
        lowered = text.lower()
        scores = {language: 0.0 for language in self.base_languages}
        matched_terms = {language: {} for language in self.base_languages}

        for language in self.base_languages:
            for term in self.language_terms.get(language, []):
                match_count = count_term_matches(lowered, term)
                if not match_count:
                    continue
                shared_count = max(1, self.term_language_counts.get(term, 1))
                weighted_count = min(match_count, 3)
                score = (term_weight(term) / shared_count) * weighted_count
                scores[language] += score
                matched_terms[language][term] = match_count

        scores = {
            language: round(score, 3)
            for language, score in scores.items()
            if score > 0
        }
        label, confidence = self.decide_from_scores(scores)
        return {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "matched_terms": {
                language: dict(list(terms.items())[:12])
                for language, terms in matched_terms.items()
                if terms
            },
        }

    def decide_from_scores(self, scores):
        if not scores:
            return "unknown", 0.0

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_language, top_score = sorted_scores[0]
        if top_score < self.min_rule_score():
            return "unknown", round(min(0.49, top_score / self.min_rule_score() * 0.49), 3)

        eligible = [
            language
            for language, score in sorted_scores
            if score >= self.min_rule_score()
            and score / top_score >= self.mixed_score_ratio()
        ]

        total_score = sum(scores.values())
        if len(eligible) >= 2:
            label = mixed_label(eligible, self.target_label_set)
            eligible_score = sum(scores[language] for language in eligible)
            confidence = 0.50 + min(0.40, eligible_score / (total_score + 2.0) * 0.40)
            return label, round(confidence, 3)

        confidence = 0.45 + min(0.50, top_score / (top_score + 3.0) * 0.50)
        return top_language, round(confidence, 3)

    def langdetect_vote(self, text):
        try:
            from langdetect import DetectorFactory, detect_langs
        except ModuleNotFoundError:
            return {
                "detector": "langdetect",
                "label": "unavailable",
                "confidence": 0.0,
                "status": "langdetect_not_installed",
            }

        try:
            DetectorFactory.seed = 0
            candidates = detect_langs(text)
        except Exception as exc:
            return {
                "detector": "langdetect",
                "label": "unknown",
                "confidence": 0.0,
                "status": f"error: {exc.__class__.__name__}",
            }

        alternatives = []
        for candidate in candidates:
            label = LANGUAGE_CODE_MAP.get(candidate.lang, "unknown")
            alternatives.append(
                {
                    "code": candidate.lang,
                    "label": label,
                    "confidence": round(float(candidate.prob), 3),
                }
            )

        if not alternatives:
            return {
                "detector": "langdetect",
                "label": "unknown",
                "confidence": 0.0,
                "alternatives": [],
            }

        best = alternatives[0]
        return {
            "detector": "langdetect",
            "label": best["label"],
            "confidence": best["confidence"],
            "alternatives": alternatives[:3],
        }

    def fasttext_model_path(self):
        configured_path = self.detector_config.get("fasttext_model_path")
        if not configured_path:
            return None
        path = Path(configured_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def load_fasttext_model(self):
        if self.fasttext_status:
            return self.fasttext_model

        path = self.fasttext_model_path()
        if path is None:
            self.fasttext_status = "fasttext_model_not_configured"
            return None
        if not path.exists():
            self.fasttext_status = "fasttext_model_not_found"
            return None

        try:
            import fasttext
        except ModuleNotFoundError:
            self.fasttext_status = "fasttext_not_installed"
            return None

        try:
            self.fasttext_model = fasttext.load_model(str(path))
        except Exception as exc:
            self.fasttext_status = f"fasttext_load_error: {exc.__class__.__name__}"
            return None
        self.fasttext_status = "available"
        return self.fasttext_model

    def fasttext_vote(self, text):
        model = self.load_fasttext_model()
        if model is None:
            return {
                "detector": "fasttext",
                "label": "unavailable",
                "confidence": 0.0,
                "status": self.fasttext_status,
                "model_path": str(self.fasttext_model_path() or ""),
            }

        labels, probabilities = model.predict(text.replace("\n", " "), k=3)
        alternatives = []
        for raw_label, probability in zip(labels, probabilities):
            code = str(raw_label).replace("__label__", "")
            alternatives.append(
                {
                    "code": code,
                    "label": LANGUAGE_CODE_MAP.get(code, "unknown"),
                    "confidence": round(float(probability), 3),
                }
            )

        if not alternatives:
            return {
                "detector": "fasttext",
                "label": "unknown",
                "confidence": 0.0,
                "alternatives": [],
            }

        best = alternatives[0]
        return {
            "detector": "fasttext",
            "label": best["label"],
            "confidence": best["confidence"],
            "alternatives": alternatives,
        }

    def usable_external_votes(self, *votes):
        return [
            vote
            for vote in votes
            if vote.get("label") not in ("unavailable", "unknown")
        ]

    def local_decision(self, rules, langdetect, fasttext):
        rules_label = rules["label"]
        rules_confidence = float(rules["confidence"])
        external_votes = self.usable_external_votes(langdetect, fasttext)

        if rules_label != "unknown":
            if not external_votes:
                return rules_label, rules_confidence, "rules"
            compatible_votes = [
                vote for vote in external_votes if labels_compatible(rules_label, vote["label"])
            ]
            if compatible_votes:
                confidence = max(
                    rules_confidence,
                    max(float(vote.get("confidence") or 0.0) for vote in compatible_votes),
                )
                return rules_label, round(min(0.95, confidence), 3), "rules_external_agree"
            conflicting_strong_votes = [
                vote
                for vote in external_votes
                if float(vote.get("confidence") or 0.0) >= self.strong_confidence()
            ]
            if (
                rules_confidence >= self.strong_confidence()
                and not conflicting_strong_votes
            ):
                return rules_label, rules_confidence, "rules_override"
            disagreement_confidence = min(
                [rules_confidence]
                + [float(vote.get("confidence") or 0.0) for vote in external_votes]
                + [0.55]
            )
            return "unknown", round(disagreement_confidence, 3), "detector_disagreement"

        strong_votes = [
            vote
            for vote in external_votes
            if float(vote.get("confidence") or 0.0) >= self.strong_confidence()
        ]
        if strong_votes:
            best_vote = max(
                strong_votes, key=lambda vote: float(vote.get("confidence") or 0.0)
            )
            incompatible_votes = [
                vote
                for vote in strong_votes
                if not labels_compatible(best_vote["label"], vote["label"])
            ]
            if incompatible_votes:
                return "unknown", 0.55, "detector_disagreement"
            return (
                best_vote["label"],
                round(float(best_vote.get("confidence") or 0.0), 3),
                best_vote.get("detector", "external_detector"),
            )

        best_confidence = max(
            [rules_confidence]
            + [min(float(vote.get("confidence") or 0.0), 0.55) for vote in external_votes]
        )
        return "unknown", best_confidence, "low_confidence"

    def fallback_needed(self, label, confidence, reason):
        if label == "too_short":
            return self.use_openai_for_short_text()
        if not self.use_openai_when_uncertain():
            return False
        if label == "unknown":
            return True
        if reason == "detector_disagreement":
            return True
        return confidence < self.fallback_confidence()

    def normalize_label(self, label):
        if label in self.target_label_set:
            return label
        if label.startswith("mixed_"):
            languages = label_languages(label)
            return mixed_label(languages, self.target_label_set) if languages else "mixed_other"
        return "unknown"

    def component_languages(self, label, votes):
        if label == "mixed_multiple":
            fallback_languages = votes.get("openai_fallback", {}).get("languages")
            if fallback_languages:
                return ordered_languages(fallback_languages)
            return languages_from_scores(
                votes.get("rules", {}).get("scores") or {},
                self.min_rule_score(),
                self.mixed_score_ratio(),
            )

        languages = label_languages(label)
        if languages:
            return ordered_languages(languages)

        fallback_languages = votes.get("openai_fallback", {}).get("languages")
        if fallback_languages:
            return ordered_languages(fallback_languages)
        return []

    def detect(self, text):
        text = normalize_text(text)
        votes = {
            "length": {"chars": len(text), "min_chars": self.min_chars()},
        }

        if len(text) < self.min_chars():
            label = "too_short"
            confidence = 1.0
            votes["final"] = {
                "label": label,
                "confidence": confidence,
                "languages": [],
                "source": "length_rule",
            }
            return {
                "label": label,
                "languages": [],
                "confidence": confidence,
                "votes": votes,
                "used_openai_fallback": False,
                "fallback_needed": self.use_openai_for_short_text(),
                "fallback_skipped_reason": None,
            }

        rules = self.rule_vote(text)
        langdetect = self.langdetect_vote(text)
        fasttext = self.fasttext_vote(text)
        votes["rules"] = rules
        votes["langdetect"] = langdetect
        votes["fasttext"] = fasttext

        label, confidence, source = self.local_decision(rules, langdetect, fasttext)
        label = self.normalize_label(label)
        confidence = round(float(confidence), 3)
        votes["local"] = {
            "label": label,
            "confidence": confidence,
            "source": source,
        }
        needs_fallback = self.fallback_needed(label, confidence, source)
        skipped_reason = None
        used_fallback = False

        if needs_fallback:
            if self.openai_fallback.is_available():
                fallback = self.openai_fallback.detect(text, self.target_labels)
                fallback_label = self.normalize_label(fallback.get("label", "unknown"))
                fallback_confidence = round(float(fallback.get("confidence") or 0.0), 3)
                votes["openai_fallback"] = {
                    "label": fallback_label,
                    "languages": ordered_languages(fallback.get("languages") or []),
                    "confidence": fallback_confidence,
                    "reason_short": fallback.get("reason_short"),
                    "model_name": fallback.get("model_name"),
                    "usage": fallback.get("usage") or {},
                }
                label = fallback_label
                confidence = fallback_confidence
                source = "openai_fallback"
                used_fallback = True
            else:
                skipped_reason = "missing_openai_api_key"
                votes["openai_fallback"] = {
                    "label": "skipped",
                    "confidence": 0.0,
                    "reason": skipped_reason,
                }

        languages = self.component_languages(label, votes)
        votes["final"] = {
            "label": label,
            "languages": languages,
            "confidence": confidence,
            "source": source,
        }
        return {
            "label": label,
            "languages": languages,
            "confidence": confidence,
            "votes": votes,
            "used_openai_fallback": used_fallback,
            "fallback_needed": needs_fallback,
            "fallback_skipped_reason": skipped_reason,
        }
