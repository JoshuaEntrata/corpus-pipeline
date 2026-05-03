import re
from pathlib import Path


AI_TERMS = [
    "ai",
    "artificial intelligence",
    "openai",
    "chatgpt",
    "gpt",
    "gpt-4",
    "gpt-4o",
    "llm",
    "large language model",
    "generative ai",
    "anthropic",
    "claude",
    "gemini",
    "deepmind",
    "llama",
    "copilot",
    "openevidence",
    "open evidence",
    "med-palm",
    "medpalm",
    "medical ai",
    "clinical ai",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "nlp",
    "algorithm",
    "model",
    "chatbot",
    "virtual assistant",
    "decision support",
    "clinical decision support",
    "automation",
    "automated",
    "ai scribe",
    "ambient scribe",
    "speech recognition",
    "computer vision",
    "predictive analytics",
    "predictive model",
    "modelo",
    "teknolohiya",
    "awtomasyon",
    "automatiko",
]

HEALTH_TERMS = [
    "health",
    "healthcare",
    "medical",
    "medicine",
    "doctor",
    "nurse",
    "hospital",
    "clinic",
    "patient",
    "diagnosis",
    "treatment",
    "symptoms",
    "disease",
    "public health",
    "mental health",
    "radiology",
    "pharmacy",
    "surgery",
    "clinical",
    "health records",
    "electronic medical record",
    "electronic health record",
    "medical record",
    "emr",
    "ehr",
    "telehealth",
    "telemedicine",
    "triage",
    "emergency room",
    "er",
    "icu",
    "laboratory",
    "lab results",
    "blood test",
    "vital signs",
    "blood pressure",
    "glucose",
    "insulin",
    "vaccine",
    "vaccination",
    "infection",
    "sepsis",
    "cancer",
    "tumor",
    "oncology",
    "chemotherapy",
    "radiotherapy",
    "biopsy",
    "screening",
    "diabetes",
    "hypertension",
    "heart disease",
    "stroke",
    "asthma",
    "copd",
    "kidney disease",
    "alzheimer",
    "dementia",
    "depression",
    "anxiety",
    "pregnancy",
    "maternal health",
    "cardiology",
    "neurology",
    "pathology",
    "dermatology",
    "kalusugan",
    "pangkalusugan",
    "doktor",
    "duktor",
    "ospital",
    "pasyente",
    "sakit",
    "sintomas",
    "gamot",
    "lunas",
    "paggamot",
    "nars",
    "klinika",
    "konsultasyon",
    "bakuna",
    "impeksyon",
    "kanser",
    "diyabetis",
    "altapresyon",
    "alta presyon",
    "presyon",
    "pagbubuntis",
    "kahimsog",
    "panglawas",
    "tambal",
    "pagtambal",
    "hilanat",
    "kanser",
    "diabetes",
    "bakuna",
    "salun-at",
    "pasiente",
    "agas",
    "panangagas",
    "balatian",
    "bulong",
    "pagbulong",
    "bakuna",
    "kanser",
    "diabetes",
    "pagbusong",
]


def parse_keyword_config(path):
    path = Path(path)
    if not path.exists():
        return AI_TERMS, HEALTH_TERMS

    terms = {"ai_terms": [], "health_terms": []}
    current_group = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped in ("ai_terms:", "health_terms:"):
                current_group = stripped[:-1]
                continue
            if current_group and stripped.startswith("- "):
                terms[current_group].append(stripped[2:].strip().lower())

    return terms["ai_terms"] or AI_TERMS, terms["health_terms"] or HEALTH_TERMS


def contains_term(text, terms):
    text = (text or "").lower()
    return any(
        re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text)
        for term in terms
    )


def prefilter_text(text, is_empty=False, is_too_short=False, ai_terms=None, health_terms=None):
    ai_terms = ai_terms or AI_TERMS
    health_terms = health_terms or HEALTH_TERMS
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
