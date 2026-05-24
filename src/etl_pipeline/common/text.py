from __future__ import annotations

import re
import unicodedata

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def clean_text(text: object, *, remove_urls: bool = False) -> str:
    if text is None:
        return ""
    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = CONTROL_CHARS_RE.sub(" ", cleaned)
    if remove_urls:
        cleaned = URL_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_for_matching(text: object, *, case_sensitive: bool = False) -> str:
    normalized = clean_text(text)
    if not case_sensitive:
        normalized = normalized.casefold()
    return normalized

