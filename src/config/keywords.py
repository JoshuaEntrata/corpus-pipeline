import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEYWORDS_PATH = PROJECT_ROOT / "configs" / "keywords.yaml"
KEYWORD_GROUPS = ("ai_terms", "health_terms")


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def flatten_terms(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        terms = []
        for child in value.values():
            terms.extend(flatten_terms(child))
        return terms
    if isinstance(value, (list, tuple, set)):
        terms = []
        for child in value:
            terms.extend(flatten_terms(child))
        return terms
    return [str(value)]


def normalize_terms(terms):
    normalized_terms = []
    seen = set()
    for term in terms:
        normalized = str(term).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_terms.append(normalized)
    return normalized_terms


def simple_keyword_yaml_load(text):
    terms = {group: [] for group in KEYWORD_GROUPS}
    current_group = None

    for raw_line in text.splitlines():
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not raw_line.startswith(" ") and stripped.endswith(":"):
            group = stripped[:-1]
            current_group = group if group in terms else None
            continue

        if current_group and stripped.startswith("- "):
            terms[current_group].append(stripped[2:].strip())

    return terms


def read_keyword_config(path):
    path = project_path(path or DEFAULT_KEYWORDS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Keyword config not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return simple_keyword_yaml_load(text)

    return yaml.safe_load(text) or {}


def load_keyword_terms(path=None):
    config = read_keyword_config(path)
    return tuple(
        normalize_terms(flatten_terms(config.get(group)))
        for group in KEYWORD_GROUPS
    )


def keyword_pattern(term):
    term = str(term).strip().lower()
    if term.endswith("*"):
        prefix = term[:-1].strip()
        if not prefix:
            return None
        return rf"(?<!\w){re.escape(prefix)}\w*"

    return rf"(?<!\w){re.escape(term)}(?!\w)"


def contains_term(text, terms):
    text = (text or "").lower()
    for term in terms:
        pattern = keyword_pattern(term)
        if pattern and re.search(pattern, text):
            return True
    return False
