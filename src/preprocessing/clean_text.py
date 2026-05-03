import re


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"(?<!\w)@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text):
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text).replace("\r", " ").replace("\n", " ")).strip()


def clean_text(text, replace_urls=True, replace_users=True):
    text = normalize_whitespace(text)
    if not text:
        return ""

    if replace_urls:
        text = URL_RE.sub("<URL>", text)
    if replace_users:
        text = USER_RE.sub("<USER>", text)

    return normalize_whitespace(text)

