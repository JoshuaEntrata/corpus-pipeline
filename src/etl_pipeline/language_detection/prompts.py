LANGUAGE_SYSTEM_PROMPT = """You detect language for standalone social-media text that has already been classified as AI-in-healthcare relevant.

Classify only the provided text. Do not infer language from a parent post, video title, thread, URL target, account name, author identity, or missing context.

Target languages:
- english
- tagalog
- cebuano
- ilocano
- hiligaynon

Rules:
- Return one or more target languages actually present in the text.
- Use mixed labels when the text meaningfully contains more than one target language.
- Use at most three languages per row.
- Order mixed labels according to the configured language order: english, tagalog, cebuano, ilocano, hiligaynon.
- Hashtags count as visible text when they contain meaningful words.
- URLs do not count unless the visible URL text itself clearly contains language-bearing words.
- Use out_of_scope when the text is empty, not enough language evidence is visible, or the language is outside the target list.

Return JSON that matches the requested schema."""


LANGUAGE_BATCH_SCHEMA = """Return strict JSON with this shape:
{
  "results": [
    {
      "row_id": "the provided row_id",
      "language_detected": ["english"],
      "language_label": "english"
    }
  ]
}

language_label must be one target language, mixed_<languages_in_order>, or out_of_scope.
Return exactly one result for every input row_id."""


def build_language_prompt(text: str) -> str:
    return build_batch_language_prompt([{"row_id": "row_0", "text": text}])


def build_batch_language_prompt(rows: list[dict[str, str]]) -> str:
    import json

    payload = [{"row_id": row["row_id"], "text": row["text"]} for row in rows]
    return (
        f"{LANGUAGE_SYSTEM_PROMPT}\n\n"
        f"{LANGUAGE_BATCH_SCHEMA}\n\n"
        f"Rows:\n{json.dumps(payload, ensure_ascii=False)}"
    )
