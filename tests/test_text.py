from etl_pipeline.common.text import clean_text, normalize_for_matching


def test_clean_text_normalizes_whitespace_and_control_chars() -> None:
    assert clean_text("  AI\x00\n healthcare\t\ttext  ") == "AI healthcare text"


def test_clean_text_can_remove_urls() -> None:
    assert clean_text("see https://example.com now", remove_urls=True) == "see now"


def test_normalize_for_matching_casefolds() -> None:
    assert normalize_for_matching("Artificial Intelligence") == "artificial intelligence"

