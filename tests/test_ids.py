from etl_pipeline.common.ids import extraction_key, row_key, row_key_from_record


def test_extraction_key_is_stable() -> None:
    assert extraction_key(" reddit ", " abc123 ") == "reddit|abc123"


def test_row_key_is_stable() -> None:
    assert row_key("youtube", "comment", "c1", "v1") == "youtube|comment|c1|v1"


def test_row_key_from_record() -> None:
    assert (
        row_key_from_record({"platform": "twitter", "category": "post", "id": "1", "associated_id": "1"})
        == "twitter|post|1|1"
    )

