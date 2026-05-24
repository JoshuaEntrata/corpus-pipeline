import json

from etl_pipeline.preprocessing.transform import transform_extraction_rows


def test_transform_explodes_comments_and_replies() -> None:
    rows, stats = transform_extraction_rows(
        [
            {
                "platform": "reddit",
                "collection_method": "targeted_id",
                "id": "p1",
                "title": "AI health",
                "text": "Main body",
                "url": "https://example.test/p1",
                "comments_json": json.dumps(
                    [
                        {
                            "id": "c1",
                            "text": "comment text",
                            "url": "https://example.test/c1",
                            "replies": [{"id": "r1", "text": "reply text", "url": "https://example.test/r1"}],
                        }
                    ]
                ),
            }
        ]
    )

    assert [row["category"] for row in rows] == ["post", "comment", "reply"]
    assert rows[0]["associated_id"] == "p1"
    assert rows[1]["source_url"] == "https://example.test/c1"
    assert stats.comments_exploded == 1
    assert stats.replies_exploded == 1


def test_transform_deduplicates_within_run() -> None:
    rows, stats = transform_extraction_rows(
        [
            {
                "platform": "reddit",
                "collection_method": "targeted_id",
                "id": "p1",
                "text": "one",
                "comments_json": "[]",
            },
            {
                "platform": "reddit",
                "collection_method": "keyword",
                "id": "p1",
                "text": "one duplicate",
                "comments_json": "[]",
            },
        ]
    )

    assert len(rows) == 1
    assert rows[0]["collection_method"] == "targeted_id"
    assert stats.deduplicated_rows == 1

