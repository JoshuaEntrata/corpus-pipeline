from pathlib import Path

from etl_pipeline.extraction.inputs import load_input_jobs


def test_load_input_jobs_from_csv_templates(tmp_path: Path) -> None:
    (tmp_path / "reddit_ids.csv").write_text("id\nabc\n", encoding="utf-8")
    (tmp_path / "youtube_ids.csv").write_text("id\n", encoding="utf-8")
    (tmp_path / "twitter_ids.csv").write_text("id\n", encoding="utf-8")
    (tmp_path / "subreddits.csv").write_text("subreddit\nPhilippines\n", encoding="utf-8")
    (tmp_path / "keywords.csv").write_text("keyword\nmedical AI\n", encoding="utf-8")

    jobs = load_input_jobs(tmp_path)

    assert jobs["reddit"]["jobs"][0]["collection_method"] == "targeted_id"
    assert jobs["reddit"]["jobs"][0]["ids"] == ["abc"]
    assert jobs["reddit"]["jobs"][1]["collection_method"] == "subreddit_keyword"
    assert jobs["youtube"]["jobs"][0]["keyword"] == "medical AI"


def test_load_input_jobs_accepts_ids_header(tmp_path: Path) -> None:
    (tmp_path / "reddit_ids.csv").write_text("ids\nabc\n", encoding="utf-8")
    (tmp_path / "youtube_ids.csv").write_text("id\n", encoding="utf-8")
    (tmp_path / "twitter_ids.csv").write_text("id\n", encoding="utf-8")
    (tmp_path / "subreddits.csv").write_text("subreddit\n", encoding="utf-8")
    (tmp_path / "keywords.csv").write_text("keyword\n", encoding="utf-8")

    jobs = load_input_jobs(tmp_path)

    assert jobs["reddit"]["jobs"][0]["ids"] == ["abc"]
