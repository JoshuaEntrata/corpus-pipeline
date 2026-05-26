from pathlib import Path

import pytest

from etl_pipeline.common.io import read_csv
from etl_pipeline.extraction import manager


class FakeExtractor:
    def extract_by_keyword(self, **_kwargs):
        return [
            {
                "platform": "reddit",
                "collection_method": "keyword",
                "id": "p1",
                "text": "first row",
            },
            {
                "platform": "reddit",
                "collection_method": "keyword",
                "id": "p2",
                "text": "second row",
            },
        ]

    def extract_comments(self, source_id, _limit=None):
        if source_id == "p2":
            raise KeyboardInterrupt
        return []


def test_extraction_persists_rows_before_mid_run_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "paths": {
            "local_data_dir": str(tmp_path / "local_data"),
            "master_dir": str(tmp_path / "local_data" / "master"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "stages": {"extraction": {"skip_seen_ids": True, "platforms": {"reddit": True}}},
    }
    monkeypatch.setattr(manager, "EXTRACTOR_REGISTRY", {"reddit": FakeExtractor})
    monkeypatch.setattr(
        manager,
        "load_input_jobs",
        lambda _inputs_dir: {
            "reddit": {
                "enabled": True,
                "jobs": [
                    {
                        "collection_method": "keyword",
                        "keyword": "ai healthcare",
                        "include_comments": True,
                    }
                ],
            }
        },
    )

    with pytest.raises(KeyboardInterrupt):
        manager.run_extraction(config, inputs_dir=tmp_path / "inputs", run_id="checkpoint-test")

    master_rows = read_csv(tmp_path / "local_data" / "master" / "extraction_raw.csv")
    run_rows = read_csv(tmp_path / "local_data" / "extraction" / "checkpoint-test" / "reddit.csv")
    state_rows = read_csv(tmp_path / "local_data" / "state" / "extraction_seen_ids.csv")

    assert [row["id"] for row in master_rows] == ["p1"]
    assert [row["id"] for row in run_rows] == ["p1"]
    assert [row["id"] for row in state_rows] == ["p1"]
