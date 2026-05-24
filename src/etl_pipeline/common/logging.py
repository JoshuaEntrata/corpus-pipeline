from __future__ import annotations

import logging
from pathlib import Path

from etl_pipeline.common.time import utc_now_iso


def setup_logging(logs_dir: str | Path = "logs") -> logging.Logger:
    path = Path(logs_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("etl_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    pipeline_handler = logging.FileHandler(path / "pipeline.log", encoding="utf-8")
    pipeline_handler.setFormatter(formatter)
    logger.addHandler(pipeline_handler)
    return logger


def log_stage_error(
    logs_dir: str | Path,
    *,
    stage: str,
    platform: str = "",
    collection_method: str = "",
    id_or_query: str = "",
    error: BaseException,
) -> None:
    path = Path(logs_dir)
    path.mkdir(parents=True, exist_ok=True)
    line = " | ".join(
        [
            utc_now_iso(),
            stage,
            platform,
            collection_method,
            id_or_query,
            type(error).__name__,
            str(error),
        ]
    )
    with (path / "error.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

