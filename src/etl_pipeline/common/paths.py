from __future__ import annotations

from pathlib import Path
from typing import Any


def local_data_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("local_data_dir", "local_data"))


def logs_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("logs_dir", "logs"))


def master_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("master_dir", local_data_dir(config) / "master"))


def state_dir(config: dict[str, Any]) -> Path:
    return local_data_dir(config) / "state"


def stage_run_dir(config: dict[str, Any], stage: str, run_id: str) -> Path:
    return local_data_dir(config) / stage / run_id


def ensure_base_dirs(config: dict[str, Any]) -> None:
    for path in (local_data_dir(config), logs_dir(config), master_dir(config), state_dir(config)):
        path.mkdir(parents=True, exist_ok=True)

