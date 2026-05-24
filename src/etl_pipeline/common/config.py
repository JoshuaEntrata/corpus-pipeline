from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise ConfigError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {yaml_path}")
    return data


def load_env(path: str | Path = ".env") -> None:
    load_dotenv(path, override=False)


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigError(f"Missing required {context} config keys: {', '.join(missing)}")


def load_pipeline_config(path: str | Path = "config/pipeline.yaml") -> dict[str, Any]:
    config = load_yaml(path)
    require_keys(config, ["run", "paths", "stages", "outputs"], "pipeline")
    return config


def is_stage_enabled(config: dict[str, Any], stage: str) -> bool:
    return bool(config.get("stages", {}).get(stage, {}).get("enabled", False))


def should_skip_seen(config: dict[str, Any], stage: str, force: bool = False) -> bool:
    if force:
        return False
    return bool(config.get("stages", {}).get(stage, {}).get("skip_seen_ids", True))

