from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from etl_pipeline.classification.run import run_classification
from etl_pipeline.common.config import is_stage_enabled, load_env, load_pipeline_config
from etl_pipeline.common.logging import setup_logging
from etl_pipeline.common.paths import ensure_base_dirs, logs_dir
from etl_pipeline.common.time import make_run_id
from etl_pipeline.extraction.manager import run_extraction
from etl_pipeline.language_detection.run import run_language_detection
from etl_pipeline.preprocessing.run import run_preprocessing


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_env()
    config = load_pipeline_config(args.config)
    ensure_base_dirs(config)
    setup_logging(logs_dir(config))
    run_id = args.run_id or make_run_id(config.get("run", {}).get("run_id_format", "%Y%m%dT%H%M%SZ"))
    inputs_dir = args.inputs or config.get("paths", {}).get("inputs_dir", "inputs")

    if args.command == "run-all":
        summaries = {}
        if is_stage_enabled(config, "extraction"):
            summaries["extraction"] = run_extraction(
                config,
                inputs_dir=inputs_dir,
                run_id=run_id,
                force=args.force,
            )
        if is_stage_enabled(config, "preprocessing"):
            summaries["preprocessing"] = run_preprocessing(config, run_id=run_id, force=args.force, limit=args.limit)
        if is_stage_enabled(config, "classification"):
            summaries["classification"] = run_classification(config, run_id=run_id, force=args.force, limit=args.limit)
        if is_stage_enabled(config, "language_detection"):
            summaries["language_detection"] = run_language_detection(
                config, run_id=run_id, force=args.force, limit=args.limit
            )
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return 0

    command_handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "extract": lambda: run_extraction(
            config,
            inputs_dir=inputs_dir,
            run_id=run_id,
            force=args.force,
        ),
        "preprocess": lambda: run_preprocessing(
            config, run_id=run_id, input_path=args.input, force=args.force, limit=args.limit
        ),
        "classify": lambda: run_classification(
            config, run_id=run_id, input_path=args.input, force=args.force, limit=args.limit
        ),
        "detect-language": lambda: run_language_detection(
            config, run_id=run_id, input_path=args.input, force=args.force, limit=args.limit
        ),
    }
    summary = command_handlers[args.command]()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etl-pipeline")
    parser.add_argument(
        "command",
        choices=["extract", "preprocess", "classify", "detect-language", "run-all"],
    )
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--inputs")
    parser.add_argument("--run-id")
    parser.add_argument("--input")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
