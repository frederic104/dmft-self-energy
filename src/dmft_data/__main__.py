"""One entry point for metadata review, paired preparation, and CPU training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from .learning.data import catalog, load_json, prepare

LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser(
        "catalog", help="Inventory block paths and beta; create a review template"
    )
    listing.add_argument("--source", type=Path, default=Path("data/raw/hdf5"))
    listing.add_argument(
        "--output", type=Path, default=Path("data/metadata/run_review.json")
    )
    for command in ("prepare", "train", "summary", "diagnose"):
        child = sub.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        if command == "diagnose":
            child.add_argument("--definitions", type=Path)
            child.add_argument("--output", type=Path)
            child.add_argument("--raw-run", action="append", default=[])
            child.add_argument("--raw-count", type=int)
    args = parser.parse_args()
    if args.command == "catalog":
        if args.output.exists():
            parser.error(
                "Review output already exists; use a new path to preserve manual edits"
            )
        result = catalog(args.source.resolve(), args.output)
        LOGGER.info("runs=%d failures=%d", len(result["runs"]), len(result["failures"]))
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # Paths are relative to repository_root, resolved relative to the config file.
    root = (
        args.config.resolve().parent / config.get("repository_root", "..")
    ).resolve()
    if args.command == "prepare":
        result = prepare(config, root)
        LOGGER.info(
            "samples=%d groups=%d excluded=%d",
            result["sample_count"],
            result["group_count"],
            len(result["excluded"]),
        )
    elif args.command == "diagnose":
        from .physics import diagnose

        result = diagnose(
            config, root, args.definitions, args.output, args.raw_run, args.raw_count
        )
        LOGGER.info(
            "physics diagnostics saved; unresolved definitions=%d; raw runs=%d",
            len(result["unresolved_definitions"]),
            len(result["raw_runs"]),
        )
    elif args.command == "summary":
        manifest = load_json(root / config["prepared"] / "manifest.json")
        result = {
            "task": manifest["task"],
            "samples": manifest["sample_count"],
            "groups": manifest["group_count"],
            "excluded": len(manifest["excluded"]),
            "exclusion_examples": manifest["excluded"][:3],
        }
        metrics_path = root / config["experiment"] / "metrics.json"
        if metrics_path.is_file():
            metrics = load_json(metrics_path)
            result["selected_model"] = metrics["candidates"][
                metrics["selected_candidate"]
            ]["specification"]
            result["test_rmse"] = [x["errors"]["rmse"] for x in metrics["test"]]
        LOGGER.info("%s", json.dumps(result, indent=2))
    else:
        from .learning.experiment import run_experiment

        result = run_experiment(config, root)
        selected = result["candidates"][result["selected_candidate"]]["specification"]
        LOGGER.info(
            "%s",
            json.dumps(
                {
                    "task": result["task"],
                    "selected_model": selected,
                    "test_rmse": [x["errors"]["rmse"] for x in result["test"]],
                }
            ),
        )


if __name__ == "__main__":
    main()
