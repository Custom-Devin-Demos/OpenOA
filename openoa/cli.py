"""Headless command line interface for running OpenOA analysis methods unattended.

Usage::

    openoa run <method> --config <path/to/config.yml> [--output-dir <dir>]
    python -m openoa.cli run <method> --config <path/to/config.yml>

Each method is implemented by a runner module in :py:mod:`openoa.runners` that exposes a
``run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]`` function and a ``Config``
class (an ``attrs`` class built on :py:class:`openoa.schema.FromDictMixin`) describing and
validating the per-method configuration schema. Runners are imported lazily so that the CLI can
list all methods even when an individual runner is not yet available.
"""

from __future__ import annotations

import sys
import json
import argparse
import importlib
from typing import Any, Protocol
from pathlib import Path

from openoa.runners import load_config


class Runner(Protocol):
    """Interface every ``openoa.runners.<method>`` module must satisfy."""

    def run(self, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Runs the analysis and returns a JSON-serializable summary of the results."""


METHODS: dict[str, str] = {
    "aep": "openoa.runners.aep",
    "electrical_losses": "openoa.runners.electrical_losses",
    "eya_gap_analysis": "openoa.runners.eya_gap_analysis",
    "turbine_long_term_gross_energy": "openoa.runners.turbine_long_term_gross_energy",
    "wake_losses": "openoa.runners.wake_losses",
    "yaw_misalignment": "openoa.runners.yaw_misalignment",
}


def get_runner(method: str) -> Runner:
    """Imports and returns the runner module for ``method``.

    Raises:
        ValueError: If ``method`` is not a known analysis method.
        NotImplementedError: If the runner module for ``method`` is not available.
    """
    try:
        module_name = METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown method '{method}'. Available methods: {', '.join(METHODS)}"
        ) from None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            raise NotImplementedError(f"No runner is available for '{method}' yet.") from e
        raise
    if not callable(getattr(module, "run", None)):
        raise NotImplementedError(f"Runner module '{module_name}' does not define run().")
    runner: Runner = module
    return runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openoa", description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an analysis method from a YAML configuration.")
    run.add_argument("method", choices=sorted(METHODS), help="The analysis method to run.")
    run.add_argument(
        "--config", required=True, type=Path, help="Path to the YAML configuration file."
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results. Defaults to `output_dir` in the config, then the CWD.",
    )

    subparsers.add_parser("list", help="List the available analysis methods.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for method in sorted(METHODS):
            print(method)
        return 0

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = get_runner(args.method)
    results = runner.run(config, output_dir)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
