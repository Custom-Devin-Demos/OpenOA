"""Headless CLI runner for the :py:class:`openoa.analysis.ElectricalLosses` analysis.

Invoke with::

    python -m openoa.cli run electrical_losses --config <config.yaml> [--output-dir DIR]
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import attrs
import numpy as np
from attrs import field, define

from openoa.schema import FromDictMixin
from openoa.runners import RunnerConfig, write_json, seed_everything
from openoa.analysis.electrical_losses import (
    DEFAULT_UQ,
    DEFAULT_NUM_SIM,
    DEFAULT_UNCERTAINTY_METER,
    DEFAULT_UNCERTAINTY_SCADA,
    DEFAULT_UNCERTAINTY_CORRECTION_THRESHOLD,
    ElectricalLosses,
)
from openoa.analysis._analysis_validators import validate_UQ_input, validate_half_closed_0_1_right

ANALYSIS_TYPE = "ElectricalLosses"
RESULTS_FILENAME = "electrical_losses_results.json"


def _to_threshold(
    value: float | int | list[float] | tuple[float, ...],
) -> tuple[float, float] | float:
    """Converts YAML-loaded thresholds (scalars or 2-element lists) to the analysis input type."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("`uncertainty_correction_threshold` must be a scalar or 2 values.")
        return (float(value[0]), float(value[1]))
    return float(value)


@define(auto_attribs=True)
class ElectricalLossesConfig(FromDictMixin):
    """The ``analysis:`` section of an ``electrical_losses`` runner config, mirroring the
    :py:class:`ElectricalLosses` constructor keyword arguments.
    """

    UQ: bool = field(default=DEFAULT_UQ, validator=attrs.validators.instance_of(bool))
    num_sim: int = field(default=DEFAULT_NUM_SIM, converter=int)
    uncertainty_meter: float = field(
        default=DEFAULT_UNCERTAINTY_METER, converter=float, validator=validate_half_closed_0_1_right
    )
    uncertainty_scada: float = field(
        default=DEFAULT_UNCERTAINTY_SCADA, converter=float, validator=validate_half_closed_0_1_right
    )
    uncertainty_correction_threshold: tuple[float, float] | float = field(
        default=DEFAULT_UNCERTAINTY_CORRECTION_THRESHOLD,
        converter=_to_threshold,
        validator=(validate_UQ_input, validate_half_closed_0_1_right),
    )

    def analysis_kwargs(self) -> dict[str, Any]:
        """Keyword arguments forwarded to :py:class:`ElectricalLosses`."""
        return attrs.asdict(self, recurse=False)


def _to_analysis_config(value: ElectricalLossesConfig | dict[str, Any]) -> ElectricalLossesConfig:
    if isinstance(value, ElectricalLossesConfig):
        return value
    analysis_config: ElectricalLossesConfig = ElectricalLossesConfig.from_dict(value)
    return analysis_config


@define(auto_attribs=True)
class ElectricalLossesRunnerConfig(RunnerConfig):
    """Full ``electrical_losses`` runner configuration: the shared runner keys plus ``analysis``."""

    analysis: ElectricalLossesConfig = field(
        factory=ElectricalLossesConfig, converter=_to_analysis_config
    )


Config = ElectricalLossesRunnerConfig


def summarize(analysis: ElectricalLosses) -> dict[str, Any]:
    """Builds the JSON-serializable summary of an :py:class:`ElectricalLosses` run."""
    losses = np.asarray(analysis.electrical_losses, dtype=float).flatten()
    return {
        "method": "electrical_losses",
        "UQ": analysis.UQ,
        "num_sim": analysis.num_sim,
        "uncertainty_meter": analysis.uncertainty_meter,
        "uncertainty_scada": analysis.uncertainty_scada,
        "uncertainty_correction_threshold": analysis.uncertainty_correction_threshold,
        "monthly_meter": analysis.monthly_meter,
        "electrical_losses_mean": float(np.mean(losses)),
        "electrical_losses_std": float(np.std(losses)),
        "electrical_losses": losses,
        "total_turbine_energy": float(analysis.total_turbine_energy),
        "total_meter_energy": float(analysis.total_meter_energy),
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Runs the electrical losses analysis from a validated config and writes the results JSON.

    Args:
        config (dict[str, Any]): The loaded YAML configuration, including ``config_dir``.
        output_dir (Path): Directory in which ``electrical_losses_results.json`` is written.

    Returns:
        dict[str, Any]: The same summary that was written to disk.
    """
    cfg = ElectricalLossesRunnerConfig.from_config(config)
    seed_everything(cfg.seed)

    plant = cfg.load_plant()
    if plant.analysis_type is None:
        plant.analysis_type = [ANALYSIS_TYPE]
    elif ANALYSIS_TYPE not in plant.analysis_type:
        plant.analysis_type.append(ANALYSIS_TYPE)
    plant.validate()

    analysis = ElectricalLosses(plant, **cfg.analysis.analysis_kwargs())
    analysis.run()

    results = summarize(analysis)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(results, output_dir / RESULTS_FILENAME)
    return results
