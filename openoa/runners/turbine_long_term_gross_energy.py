"""Headless runner for :py:class:`openoa.analysis.TurbineLongTermGrossEnergy`.

Configuration layout (see :py:mod:`openoa.runners` for the shared ``plant``/``seed`` keys)::

    seed: 42
    plant:
      loader: examples.project_ENGIE:prepare
      loader_kwargs: {path: ../data/la_haute_borne}
    analysis:
      UQ: true
      num_sim: 10
      reanalysis_products: [era5, merra2]
      uncertainty_scada: 0.005
      wind_bin_threshold: [1.0, 3.0]
      max_power_filter: [0.8, 0.9]
      correction_threshold: [0.85, 0.95]
"""

from __future__ import annotations

from typing import Any, cast
from pathlib import Path

import attrs
import numpy as np
import pandas as pd
from attrs import field, define

from openoa.schema import FromDictMixin
from openoa.runners import RunnerConfig, write_json
from openoa.analysis.turbine_long_term_gross_energy import TurbineLongTermGrossEnergy

METHOD = "turbine_long_term_gross_energy"
ANALYSIS_TYPE = "TurbineLongTermGrossEnergy"

Threshold = float | tuple[float, float]


def _to_threshold(value: float | int | list[float] | tuple[float, ...]) -> Threshold:
    """Converts YAML scalars/sequences into the ``float | tuple[float, float]`` the analysis
    expects; the analysis validators enforce the UQ-dependent shape."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Threshold ranges must have exactly 2 values, received: {value}")
        return (float(value[0]), float(value[1]))
    return float(value)


def _to_products(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


@define(auto_attribs=True)
class AnalysisConfig(FromDictMixin):
    """The ``analysis:`` section: constructor kwargs of ``TurbineLongTermGrossEnergy`` and the
    kwargs of its ``run()`` method (which share the same names)."""

    UQ: bool = field(default=True, converter=bool)
    num_sim: int = field(default=10, converter=int, validator=attrs.validators.ge(1))
    reanalysis_products: list[str] | None = field(default=None, converter=_to_products)
    uncertainty_scada: float = field(default=0.005, converter=float)
    wind_bin_threshold: Threshold = field(default=(1.0, 3.0), converter=_to_threshold)
    max_power_filter: Threshold = field(default=(0.8, 0.9), converter=_to_threshold)
    correction_threshold: Threshold = field(default=(0.85, 0.95), converter=_to_threshold)

    def constructor_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``TurbineLongTermGrossEnergy(...)``; values are the heterogeneous
        field types above, hence ``Any``."""
        return {
            "UQ": self.UQ,
            "num_sim": self.num_sim,
            "reanalysis_products": self.reanalysis_products,
            "uncertainty_scada": self.uncertainty_scada,
            "wind_bin_threshold": self.wind_bin_threshold,
            "max_power_filter": self.max_power_filter,
            "correction_threshold": self.correction_threshold,
        }


def _to_analysis_config(value: AnalysisConfig | dict[str, Any] | None) -> AnalysisConfig:
    if isinstance(value, AnalysisConfig):
        return value
    config: AnalysisConfig = AnalysisConfig.from_dict({} if value is None else value)
    return config


@define(auto_attribs=True)
class Config(RunnerConfig):
    """Full configuration schema for the ``turbine_long_term_gross_energy`` runner."""

    analysis: AnalysisConfig = field(factory=AnalysisConfig, converter=_to_analysis_config)


def summarize(analysis: TurbineLongTermGrossEnergy) -> dict[str, Any]:
    """Collects the JSON-serializable results; the values are a mix of floats, lists, and nested
    mappings, hence the ``Any`` value type."""
    plant_gross_gwh = np.asarray(analysis.plant_gross, dtype=np.float64).flatten() / 1e6
    turb_mo = analysis.turb_lt_gross.resample("MS").sum()
    turb_mo_index = cast(pd.DatetimeIndex, turb_mo.index)
    turbine_lt_gross_gwh = turb_mo.groupby(turb_mo_index.month).mean().sum(axis=0) / 1e6
    products = list(analysis._inputs["reanalysis_product"])
    summary: dict[str, Any] = {
        "method": METHOD,
        "analysis_type": ANALYSIS_TYPE,
        "UQ": analysis.UQ,
        "num_sim": int(analysis.num_sim),
        "reanalysis_products": list(analysis.reanalysis_products),
        "plant_gross_energy_gwh_mean": float(plant_gross_gwh.mean()),
        "plant_gross_energy_gwh_std": float(plant_gross_gwh.std()),
        "plant_gross_energy_gwh": plant_gross_gwh.tolist(),
        "simulation_reanalysis_products": [str(p) for p in products],
        "last_simulation_turbine_lt_gross_energy_gwh": {
            str(turbine): float(value) for turbine, value in turbine_lt_gross_gwh.items()
        },
    }
    if not analysis.UQ:
        summary["plant_gross_energy_gwh_by_product"] = {
            str(p): float(v) for p, v in zip(products, plant_gross_gwh)
        }
    return summary


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Validates ``config``, runs the long-term gross energy analysis, and writes
    ``<output_dir>/turbine_long_term_gross_energy_results.json``."""
    cfg = Config.from_config(config)
    plant = cfg.load_plant()
    analysis_types = plant.analysis_type or []
    if not {ANALYSIS_TYPE, "all"}.intersection(analysis_types):
        plant.analysis_type = [*analysis_types, ANALYSIS_TYPE]

    analysis = TurbineLongTermGrossEnergy(plant, **cfg.analysis.constructor_kwargs())
    analysis.run(
        num_sim=cfg.analysis.num_sim,
        reanalysis_products=cfg.analysis.reanalysis_products,
        uncertainty_scada=cfg.analysis.uncertainty_scada,
        wind_bin_threshold=cfg.analysis.wind_bin_threshold,
        max_power_filter=cfg.analysis.max_power_filter,
        correction_threshold=cfg.analysis.correction_threshold,
    )

    results = summarize(analysis)
    write_json(results, Path(output_dir) / f"{METHOD}_results.json")
    return results
