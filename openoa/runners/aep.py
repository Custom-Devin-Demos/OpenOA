"""Headless runner for the Monte Carlo AEP analysis (``python -m openoa.cli run aep``)."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import field, define

from openoa.schema import FromDictMixin
from openoa.runners import RunnerConfig, write_json
from openoa.analysis.aep import MonteCarloAEP

ANALYSIS_TYPE = "MonteCarloAEP"
RESULTS_FILENAME = "aep_results.json"


def _to_optional_list(value: str | list[str] | None) -> list[str] | None:
    if value is None or isinstance(value, list):
        return value
    return [value]


@define(auto_attribs=True)
class AEPRunSettings(FromDictMixin):
    """The ``analysis.run`` section: keyword arguments of :py:meth:`MonteCarloAEP.run`."""

    num_sim: int = field(default=20, converter=int)
    progress_bar: bool = field(default=False, converter=bool)


def _to_run_settings(value: AEPRunSettings | dict[str, Any]) -> AEPRunSettings:
    if isinstance(value, AEPRunSettings):
        return value
    settings: AEPRunSettings = AEPRunSettings.from_dict(value)
    return settings


@define(auto_attribs=True)
class AEPAnalysisConfig(FromDictMixin):
    """The ``analysis`` section: constructor keyword arguments of
    :py:class:`~openoa.analysis.aep.MonteCarloAEP` plus the ``run`` settings. Values that are
    not provided fall back to the analysis class defaults.
    """

    reg_temperature: bool = field(default=False, converter=bool)
    reg_wind_direction: bool = field(default=False, converter=bool)
    reanalysis_products: list[str] | None = field(default=None, converter=_to_optional_list)
    uncertainty_meter: float = field(default=0.005, converter=float)
    uncertainty_losses: float = field(default=0.05, converter=float)
    uncertainty_windiness: tuple[float, float] = field(default=(10.0, 20.0), converter=tuple)
    uncertainty_loss_max: tuple[float, float] = field(default=(10.0, 20.0), converter=tuple)
    outlier_detection: bool = field(default=False, converter=bool)
    uncertainty_outlier: tuple[float, float] = field(default=(1.0, 3.0), converter=tuple)
    uncertainty_nan_energy: float = field(default=0.01, converter=float)
    time_resolution: str = field(default="MS", converter=str)
    end_date_lt: str | None = field(default=None)
    reg_model: str = field(default="lin", converter=str)
    ml_setup_kwargs: dict[str, Any] = field(factory=dict)  # Any: sklearn/pygam hyperparameters
    n_jobs: int | None = field(default=None)
    apply_iav: bool = field(default=True, converter=bool)
    run: AEPRunSettings = field(factory=AEPRunSettings, converter=_to_run_settings)

    def constructor_kwargs(self) -> dict[str, Any]:  # Any: heterogeneous MonteCarloAEP kwargs
        kwargs = {
            "reg_temperature": self.reg_temperature,
            "reg_wind_direction": self.reg_wind_direction,
            "reanalysis_products": self.reanalysis_products,
            "uncertainty_meter": self.uncertainty_meter,
            "uncertainty_losses": self.uncertainty_losses,
            "uncertainty_windiness": self.uncertainty_windiness,
            "uncertainty_loss_max": self.uncertainty_loss_max,
            "outlier_detection": self.outlier_detection,
            "uncertainty_outlier": self.uncertainty_outlier,
            "uncertainty_nan_energy": self.uncertainty_nan_energy,
            "time_resolution": self.time_resolution,
            "end_date_lt": self.end_date_lt,
            "reg_model": self.reg_model,
            "ml_setup_kwargs": self.ml_setup_kwargs,
            "n_jobs": self.n_jobs,
            "apply_iav": self.apply_iav,
        }
        return {k: v for k, v in kwargs.items() if v is not None}


def _to_analysis_config(value: AEPAnalysisConfig | dict[str, Any]) -> AEPAnalysisConfig:
    if isinstance(value, AEPAnalysisConfig):
        return value
    config: AEPAnalysisConfig = AEPAnalysisConfig.from_dict(value)
    return config


@define(auto_attribs=True)
class AEPRunnerConfig(RunnerConfig):
    """Complete configuration for ``python -m openoa.cli run aep``."""

    analysis: AEPAnalysisConfig = field(factory=AEPAnalysisConfig, converter=_to_analysis_config)


def _column_stats(results: pd.DataFrame, column: str) -> dict[str, float]:
    values = results[column].to_numpy(dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize(analysis: MonteCarloAEP) -> dict[str, Any]:  # Any: nested scalars and lists
    """Builds the JSON-serializable summary of a completed Monte Carlo AEP analysis."""
    results = analysis.results
    return {
        "analysis_type": ANALYSIS_TYPE,
        "num_sim": int(len(results)),
        "reanalysis_products": list(analysis.reanalysis_products),
        "time_resolution": analysis.time_resolution,
        "reg_model": analysis.reg_model,
        "period_of_record": {
            "start": str(analysis.start_por),
            "end": str(analysis.end_por),
        },
        "aep_GWh": _column_stats(results, "aep_GWh"),
        "avail_pct": _column_stats(results, "avail_pct"),
        "curt_pct": _column_stats(results, "curt_pct"),
        "lt_por_ratio": _column_stats(results, "lt_por_ratio"),
        "r2": _column_stats(results, "r2"),
        "mse": _column_stats(results, "mse"),
        "iav": _column_stats(results, "iav"),
        "n_points": _column_stats(results, "n_points"),
        "simulations": {col: results[col].tolist() for col in results.columns},
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:  # Any: parsed YAML
    """Runs the Monte Carlo AEP analysis headlessly and writes ``aep_results.json``."""
    runner_config = AEPRunnerConfig.from_config(config)
    plant = runner_config.load_plant()
    existing = plant.analysis_type or []
    if ANALYSIS_TYPE not in existing:
        plant.analysis_type = [*existing, ANALYSIS_TYPE]

    settings = runner_config.analysis
    analysis = MonteCarloAEP(plant, **settings.constructor_kwargs())
    analysis.run(num_sim=settings.run.num_sim, progress_bar=settings.run.progress_bar)

    summary = summarize(analysis)
    write_json(summary, Path(output_dir) / RESULTS_FILENAME)
    return summary
