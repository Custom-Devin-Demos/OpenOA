"""Headless runner for the :py:class:`~openoa.analysis.wake_losses.WakeLosses` method.

Configuration file layout (see :py:mod:`openoa.runners` for the shared ``plant``/``seed`` keys)::

    seed: 42
    plant:
      loader: examples.project_ENGIE:prepare
      loader_kwargs: {path: ../data/la_haute_borne}
    analysis:
      wind_direction_data_type: scada     # "scada" -> "WakeLosses-scada", "tower" -> "WakeLosses-tower"
      wind_direction_asset_ids: [R80711, R80721, R80736]
      wind_direction_offset: 15.85        # optional northing calibration added to the direction column
      end_date: "2015-11-25 00:00"
      UQ: false
      run:                                # keyword arguments of `WakeLosses.run()`
        num_years_LT: 20
        freestream_sector_width: 90.0
        wind_bin_mad_thresh: 7.0
        no_wakes_ws_thresh_LT_corr: 15.0

Any ``analysis`` key that is omitted (``null``) falls back to the ``WakeLosses`` default.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import attrs
import numpy as np
import pandas as pd
from attrs import field, define

from openoa.plant import PlantData
from openoa.schema import FromDictMixin
from openoa.runners import RunnerConfig, write_json
from openoa.analysis.wake_losses import WakeLosses

METHOD = "wake_losses"
ANALYSIS_TYPES = {"scada": "WakeLosses-scada", "tower": "WakeLosses-tower"}

Bounds = float | tuple[float, float]
IntBounds = int | tuple[int, int]


def _to_bounds(value: float | list[float] | tuple[float, float] | None) -> Bounds | None:
    """Converts YAML lists to the ``(lower, upper)`` tuples expected by the UQ validators."""
    if value is None or isinstance(value, (int, float)):
        return value
    if len(value) != 2:
        raise ValueError(f"UQ bounds must have exactly two values, but received: {value}.")
    return (float(value[0]), float(value[1]))


def _to_int_bounds(value: int | list[int] | tuple[int, int] | None) -> IntBounds | None:
    """Converts YAML lists to the ``(lower, upper)`` integer tuples expected by the UQ validators."""
    if value is None or isinstance(value, int):
        return value
    if len(value) != 2:
        raise ValueError(f"UQ bounds must have exactly two values, but received: {value}.")
    return (int(value[0]), int(value[1]))


def _to_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """Converts date-like strings to a ``Timestamp``, leaving ``None`` untouched."""
    return None if value is None else pd.Timestamp(value)


@define(auto_attribs=True)
class WakeLossesRunConfig(FromDictMixin):
    """The keyword arguments of :py:meth:`WakeLosses.run`; ``None`` keeps the constructor value."""

    num_sim: int | None = field(default=None)
    reanalysis_products: list[str] | None = field(default=None)
    wd_bin_width: float | None = field(default=None)
    freestream_sector_width: Bounds | None = field(default=None, converter=_to_bounds)
    freestream_power_method: str | None = field(default=None)
    freestream_wind_speed_method: str | None = field(default=None)
    correct_for_derating: bool | None = field(default=None)
    derating_filter_wind_speed_start: Bounds | None = field(default=None, converter=_to_bounds)
    max_power_filter: Bounds | None = field(default=None, converter=_to_bounds)
    wind_bin_mad_thresh: Bounds | None = field(default=None, converter=_to_bounds)
    correct_for_ws_heterogeneity: bool | None = field(default=None)
    ws_speedup_factor_map: str | None = field(default=None)
    wd_bin_width_LT_corr: float | None = field(default=None)
    ws_bin_width_LT_corr: float | None = field(default=None)
    num_years_LT: IntBounds | None = field(default=None, converter=_to_int_bounds)
    assume_no_wakes_high_ws_LT_corr: bool | None = field(default=None)
    no_wakes_ws_thresh_LT_corr: float | None = field(default=None)
    min_ws_bin_lin_reg: float | None = field(default=None)
    bin_count_thresh_lin_reg: int | None = field(default=None)


def _to_run_config(value: WakeLossesRunConfig | dict[str, Any] | None) -> WakeLossesRunConfig:
    if isinstance(value, WakeLossesRunConfig):
        return value
    config: WakeLossesRunConfig = WakeLossesRunConfig.from_dict(value or {})
    return config


@define(auto_attribs=True)
class WakeLossesAnalysisConfig(FromDictMixin):
    """The ``analysis:`` section: constructor arguments of :py:class:`WakeLosses`, the optional
    ``wind_direction_offset`` pre-processing step, and the ``run:`` keyword arguments."""

    wind_direction_col: str = field(default="WMET_HorWdDir", converter=str)
    wind_direction_data_type: str = field(
        default="scada", validator=attrs.validators.in_(tuple(ANALYSIS_TYPES))
    )
    wind_direction_asset_ids: list[str] | None = field(default=None)
    wind_direction_offset: float | None = field(default=None)
    UQ: bool = field(default=True, converter=bool)
    num_sim: int = field(default=100, converter=int)
    start_date: pd.Timestamp | None = field(default=None, converter=_to_timestamp)
    end_date: pd.Timestamp | None = field(default=None, converter=_to_timestamp)
    reanalysis_products: list[str] | None = field(default=None)
    end_date_lt: pd.Timestamp | None = field(default=None, converter=_to_timestamp)
    wd_bin_width: float | None = field(default=None)
    freestream_sector_width: Bounds | None = field(default=None, converter=_to_bounds)
    freestream_power_method: str | None = field(default=None)
    freestream_wind_speed_method: str | None = field(default=None)
    correct_for_derating: bool | None = field(default=None)
    derating_filter_wind_speed_start: Bounds | None = field(default=None, converter=_to_bounds)
    max_power_filter: Bounds | None = field(default=None, converter=_to_bounds)
    wind_bin_mad_thresh: Bounds | None = field(default=None, converter=_to_bounds)
    correct_for_ws_heterogeneity: bool | None = field(default=None)
    ws_speedup_factor_map: str | None = field(default=None)
    wd_bin_width_LT_corr: float | None = field(default=None)
    ws_bin_width_LT_corr: float | None = field(default=None)
    num_years_LT: IntBounds | None = field(default=None, converter=_to_int_bounds)
    assume_no_wakes_high_ws_LT_corr: bool | None = field(default=None)
    no_wakes_ws_thresh_LT_corr: float | None = field(default=None)
    min_ws_bin_lin_reg: float | None = field(default=None)
    bin_count_thresh_lin_reg: int | None = field(default=None)
    run: WakeLossesRunConfig = field(factory=WakeLossesRunConfig, converter=_to_run_config)

    @property
    def analysis_type(self) -> str:
        return ANALYSIS_TYPES[self.wind_direction_data_type]

    def constructor_kwargs(self) -> dict[str, Any]:
        """The non-``None`` :py:class:`WakeLosses` constructor arguments; the values are the
        heterogeneous field types, hence ``Any``."""
        skip = {"wind_direction_offset", "run", "ws_speedup_factor_map"}
        kwargs = {
            a.name: getattr(self, a.name)
            for a in attrs.fields(WakeLossesAnalysisConfig)
            if a.name not in skip and getattr(self, a.name) is not None
        }
        return kwargs

    def run_kwargs(self, config_dir: Path) -> dict[str, Any]:
        """The :py:meth:`WakeLosses.run` keyword arguments, resolving ``ws_speedup_factor_map``
        against the configuration directory; the values are heterogeneous, hence ``Any``."""
        kwargs: dict[str, Any] = attrs.asdict(self.run, recurse=False)
        kwargs["ws_speedup_factor_map"] = _resolve_map(
            kwargs["ws_speedup_factor_map"] or self.ws_speedup_factor_map, config_dir
        )
        return kwargs


def _resolve_map(path: str | None, config_dir: Path) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else (config_dir / p).resolve())


def _to_analysis_config(
    value: WakeLossesAnalysisConfig | dict[str, Any] | None,
) -> WakeLossesAnalysisConfig:
    if isinstance(value, WakeLossesAnalysisConfig):
        return value
    config: WakeLossesAnalysisConfig = WakeLossesAnalysisConfig.from_dict(value or {})
    return config


@define(auto_attribs=True)
class Config(RunnerConfig):
    """Full configuration of the ``wake_losses`` runner."""

    analysis: WakeLossesAnalysisConfig = field(
        factory=WakeLossesAnalysisConfig, converter=_to_analysis_config
    )


def prepare_plant(plant: PlantData, analysis: WakeLossesAnalysisConfig) -> PlantData:
    """Appends the wake losses ``analysis_type`` and applies the optional wind direction offset
    (northing calibration) to the direction column of the selected data type."""
    analysis_type = plant.analysis_type or []
    if analysis.analysis_type not in analysis_type:
        analysis_type.append(analysis.analysis_type)
    plant.analysis_type = analysis_type
    plant.validate()

    if analysis.wind_direction_offset is not None:
        data = plant.scada if analysis.wind_direction_data_type == "scada" else plant.tower
        if data is None:
            raise ValueError(
                f"`plant.{analysis.wind_direction_data_type}` is required to apply "
                "`wind_direction_offset`."
            )
        col = analysis.wind_direction_col
        data[col] = (data[col] + analysis.wind_direction_offset) % 360.0
    return plant


def summarize(analysis: WakeLosses) -> dict[str, Any]:
    """Collects the scalar and array results of a completed ``WakeLosses`` analysis."""
    results: dict[str, Any] = {
        "method": METHOD,
        "analysis_type": ANALYSIS_TYPES[analysis.wind_direction_data_type],
        "UQ": analysis.UQ,
        "num_sim": analysis.num_sim if analysis.UQ else 1,
        "turbine_ids": list(analysis.turbine_ids),
        "start_date": str(analysis.start_date),
        "end_date": str(analysis.end_date),
        "end_date_lt": str(analysis.end_date_lt),
        "wake_losses_por": analysis.wake_losses_por,
        "wake_losses_lt": analysis.wake_losses_lt,
        "turbine_wake_losses_por": np.asarray(analysis.turbine_wake_losses_por),
        "turbine_wake_losses_lt": np.asarray(analysis.turbine_wake_losses_lt),
        "wake_losses_por_wd": analysis.wake_losses_por_wd,
        "wake_losses_lt_wd": analysis.wake_losses_lt_wd,
        "turbine_wake_losses_por_wd": analysis.turbine_wake_losses_por_wd,
        "turbine_wake_losses_lt_wd": analysis.turbine_wake_losses_lt_wd,
        "wake_losses_por_ws": analysis.wake_losses_por_ws,
        "wake_losses_lt_ws": analysis.wake_losses_lt_ws,
        "turbine_wake_losses_por_ws": analysis.turbine_wake_losses_por_ws,
        "turbine_wake_losses_lt_ws": analysis.turbine_wake_losses_lt_ws,
    }
    if analysis.UQ:
        results.update(
            {
                "wake_losses_por_mean": analysis.wake_losses_por_mean,
                "wake_losses_por_std": analysis.wake_losses_por_std,
                "wake_losses_lt_mean": analysis.wake_losses_lt_mean,
                "wake_losses_lt_std": analysis.wake_losses_lt_std,
                "turbine_wake_losses_por_mean": analysis.turbine_wake_losses_por_mean,
                "turbine_wake_losses_por_std": analysis.turbine_wake_losses_por_std,
                "turbine_wake_losses_lt_mean": analysis.turbine_wake_losses_lt_mean,
                "turbine_wake_losses_lt_std": analysis.turbine_wake_losses_lt_std,
            }
        )
    return results


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Runs the wake losses analysis described by ``config`` and writes
    ``<output_dir>/wake_losses_results.json``."""
    cfg = Config.from_config(config)
    plant = prepare_plant(cfg.load_plant(), cfg.analysis)

    analysis = WakeLosses(plant=plant, **cfg.analysis.constructor_kwargs())
    analysis.run(**cfg.analysis.run_kwargs(cfg.config_dir))

    results = summarize(analysis)
    write_json(results, output_dir / f"{METHOD}_results.json")
    return results
