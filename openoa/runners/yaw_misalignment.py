"""Headless runner for the :py:class:`~openoa.analysis.yaw_misalignment.StaticYawMisalignment`
analysis, used by ``python -m openoa.cli run yaw_misalignment --config <yaml>``.

Configuration file layout::

    seed: 42
    plant:
      loader: examples.project_ENGIE:prepare
      loader_kwargs: {path: ../data/la_haute_borne}
    analysis:                        # constructor kwargs of `StaticYawMisalignment`
      UQ: false
      num_sim: 1
      ws_bins: [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      use_power_coeff: true
      run:                           # optional overrides passed to `StaticYawMisalignment.run`
        min_vane_bin_count: 50
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import attrs
from attrs import field, define

from openoa.schema import FromDictMixin
from openoa.runners import RunnerConfig, write_json
from openoa.analysis.yaw_misalignment import StaticYawMisalignment

ANALYSIS_TYPE = "StaticYawMisalignment"
RESULTS_FILE = "yaw_misalignment_results.json"


def _to_float_or_bounds(
    value: float | list[float] | tuple[float, float],
) -> float | tuple[float, float]:
    """Converts a YAML scalar or two-element list to the ``float | tuple[float, float]`` input
    expected by the UQ-enabled analysis parameters."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a scalar or a pair of bounds, but received: {value}")
        return (float(value[0]), float(value[1]))
    return float(value)


def _to_optional_float_or_bounds(
    value: float | list[float] | tuple[float, float] | None,
) -> float | tuple[float, float] | None:
    return None if value is None else _to_float_or_bounds(value)


def _to_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _to_optional_int(value: int | None) -> int | None:
    return None if value is None else int(value)


def _to_optional_bool(value: bool | None) -> bool | None:
    return None if value is None else bool(value)


def _to_float_list(value: list[float]) -> list[float]:
    return [float(v) for v in value]


def _to_optional_float_list(value: list[float] | None) -> list[float] | None:
    return None if value is None else _to_float_list(value)


@define(auto_attribs=True)
class RunOverrides(FromDictMixin):
    """Optional keyword arguments forwarded to :py:meth:`StaticYawMisalignment.run`; ``None``
    keeps the value set on the analysis object."""

    num_sim: int | None = field(default=None, converter=_to_optional_int)
    ws_bins: list[float] | None = field(default=None, converter=_to_optional_float_list)
    ws_bin_width: float | None = field(default=None, converter=_to_optional_float)
    vane_bin_width: float | None = field(default=None, converter=_to_optional_float)
    min_vane_bin_count: int | None = field(default=None, converter=_to_optional_int)
    max_abs_vane_angle: float | None = field(default=None, converter=_to_optional_float)
    pitch_thresh: float | None = field(default=None, converter=_to_optional_float)
    num_power_bins: int | None = field(default=None, converter=_to_optional_int)
    min_power_filter: float | None = field(default=None, converter=_to_optional_float)
    max_power_filter: float | tuple[float, float] | None = field(
        default=None, converter=_to_optional_float_or_bounds
    )
    power_bin_mad_thresh: float | tuple[float, float] | None = field(
        default=None, converter=_to_optional_float_or_bounds
    )
    use_power_coeff: bool | None = field(default=None, converter=_to_optional_bool)

    def as_kwargs(self) -> dict[str, Any]:
        """The overrides as keyword arguments; values are heterogeneous, hence ``Any``."""
        return attrs.asdict(self, recurse=False)


def _to_run_overrides(value: RunOverrides | dict[str, Any] | None) -> RunOverrides:
    if value is None:
        return RunOverrides()
    if isinstance(value, RunOverrides):
        return value
    known = {a.name for a in attrs.fields(RunOverrides)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"Unknown `analysis.run` keys: {unknown}. Valid keys: {sorted(known)}")
    overrides: RunOverrides = RunOverrides.from_dict(value)
    return overrides


@define(auto_attribs=True)
class AnalysisConfig(FromDictMixin):
    """The ``analysis:`` section: constructor keyword arguments of
    :py:class:`StaticYawMisalignment` plus an optional ``run:`` mapping of overrides."""

    turbine_ids: list[str] | None = field(default=None)
    UQ: bool = field(default=False, converter=bool)
    num_sim: int = field(default=1, converter=int)
    ws_bins: list[float] = field(default=[5.0, 6.0, 7.0, 8.0], converter=_to_float_list)
    ws_bin_width: float = field(default=1.0, converter=float)
    vane_bin_width: float = field(default=1.0, converter=float)
    min_vane_bin_count: int = field(default=100, converter=int)
    max_abs_vane_angle: float = field(default=25.0, converter=float)
    pitch_thresh: float = field(default=0.5, converter=float)
    num_power_bins: int = field(default=25, converter=int)
    min_power_filter: float = field(default=0.01, converter=float)
    max_power_filter: float | tuple[float, float] = field(
        default=(0.92, 0.98), converter=_to_float_or_bounds
    )
    power_bin_mad_thresh: float | tuple[float, float] = field(
        default=(4.0, 10.0), converter=_to_float_or_bounds
    )
    use_power_coeff: bool = field(default=False, converter=bool)
    run: RunOverrides = field(factory=RunOverrides, converter=_to_run_overrides)

    def constructor_kwargs(self) -> dict[str, Any]:
        """Constructor keyword arguments; values are heterogeneous, hence ``Any``."""
        kwargs = attrs.asdict(self, recurse=False)
        kwargs.pop("run")
        return kwargs


def _to_analysis_config(value: AnalysisConfig | dict[str, Any] | None) -> AnalysisConfig:
    if value is None:
        return AnalysisConfig()
    if isinstance(value, AnalysisConfig):
        return value
    known = {a.name for a in attrs.fields(AnalysisConfig)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"Unknown `analysis` keys: {unknown}. Valid keys: {sorted(known)}")
    config: AnalysisConfig = AnalysisConfig.from_dict(value)
    return config


@define(auto_attribs=True)
class Config(RunnerConfig):
    """Full configuration for the static yaw misalignment runner."""

    analysis: AnalysisConfig = field(factory=AnalysisConfig, converter=_to_analysis_config)


def summarize(analysis: StaticYawMisalignment) -> dict[str, Any]:
    """Collects the scalar and array results of a completed analysis into a JSON-serializable
    mapping; the values are heterogeneous (lists, arrays, bools), hence ``Any``."""
    turbine_ids = analysis.turbine_ids if analysis.turbine_ids is not None else []
    summary: dict[str, Any] = {
        "analysis_type": ANALYSIS_TYPE,
        "turbine_ids": list(turbine_ids),
        "UQ": analysis.UQ,
        "num_sim": analysis.num_sim,
        "ws_bins": list(analysis.ws_bins),
        "yaw_misalignment": analysis.yaw_misalignment,
        "yaw_misalignment_ws": analysis.yaw_misalignment_ws,
        "mean_vane_angle": analysis.mean_vane_angle,
        "mean_vane_angle_ws": analysis.mean_vane_angle_ws,
    }
    if analysis.UQ:
        summary.update(
            {
                "yaw_misalignment_avg": analysis.yaw_misalignment_avg,
                "yaw_misalignment_std": analysis.yaw_misalignment_std,
                "yaw_misalignment_95ci": analysis.yaw_misalignment_95ci,
                "yaw_misalignment_avg_ws": analysis.yaw_misalignment_avg_ws,
                "yaw_misalignment_std_ws": analysis.yaw_misalignment_std_ws,
                "yaw_misalignment_95ci_ws": analysis.yaw_misalignment_95ci_ws,
            }
        )
    return summary


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Runs the static yaw misalignment analysis described by ``config`` and writes the results
    to ``output_dir / yaw_misalignment_results.json``."""
    cfg = Config.from_config(config)
    plant = cfg.load_plant()
    if plant.analysis_type is None:
        plant.analysis_type = []
    if ANALYSIS_TYPE not in plant.analysis_type:
        plant.analysis_type.append(ANALYSIS_TYPE)

    analysis = StaticYawMisalignment(plant=plant, **cfg.analysis.constructor_kwargs())
    analysis.run(**cfg.analysis.run.as_kwargs())

    summary = summarize(analysis)
    write_json(summary, Path(output_dir) / RESULTS_FILE)
    return summary
