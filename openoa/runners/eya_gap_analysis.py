"""Headless runner for :py:class:`openoa.analysis.eya_gap_analysis.EYAGapAnalysis`."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import attrs
from attrs import field, define

from openoa.schema import FromDictMixin
from openoa.runners import RunnerConfig, write_json
from openoa.analysis.eya_gap_analysis import (
    OAResults,
    EYAEstimate,
    EYAGapAnalysis,
    _to_oa_results,
    _to_eya_estimate,
)

METHOD = "eya_gap_analysis"


@define(auto_attribs=True)
class AnalysisSpec(FromDictMixin):
    """Configuration for the EYA gap analysis."""

    eya_estimates: EYAEstimate = field(converter=_to_eya_estimate)
    oa_results: OAResults = field(converter=_to_oa_results)


def _to_analysis_spec(value: AnalysisSpec | dict[str, Any]) -> AnalysisSpec:
    if isinstance(value, AnalysisSpec):
        return value
    known = {attribute.name for attribute in attrs.fields(AnalysisSpec) if attribute.init}
    unknown = set(value).difference(known)
    if unknown:
        raise ValueError(f"Unknown analysis keys: {sorted(unknown)}")
    return AnalysisSpec.from_dict(value)


@define(auto_attribs=True, kw_only=True)
class Config(RunnerConfig):
    """Runner configuration for the EYA gap analysis."""

    analysis: AnalysisSpec = field(converter=_to_analysis_spec)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run EYA gap analysis and write a JSON result."""
    cfg = Config.from_config(config)
    plant = cfg.load_plant()
    analysis = EYAGapAnalysis(
        plant=plant,
        eya_estimates=cfg.analysis.eya_estimates,
        oa_results=cfg.analysis.oa_results,
    )
    analysis.run()
    eya_aep, tie_diff, avail_diff, elec_diff, unaccounted = analysis.compiled_data
    results: dict[str, Any] = {
        "eya_aep_GWh": eya_aep,
        "turbine_ideal_energy_diff_GWh": tie_diff,
        "availability_losses_diff_GWh": avail_diff,
        "electrical_losses_diff_GWh": elec_diff,
        "unaccounted_diff_GWh": unaccounted,
        "oa_aep_GWh": analysis.oa_results.aep,
        "compiled_data": list(analysis.compiled_data),
        "eya_estimates": attrs.asdict(analysis.eya_estimates),
        "oa_results": attrs.asdict(analysis.oa_results),
    }
    write_json(results, output_dir / f"{METHOD}_results.json")
    return results
