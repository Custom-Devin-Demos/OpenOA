"""Deterministic regression tests for the ``electrical_losses`` CLI runner on the bundled
La Haute Borne example data.

Expected values were produced by ``openoa.runners.electrical_losses.run`` with ``seed: 42`` and
confirmed identical across two independent runs.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np
import pytest
import numpy.testing as npt

from openoa.runners.electrical_losses import RESULTS_FILENAME, run

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "configs"


def _base_config() -> dict[str, Any]:
    return {
        "config_dir": str(EXAMPLE_CONFIG),
        "seed": 42,
        "plant": {
            "loader": "examples.project_ENGIE:prepare",
            "loader_kwargs": {"path": "../data/la_haute_borne"},
            "analysis_type": "ElectricalLosses",
        },
    }


@pytest.fixture(scope="module")
def results_no_uq(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    output_dir = tmp_path_factory.mktemp("electrical_losses_no_uq")
    config = _base_config()
    config["analysis"] = {"UQ": False, "uncertainty_correction_threshold": 0.95}
    return run(config, output_dir), output_dir


@pytest.fixture(scope="module")
def results_uq(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    output_dir = tmp_path_factory.mktemp("electrical_losses_uq")
    config = _base_config()
    config["analysis"] = {
        "UQ": True,
        "num_sim": 300,
        "uncertainty_correction_threshold": [0.9, 0.995],
    }
    return run(config, output_dir), output_dir


def test_electrical_losses_without_uq(results_no_uq: tuple[dict[str, Any], Path]) -> None:
    results, _ = results_no_uq
    assert results["UQ"] is False
    assert results["monthly_meter"] is False
    assert results["num_sim"] == 1
    npt.assert_allclose(
        results["electrical_losses_mean"], 0.019994645742960393, rtol=1e-6, atol=1e-8
    )
    npt.assert_allclose(results["electrical_losses_std"], 0.0, rtol=0, atol=1e-12)
    npt.assert_allclose(results["total_turbine_energy"], 23698138.195996672, rtol=1e-6)
    npt.assert_allclose(results["total_meter_energy"], 23224302.318, rtol=1e-6)
    npt.assert_array_equal(np.asarray(results["electrical_losses"]).shape, (1,))


def test_electrical_losses_with_uq(results_uq: tuple[dict[str, Any], Path]) -> None:
    results, _ = results_uq
    assert results["UQ"] is True
    assert results["num_sim"] == 300
    npt.assert_allclose(
        results["electrical_losses_mean"], 0.01989302371299591, rtol=1e-6, atol=1e-8
    )
    npt.assert_allclose(
        results["electrical_losses_std"], 0.006861373869488184, rtol=1e-6, atol=1e-8
    )
    npt.assert_allclose(results["total_turbine_energy"], 23742844.792124648, rtol=1e-6)
    npt.assert_array_equal(np.asarray(results["electrical_losses"]).shape, (300,))
    # The pinned values are consistent with the notebook-scale regression (mean 0.02, std 0.0069)
    npt.assert_almost_equal(results["electrical_losses_mean"], 0.02, decimal=3)
    npt.assert_almost_equal(results["electrical_losses_std"], 0.0069, decimal=3)


def test_results_json_matches_returned_summary(results_uq: tuple[dict[str, Any], Path]) -> None:
    results, output_dir = results_uq
    path = output_dir / RESULTS_FILENAME
    assert path.is_file()
    with path.open() as f:
        written = json.load(f)
    assert written["method"] == "electrical_losses"
    npt.assert_allclose(
        written["electrical_losses_mean"], results["electrical_losses_mean"], rtol=1e-12
    )
    npt.assert_allclose(written["electrical_losses"], results["electrical_losses"], rtol=1e-12)
