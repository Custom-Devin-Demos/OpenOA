"""Deterministic regression of the ``turbine_long_term_gross_energy`` CLI runner against the
results reported by ``examples/03_turbine_ideal_energy.ipynb`` for the La Haute Borne data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy.testing as npt

from openoa.runners import load_config
from openoa.runners.turbine_long_term_gross_energy import METHOD, run

from test.conftest import example_data_path_str  # isort: skip

# Notebook non-UQ case (`ta.plant_gross`): [13513583.66650822, 13569887.73079864] Wh
NOTEBOOK_NON_UQ_BY_PRODUCT = {"era5": 13.51358367, "merra2": 13.56988773}
NOTEBOOK_NON_UQ_MEAN = 13.5417357

# UQ case with the notebook's sampling ranges but `num_sim=10` (seed 42) to keep runtime reasonable
UQ_NUM_SIM = 10
UQ_MEAN = 13.61344094
UQ_STD = 0.28508504


def _config(analysis: dict[str, object]) -> dict[str, object]:
    return {
        "seed": 42,
        "config_dir": str(Path(example_data_path_str).parent),
        "plant": {
            "loader": "examples.project_ENGIE:prepare",
            "loader_kwargs": {"path": example_data_path_str, "use_cleansed": False},
            "analysis_type": "TurbineLongTermGrossEnergy",
        },
        "analysis": analysis,
    }


@pytest.fixture(scope="module")
def non_uq_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output_dir = tmp_path_factory.mktemp("tlge_non_uq")
    return run(
        _config(
            {
                "UQ": False,
                "wind_bin_threshold": 2.0,
                "max_power_filter": 0.9,
                "correction_threshold": 0.9,
                "reanalysis_products": ["era5", "merra2"],
            }
        ),
        output_dir,
    )


def test_non_uq_matches_notebook(non_uq_results: dict[str, object]) -> None:
    assert non_uq_results["UQ"] is False
    assert non_uq_results["num_sim"] == 2
    npt.assert_allclose(
        non_uq_results["plant_gross_energy_gwh_mean"], NOTEBOOK_NON_UQ_MEAN, rtol=0, atol=1e-6
    )
    by_product = non_uq_results["plant_gross_energy_gwh_by_product"]
    assert isinstance(by_product, dict)
    assert sorted(by_product) == sorted(NOTEBOOK_NON_UQ_BY_PRODUCT)
    for product, expected in NOTEBOOK_NON_UQ_BY_PRODUCT.items():
        npt.assert_allclose(by_product[product], expected, rtol=0, atol=1e-6)


def test_uq_is_deterministic_with_seed(tmp_path: Path) -> None:
    analysis = {
        "UQ": True,
        "num_sim": UQ_NUM_SIM,
        "reanalysis_products": ["era5", "merra2"],
        "uncertainty_scada": 0.005,
        "wind_bin_threshold": [1.0, 3.0],
        "max_power_filter": [0.8, 0.9],
        "correction_threshold": [0.85, 0.95],
    }
    first = run(_config(analysis), tmp_path / "run1")
    second = run(_config(analysis), tmp_path / "run2")

    assert first["num_sim"] == UQ_NUM_SIM
    npt.assert_allclose(first["plant_gross_energy_gwh_mean"], UQ_MEAN, rtol=0, atol=1e-6)
    npt.assert_allclose(first["plant_gross_energy_gwh_std"], UQ_STD, rtol=0, atol=1e-6)
    npt.assert_allclose(first["plant_gross_energy_gwh"], second["plant_gross_energy_gwh"], rtol=0)
    assert first == second

    written = json.loads((tmp_path / "run1" / f"{METHOD}_results.json").read_text())
    npt.assert_allclose(written["plant_gross_energy_gwh"], first["plant_gross_energy_gwh"], rtol=0)


def test_example_config_loads() -> None:
    config = load_config(
        Path(__file__).resolve().parents[2] / "examples" / "configs" / f"{METHOD}.yaml"
    )
    assert config["seed"] == 42
    assert config["analysis"]["num_sim"] == UQ_NUM_SIM
