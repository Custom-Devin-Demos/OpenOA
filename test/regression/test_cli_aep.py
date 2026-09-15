"""Deterministic regression of the Monte Carlo AEP analysis through the CLI runner.

The configuration mirrors ``examples/02a_plant_aep_analysis.ipynb`` (monthly resolution, MERRA-2 and
ERA5, temperature and wind direction regressors) with a small number of simulations, seeded via
``seed_everything``. Expected values were derived from two consecutive runs of the runner, which
produced identical results.
"""

import json
from pathlib import Path

import pytest
import numpy.testing as nptest

from openoa import cli
from openoa.runners import load_config

from test.conftest import example_data_path_str  # isort: skip

NUM_SIM = 20
EXPECTED = {
    "aep_GWh": {"mean": 12.375155355856247, "std": 1.0543114879375586},
    "avail_pct": {"mean": 0.011712643588636884, "std": 0.0005720839830132585},
    "curt_pct": {"mean": 0.0006443345139019457, "std": 4.2395839011473886e-05},
}
RTOL = 1e-6
ATOL = 1e-9


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "aep.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 42",
                "plant:",
                "  loader: examples.project_ENGIE:prepare",
                "  loader_kwargs:",
                f"    path: {example_data_path_str}",
                "    use_cleansed: false",
                "analysis:",
                "  reanalysis_products: [merra2, era5]",
                "  time_resolution: MS",
                "  reg_temperature: true",
                "  reg_wind_direction: true",
                "  reg_model: lin",
                "  run:",
                f"    num_sim: {NUM_SIM}",
                "    progress_bar: false",
                "",
            ]
        )
    )
    return config


def _run(config: Path, output_dir: Path) -> dict:
    assert cli.main(["run", "aep", "--config", str(config), "--output-dir", str(output_dir)]) == 0
    with (output_dir / "aep_results.json").open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cli_results(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    tmp_path = tmp_path_factory.mktemp("aep")
    config = _write_config(tmp_path)
    first = _run(config, tmp_path / "run1")
    second = _run(config, tmp_path / "run2")
    return first, second


def test_config_is_valid(tmp_path: Path):
    config = load_config(_write_config(tmp_path))
    assert config["seed"] == 42
    assert config["analysis"]["run"]["num_sim"] <= 50


def test_long_term_aep(cli_results: tuple[dict, dict]):
    results, _ = cli_results
    assert results["analysis_type"] == "MonteCarloAEP"
    assert results["num_sim"] == NUM_SIM
    assert len(results["simulations"]["aep_GWh"]) == NUM_SIM
    nptest.assert_allclose(
        results["aep_GWh"]["mean"], EXPECTED["aep_GWh"]["mean"], rtol=RTOL, atol=ATOL
    )
    nptest.assert_allclose(
        results["aep_GWh"]["std"], EXPECTED["aep_GWh"]["std"], rtol=RTOL, atol=ATOL
    )


def test_long_term_losses(cli_results: tuple[dict, dict]):
    results, _ = cli_results
    for key in ("avail_pct", "curt_pct"):
        nptest.assert_allclose(results[key]["mean"], EXPECTED[key]["mean"], rtol=RTOL, atol=ATOL)
        nptest.assert_allclose(results[key]["std"], EXPECTED[key]["std"], rtol=RTOL, atol=ATOL)


def test_repeat_run_is_identical(cli_results: tuple[dict, dict]):
    first, second = cli_results
    assert first == second
