import json
from pathlib import Path

import pytest
import numpy.testing as npt

from openoa import cli
from openoa.runners import load_config
from openoa.runners import eya_gap_analysis as runner

CONFIG = Path(__file__).resolve().parents[2] / "examples" / "configs" / "eya_gap_analysis.yaml"
EXPECTED_COMPILED = [
    467.0,
    -41.441648489739976,
    6.594368935819699,
    6.23089978187688,
    9.616379772043388,
]


def test_cli_eya_gap_analysis(tmp_path: Path) -> None:
    results = runner.run(load_config(CONFIG), tmp_path)
    npt.assert_allclose(results["compiled_data"], EXPECTED_COMPILED, rtol=1e-9, atol=1e-9)
    assert results["eya_aep_GWh"] == 467.0
    assert results["oa_aep_GWh"] == 448.0
    for key, expected in zip(
        (
            "turbine_ideal_energy_diff_GWh",
            "availability_losses_diff_GWh",
            "electrical_losses_diff_GWh",
            "unaccounted_diff_GWh",
        ),
        EXPECTED_COMPILED[1:],
    ):
        npt.assert_allclose(results[key], expected, rtol=1e-9, atol=1e-9)
    out = json.loads((tmp_path / "eya_gap_analysis_results.json").read_text())
    npt.assert_allclose(out["compiled_data"], EXPECTED_COMPILED, rtol=1e-9, atol=1e-9)


def test_cli_eya_gap_analysis_main(tmp_path: Path) -> None:
    assert (
        cli.main(
            ["run", "eya_gap_analysis", "--config", str(CONFIG), "--output-dir", str(tmp_path)]
        )
        == 0
    )
    assert (tmp_path / "eya_gap_analysis_results.json").is_file()


def test_cli_eya_gap_analysis_rejects_unknown_analysis_key() -> None:
    cfg = load_config(CONFIG)
    cfg["analysis"]["bogus"] = 1
    with pytest.raises(ValueError, match="bogus"):
        runner.Config.from_config(cfg)
