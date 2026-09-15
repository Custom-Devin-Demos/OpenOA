"""Deterministic regression test for the static yaw misalignment analysis run through the CLI
runner (``python -m openoa.cli run yaw_misalignment``) on the La Haute Borne example data, using the
non-UQ parameters from ``examples/07_static_yaw_misalignment.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.testing as npt

from openoa import cli
from openoa.runners import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples" / "configs" / "yaw_misalignment.yaml"

TURBINE_IDS = ["R80711", "R80721", "R80736", "R80790"]
WS_BINS = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

EXPECTED_YAW_MISALIGNMENT = np.array([1.22161471, 3.19249435, 1.25333118, 2.87197069])
EXPECTED_YAW_MISALIGNMENT_WS = np.array(
    [
        [0.32974873, -0.24664293, -0.56610949, -0.73125953, 0.33614753, 2.41273704, 7.01668163],
        [0.41128653, 1.85770713, 0.79602991, 0.35301727, 4.45674720, 8.09740374, 6.37526869],
        [0.68372166, 0.93489078, 0.14713785, -1.15133488, 0.77473994, 2.82345208, 4.56071085],
        [0.77126696, 1.43002486, 0.38850428, 1.32806772, 4.03322204, 5.71170124, 6.44100776],
    ]
)

# The estimates come from `scipy.optimize.curve_fit`, so allow for solver-level noise only
RTOL = 1e-6
ATOL = 1e-5


def _run_cli(output_dir: Path) -> dict:
    runner = cli.get_runner("yaw_misalignment")
    config = load_config(CONFIG)
    assert config["seed"] == 42
    return runner.run(config, output_dir)


def test_cli_yaw_misalignment_regression(tmp_path: Path):
    results = _run_cli(tmp_path / "run1")

    assert results["analysis_type"] == "StaticYawMisalignment"
    assert results["turbine_ids"] == TURBINE_IDS
    assert results["UQ"] is False
    assert results["ws_bins"] == WS_BINS

    npt.assert_allclose(
        results["yaw_misalignment"], EXPECTED_YAW_MISALIGNMENT, rtol=RTOL, atol=ATOL
    )
    npt.assert_allclose(
        results["yaw_misalignment_ws"], EXPECTED_YAW_MISALIGNMENT_WS, rtol=RTOL, atol=ATOL
    )
    assert np.asarray(results["yaw_misalignment_ws"]).shape == (len(TURBINE_IDS), len(WS_BINS))

    # The JSON artifact holds the same numbers as the returned summary
    with (tmp_path / "run1" / "yaw_misalignment_results.json").open() as f:
        written = json.load(f)
    npt.assert_allclose(written["yaw_misalignment"], results["yaw_misalignment"], rtol=0, atol=0)
    npt.assert_allclose(
        written["yaw_misalignment_ws"], results["yaw_misalignment_ws"], rtol=0, atol=0
    )


def test_cli_yaw_misalignment_is_deterministic(tmp_path: Path):
    first = _run_cli(tmp_path / "run1")
    second = _run_cli(tmp_path / "run2")
    npt.assert_array_equal(first["yaw_misalignment"], second["yaw_misalignment"])
    npt.assert_array_equal(first["yaw_misalignment_ws"], second["yaw_misalignment_ws"])
