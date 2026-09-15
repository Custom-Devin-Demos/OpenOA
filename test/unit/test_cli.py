from pathlib import Path

import yaml
import pytest
from attrs import field, define

from openoa import cli
from openoa.runners import PlantSpec, RunnerConfig, load_config

from test.conftest import example_data_path_str  # isort: skip


def test_list_methods(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out.split()
    assert out == sorted(cli.METHODS)


def test_unknown_method_rejected_by_argparse():
    with pytest.raises(SystemExit):
        cli.main(["run", "not_a_method", "--config", "x.yml"])


def test_get_runner_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        cli.get_runner("not_a_method")


def test_load_config_records_config_dir(tmp_path: Path):
    cfg = tmp_path / "run.yml"
    cfg.write_text(yaml.safe_dump({"seed": 1, "plant": {"loader": "a:b"}}))
    data = load_config(cfg)
    assert data["seed"] == 1
    assert data["config_dir"] == str(tmp_path)


def test_load_config_rejects_non_mapping(tmp_path: Path):
    cfg = tmp_path / "run.yml"
    cfg.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(cfg)


def test_plant_spec_requires_loader_or_metadata():
    with pytest.raises(ValueError, match="loader"):
        PlantSpec.from_dict({})
    with pytest.raises(ValueError, match="module:function"):
        PlantSpec.from_dict({"loader": "no_colon"})
    with pytest.raises(ValueError, match="analysis_type"):
        PlantSpec.from_dict({"loader": "a:b", "analysis_type": "NotAnAnalysis"})


@define(auto_attribs=True)
class _DemoConfig(RunnerConfig):
    num_sim: int = field(default=10)


def test_runner_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        _DemoConfig.from_config({"plant": {"loader": "a:b"}, "bogus": 1})


def test_runner_config_requires_plant():
    with pytest.raises(AttributeError, match="plant"):
        _DemoConfig.from_config({"num_sim": 3})


def test_runner_config_from_config_and_loader(tmp_path: Path):
    config = _DemoConfig.from_config(
        {
            "config_dir": str(tmp_path),
            "seed": 42,
            "num_sim": 3,
            "plant": {
                "loader": "examples.project_ENGIE:prepare",
                "loader_kwargs": {"path": example_data_path_str},
                "analysis_type": "ElectricalLosses",
            },
        }
    )
    assert config.num_sim == 3
    assert config.config_dir == tmp_path
    plant = config.load_plant()
    assert "ElectricalLosses" in plant.analysis_type
    assert plant.scada is not None
