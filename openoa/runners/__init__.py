"""Shared building blocks for the headless analysis runners used by :py:mod:`openoa.cli`.

A runner module (``openoa/runners/<method>.py``) must provide:

- ``Config``: an ``attrs`` class deriving from :py:class:`RunnerConfig` that declares and
  validates the method's configuration schema. Build it with :py:meth:`RunnerConfig.from_config`
  so unknown keys are rejected and required keys are enforced through the existing
  :py:class:`openoa.schema.FromDictMixin` machinery.
- ``run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]``: builds the ``Config``,
  loads the :py:class:`~openoa.plant.PlantData` described by ``config["plant"]``, runs the
  analysis, writes any artifacts to ``output_dir``, and returns a JSON-serializable summary.

Configuration file layout::

    output_dir: results/aep          # optional; overridden by --output-dir
    seed: 42                         # optional; seeds numpy and random before running
    plant:                           # see `PlantSpec`
      metadata: examples/data/plant_meta.yml
      analysis_type: MonteCarloAEP
      scada: data/scada.csv
      meter: data/meter.csv
      reanalysis:
        era5: data/era5.csv
    # ...method-specific keys, defined by the runner's `Config` class
"""

from __future__ import annotations

import json
import random
import importlib
from typing import Any, TypeVar
from pathlib import Path

import yaml
import attrs
import numpy as np
from attrs import field, define

from openoa.plant import PlantData
from openoa.schema import FromDictMixin
from openoa.schema.metadata import ANALYSIS_REQUIREMENTS

CONFIG_DIR_KEY = "config_dir"
C = TypeVar("C", bound="RunnerConfig")


def load_config(path: str | Path) -> dict[str, Any]:
    """Reads a YAML configuration file into a dictionary and records the file's directory under
    ``config_dir`` so relative data paths can be resolved against it."""
    path = Path(path).resolve()
    with path.open("rt") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file '{path}' must contain a mapping at the top level.")
    config = {str(k): v for k, v in data.items()}
    config.setdefault(CONFIG_DIR_KEY, str(path.parent))
    return config


def seed_everything(seed: int | None) -> None:
    """Seeds ``random`` and ``numpy`` so that Monte Carlo based methods are reproducible."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _resolve(path: str | Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def _resolve_reanalysis(
    data: dict[str, str | Path] | None, base_dir: Path
) -> dict[str, Path] | None:
    if data is None:
        return None
    resolved: dict[str, Path] = {}
    for name, path in data.items():
        p = _resolve(path, base_dir)
        if p is None:
            raise ValueError(f"Reanalysis product '{name}' has no file path.")
        resolved[name] = p
    return resolved


def _validate_analysis_type(inst: PlantSpec, attribute: attrs.Attribute[Any], value: Any) -> None:
    valid = [*ANALYSIS_REQUIREMENTS, "all"]
    values = value if isinstance(value, list) else [value]
    invalid = [v for v in values if v is not None and v not in valid]
    if invalid:
        raise ValueError(f"Invalid `analysis_type` values {invalid}; must be one of {valid}.")


@define(auto_attribs=True)
class PlantSpec(FromDictMixin):
    """Describes how to construct the :py:class:`~openoa.plant.PlantData` for a run.

    Either provide the CSV file paths for each data type alongside ``metadata`` (the
    ``PlantMetaData`` YAML/JSON file), or provide ``loader`` as ``"package.module:function"``
    pointing at a callable that returns a ``PlantData`` (for instance
    ``"examples.project_ENGIE:prepare"``) together with its ``loader_kwargs``. Relative paths are
    resolved against the configuration file's directory.
    """

    metadata: str | Path | None = field(default=None)
    analysis_type: str | list[str] | None = field(default=None, validator=_validate_analysis_type)
    scada: str | Path | None = field(default=None)
    meter: str | Path | None = field(default=None)
    tower: str | Path | None = field(default=None)
    status: str | Path | None = field(default=None)
    curtail: str | Path | None = field(default=None)
    asset: str | Path | None = field(default=None)
    reanalysis: dict[str, str | Path] | None = field(default=None)
    loader: str | None = field(default=None)
    loader_kwargs: dict[str, Any] = field(factory=dict)

    def __attrs_post_init__(self) -> None:
        if self.loader is None and self.metadata is None:
            raise ValueError("`plant` requires either `loader` or `metadata` plus data file paths.")
        if self.loader is not None and ":" not in self.loader:
            raise ValueError("`plant.loader` must be formatted as 'package.module:function'.")

    def load(self, base_dir: Path) -> PlantData:
        """Builds the ``PlantData`` object described by this specification, resolving relative
        file paths against ``base_dir``."""
        if self.loader is not None:
            module_name, _, func_name = self.loader.partition(":")
            module = importlib.import_module(module_name)
            func = getattr(module, func_name, None)
            if not callable(func):
                raise ValueError(f"`plant.loader` '{self.loader}' is not a callable.")
            kwargs = {
                k: str(_resolve(v, base_dir)) if k == "path" else v
                for k, v in self.loader_kwargs.items()
            }
            plant = func(**kwargs)
            if not isinstance(plant, PlantData):
                raise TypeError(f"`plant.loader` '{self.loader}' did not return a PlantData.")
        else:
            data: dict[str, Any] = {
                "metadata": _resolve(self.metadata, base_dir),
                "analysis_type": None,
                "scada": _resolve(self.scada, base_dir),
                "meter": _resolve(self.meter, base_dir),
                "tower": _resolve(self.tower, base_dir),
                "status": _resolve(self.status, base_dir),
                "curtail": _resolve(self.curtail, base_dir),
                "asset": _resolve(self.asset, base_dir),
                "reanalysis": _resolve_reanalysis(self.reanalysis, base_dir),
            }
            plant = PlantData(**data)
        if self.analysis_type is not None:
            types = (
                self.analysis_type if isinstance(self.analysis_type, list) else [self.analysis_type]
            )
            existing = plant.analysis_type or []
            plant.analysis_type = [*existing, *[t for t in types if t not in existing]]
            plant.validate()
        return plant


def _to_plant_spec(value: PlantSpec | dict[str, Any]) -> PlantSpec:
    if isinstance(value, PlantSpec):
        return value
    spec: PlantSpec = PlantSpec.from_dict(value)
    return spec


@define(auto_attribs=True)
class RunnerConfig(FromDictMixin):
    """Base configuration shared by every runner. Subclasses add the method-specific fields."""

    plant: PlantSpec = field(converter=_to_plant_spec)
    seed: int | None = field(default=None)
    output_dir: str | Path | None = field(default=None)
    config_dir: Path = field(factory=Path.cwd, converter=Path)

    @classmethod
    def from_config(cls: type[C], data: dict[str, Any]) -> C:
        """Builds the configuration from a parsed YAML mapping, rejecting unknown keys."""
        known = {a.name for a in attrs.fields(cls) if a.init}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown configuration keys for {cls.__name__}: {unknown}. "
                f"Valid keys: {sorted(known)}"
            )
        config: C = cls.from_dict(data)
        return config

    def load_plant(self) -> PlantData:
        """Seeds the random number generators and builds the ``PlantData``."""
        seed_everything(self.seed)
        return self.plant.load(self.config_dir)


def _json_default(o: object) -> object:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(data: dict[str, Any], path: Path) -> Path:
    """Writes ``data`` to ``path`` as JSON, converting numpy scalars and arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt") as f:
        json.dump(data, f, indent=2, default=_json_default)
    return path
