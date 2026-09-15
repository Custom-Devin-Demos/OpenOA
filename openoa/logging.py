from __future__ import annotations

import os
import json
import logging
import logging.config
from typing import Any, TypeVar, Callable, ParamSpec
from pathlib import Path
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")


def setup_logging(
    console: bool = True,
    level: str = "WARNING",
    configuration: str | Path = "logging.json",
    env_key: str = "LOG_CFG",
) -> None:
    """Setup logging configuration"""
    if (value := os.getenv(env_key, None)) is not None:
        configuration = value
    configuration = Path(configuration).resolve()
    if configuration.is_file():
        with configuration.open("rt") as f:
            config: dict[str, Any] = json.load(f)
        config.setdefault("root", {})["level"] = level
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=level)

    logging.captureWarnings(True)


def logged_method_call(the_method: Callable[P, R], msg: str = "call") -> Callable[P, R]:
    @wraps(the_method)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = logging.getLogger(the_method.__module__)
        owner = args[0] if args else None
        logger.debug(f"{type(owner).__name__}#{id(owner)}.{the_method.__name__}: {msg}")
        return the_method(*args, **kwargs)

    _wrapper.__doc__ = the_method.__doc__
    return _wrapper


def logged_function_call(the_function: Callable[P, R], msg: str = "call") -> Callable[P, R]:
    @wraps(the_function)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = logging.getLogger(the_function.__module__)
        logger.debug(f"{the_function.__name__}: {msg}")
        return the_function(*args, **kwargs)

    return _wrapper


def set_log_level(value: str) -> str:
    """Update the logging level and return the validated value."""
    valid = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if value not in valid:
        raise ValueError(f"`log_level` is invalid. Please use one of: {valid}")
    logging.getLogger().setLevel(value)
    return value
