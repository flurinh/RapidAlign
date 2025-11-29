"""Configuration helpers for CLI scripts."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "baseline.json"


def apply_config(args: Namespace, parser: ArgumentParser) -> Namespace:
    config_path = getattr(args, "config", None)
    paths: list[Path] = []
    if config_path is not None:
        p = Path(config_path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        paths.append(p)
    else:
        default_path = _default_config_path()
        if default_path.is_file():
            paths.append(default_path)
    merged: dict[str, Any] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            merged.update(json.load(fh))
    for key, value in merged.items():
        setattr(args, key, value)
    return args


__all__ = ["apply_config"]
