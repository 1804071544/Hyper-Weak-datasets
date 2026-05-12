"""Configuration loading helpers for dataset generation scripts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"
RAW_DATA_DIRNAME = "Raw_Data"
INACCURATE_DIRNAME = "Inaccurate"
INCOMPLETE_DIRNAME = "Incomplete"
INEXACT_DIRNAME = "Inexact"
_DATASET_ROOT_OVERRIDE: Path | None = None


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in config file: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_shared_paths_config() -> dict[str, Any]:
    """Load shared path settings used by dataset generation scripts."""
    if not SHARED_PATHS_CONFIG.exists():
        return {}
    return _read_yaml(SHARED_PATHS_CONFIG)


def set_dataset_root(path: str | Path | None) -> None:
    """Override the configured dataset root for the current process."""
    global _DATASET_ROOT_OVERRIDE
    _DATASET_ROOT_OVERRIDE = Path(path) if path is not None else None


def get_dataset_root() -> Path:
    """Return the globally configured dataset root directory."""
    if _DATASET_ROOT_OVERRIDE is not None:
        return _DATASET_ROOT_OVERRIDE

    shared = load_shared_paths_config()
    dataset_root = shared.get("data", {}).get("dataset_root")
    if not dataset_root:
        raise ValueError(
            f"`data.dataset_root` is not configured in {SHARED_PATHS_CONFIG}."
        )
    return Path(dataset_root)


def get_raw_data_dir() -> Path:
    """Return the raw data directory below the shared dataset root."""
    return get_dataset_root() / RAW_DATA_DIRNAME


def get_processed_data_dir() -> Path:
    """Return the default readable dataset directory below the shared dataset root."""
    return get_raw_data_dir()


def get_weak_label_dir() -> Path:
    """Return the weak-label root directory below the shared dataset root."""
    return get_dataset_root()


def get_weak_label_subdir(name: str) -> Path:
    """Return one weak-label scenario directory below the weak-label root."""
    subdirs = {
        "inaccurate": INACCURATE_DIRNAME,
        "incomplete": INCOMPLETE_DIRNAME,
        "inexact": INEXACT_DIRNAME,
    }
    subdir = subdirs.get(name)
    if subdir is None:
        raise ValueError(f"Unknown weak-label subdir name: {name}")
    return get_weak_label_dir() / subdir


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config merged with shared path settings."""
    config_path = Path(path)
    config = _read_yaml(config_path)
    if config_path.resolve() == SHARED_PATHS_CONFIG.resolve():
        return config
    shared = load_shared_paths_config()
    return _deep_merge(shared, config)
