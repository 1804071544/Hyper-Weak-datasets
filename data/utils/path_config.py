"""Shared path helpers for preprocessing scripts under ``data/weak_label_gen``."""

from __future__ import annotations

from pathlib import Path

from data.utils.config import (
    get_dataset_root as _get_dataset_root,
    get_processed_data_dir as _get_processed_data_dir,
    get_raw_data_dir as _get_raw_data_dir,
    get_weak_label_dir as _get_weak_label_dir,
    get_weak_label_subdir as _get_weak_label_subdir,
)


def get_dataset_root() -> Path:
    """Return the only public root path used by preprocessing code."""
    return _get_dataset_root()


def get_raw_data_dir() -> Path:
    return _get_raw_data_dir()


def get_processed_data_dir() -> Path:
    return _get_processed_data_dir()


def get_weak_label_dir() -> Path:
    return _get_weak_label_dir()


def get_inaccurate_label_dir() -> Path:
    return _get_weak_label_subdir("inaccurate")


def get_incomplete_label_dir() -> Path:
    return _get_weak_label_subdir("incomplete")


def get_inexact_label_dir() -> Path:
    return _get_weak_label_subdir("inexact")
