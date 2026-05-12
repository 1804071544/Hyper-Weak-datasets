"""
Create or read per-dataset label value mappings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LABEL_MAPPING_FILENAME = "label_mapping.json"


def get_or_create_label_mapping(
    labels: np.ndarray,
    dataset_dir: str | Path,
    dataset_name: str,
    *,
    filename: str = LABEL_MAPPING_FILENAME,
) -> dict[str, Any]:
    """
    Read an existing label mapping or compute a numeric fallback mapping.

    The mapping is read from or saved directly under each
    ``Raw_Data/<dataset>`` directory. To use semantic class names, edit that
    dataset-level ``label_mapping.json`` file. If it does not exist, labels are
    named by their numeric values.
    """
    mapping_path = Path(dataset_dir) / filename
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as file:
            mapping = json.load(file)
        print(f"Loaded existing label mapping: {mapping_path}")
        return mapping

    values, counts = np.unique(labels, return_counts=True)
    entries = []
    for value, count in zip(values, counts):
        label_value = int(value)
        entries.append(
            {
                "label": label_value,
                "land_cover": str(label_value),
                "pixel_count": int(count),
            }
        )

    mapping = {
        "dataset": dataset_name,
        "labels": entries,
    }

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Saved label mapping: {mapping_path}")
    return mapping


def print_label_mapping(label_mapping: dict[str, Any]) -> None:
    """Print a compact label mapping summary."""
    print("\nLabel Mapping:")
    for entry in label_mapping.get("labels", []):
        print(
            f"  Class {entry['label']}: {entry['land_cover']} "
            f"(Pixels = {entry['pixel_count']})"
        )
