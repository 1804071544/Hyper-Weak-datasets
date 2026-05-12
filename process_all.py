"""
Convert fully supervised hyperspectral labels into weak-supervision datasets.

The output root contains one folder per sub-dataset:

    root_path/
        <sub-dataset-a>
            /images
            /clean_labels
            /incomplete_labels
            /inaccurate_labels
            /inexact_labels
        <sub-dataset-b>
            ...
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("SKIMAGE_DATADIR", str(Path(tempfile.gettempdir()) / "skimage_data"))

from data.weak_label_gen.Inaccurate import generate_inaccurate_labels
from data.weak_label_gen.Incomplete import generate_nested_incomplete_labels
from data.weak_label_gen.Inexact import (
    generate_point_level_labels,
    generate_scribble_level_labels,
)
from data.sub_dataset.AeroRIT import load_aerorit_data
from data.sub_dataset.Chikusei import load_chikusei_data
from data.sub_dataset.Houston2013 import load_houston2013_data
from data.sub_dataset.Houston2018 import load_houston2018_data
from data.sub_dataset.Pavia_Centre import load_pavia_centre_data
from data.sub_dataset.Pavia_University import load_pavia_university_data
from data.sub_dataset.Washington_DC import load_washington_dc_data
from data.utils.config import set_dataset_root
from data.utils.format_saver import save_supervision_dataset
from data.utils.path_config import get_dataset_root


DatasetLoader = Callable[[], dict[str, Any]]


DATASET_LOADERS: dict[str, DatasetLoader] = {
    "AeroRIT": load_aerorit_data,
    "Chikusei": load_chikusei_data,
    "Houston2013": load_houston2013_data,
    "Houston2018": load_houston2018_data,
    "Pavia_Centre": load_pavia_centre_data,
    "Pavia_University": load_pavia_university_data,
    "Washington_DC": load_washington_dc_data,
}

DEFAULT_DATASETS_TO_PROCESS = (
    "AeroRIT",
    "Chikusei",
    "Houston2013",
    "Houston2018",
    "Pavia_Centre",
    "Pavia_University",
    "Washington_DC",
)

DEFAULT_INCOMPLETE_CONFIG = {
    "sampling_rates": [0.01, 0.05, 0.10, 0.20, 0.50],  # Pixel sampling rates for nested incomplete label maps.
    "bg_value": 0,  # Label value assigned to unlabeled/background pixels.
    "seed": 42,  # Random seed for reproducible stratified sampling.
    "show_progress": True,  # Whether to show progress bars during label generation.
}

DEFAULT_INACCURATE_CONFIG = {
    "noise_ratio": 0.30,  # Ratio of labeled foreground pixels to corrupt.
    "noise_weights": [1, 1, 1],  # Relative weights for symmetric, asymmetric, and boundary noise.
    "max_shift": 15,  # Maximum spatial shift, in pixels, for boundary-related noise.
    "adaptive_shift_ratio": 0.25,  # Scale factor used to adapt boundary shifts to local object geometry.
    "smooth_sigma": 10.0,  # Gaussian smoothing sigma used when estimating boundary noise regions.
    "bg_value": 0,  # Label value treated as background and excluded from foreground corruption.
    "seed": 42,  # Random seed for reproducible noise generation.
    "asym_chunk_size": 200000,  # Chunk size for asymmetric noise processing to limit memory use.
    "n_jobs": 1,  # Number of worker threads for noise generation.
    "show_progress": True,  # Whether to show progress bars during label generation.
}

DEFAULT_INEXACT_CONFIG = {
    "bg_value": 0,  # Label value assigned to unlabeled/background pixels.
    "show_progress": True,  # Whether to show progress bars during point/scribble generation.
}


def process_datasets(
    dataset_names: list[str] | tuple[str, ...],
    *,
    incomplete_config: dict[str, Any] | None = None,
    inaccurate_config: dict[str, Any] | None = None,
    inexact_config: dict[str, Any] | None = None,
) -> None:
    """Process selected datasets below ``F:/Hyper_Weak/<dataset>``."""
    incomplete_params = {**DEFAULT_INCOMPLETE_CONFIG, **(incomplete_config or {})}
    inaccurate_params = {**DEFAULT_INACCURATE_CONFIG, **(inaccurate_config or {})}
    inexact_params = {**DEFAULT_INEXACT_CONFIG, **(inexact_config or {})}

    for dataset_name in dataset_names:
        loader = DATASET_LOADERS.get(dataset_name)
        if loader is None:
            available = ", ".join(DATASET_LOADERS)
            raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {available}")

        print(f"\n========== Processing {dataset_name} ==========")
        print(f"[{dataset_name}] Step 1/4: load data")
        dataset = loader()
        images = _extract_images(dataset)
        clean_labels = np.asarray(dataset["labels"])
        primary_image = _select_primary_image(dataset_name, images)

        print(f"[{dataset_name}] Step 2/4: generate incomplete labels")
        _process_incomplete(
            dataset_name,
            dataset,
            clean_labels,
            incomplete_params,
        )

        print(f"[{dataset_name}] Step 3/4: generate inaccurate labels")
        _process_inaccurate(
            dataset_name,
            dataset,
            primary_image,
            clean_labels,
            inaccurate_params,
        )

        print(f"[{dataset_name}] Step 4/4: generate inexact labels")
        _process_inexact(
            dataset_name,
            dataset,
            clean_labels,
            inexact_params,
        )

        print(f"Finished {dataset_name}.")


def _process_incomplete(
    dataset_name: str,
    dataset: dict[str, Any],
    clean_labels: np.ndarray,
    params: dict[str, Any],
) -> None:
    output_dir = _dataset_output_dir(dataset_name)
    sampling_rates = params.get("sampling_rates", [params.get("sampling_rate", 0.01)])
    bg_value = params.get("bg_value", 0)
    seed = params.get("seed")
    show_progress = params.get("show_progress", True)
    nested_labels = generate_nested_incomplete_labels(
        clean_labels,
        sampling_rates,
        bg_value=bg_value,
        seed=seed,
        show_progress=show_progress,
    )
    weak_labels = {
        f"incomplete_labels_{_format_percent_name(sampling_rate)}": nested_labels[float(sampling_rate)]
        for sampling_rate in sampling_rates
    }

    save_supervision_dataset(
        output_dir,
        dataset,
        clean_labels,
        weak_labels,
        "Incomplete",
        params,
        weak_dir_name="incomplete_labels",
    )


def _process_inaccurate(
    dataset_name: str,
    dataset: dict[str, Any],
    primary_image: np.ndarray,
    clean_labels: np.ndarray,
    params: dict[str, Any],
) -> None:
    output_dir = _dataset_output_dir(dataset_name)
    noisy_labels, noise_type_mask, separate_noise_labels = generate_inaccurate_labels(
        primary_image,
        clean_labels,
        **params,
    )
    weak_labels = {
        "inaccurate_labels": noisy_labels,
        "noise_type_mask": noise_type_mask,
        **separate_noise_labels,
    }
    save_supervision_dataset(
        output_dir,
        dataset,
        clean_labels,
        weak_labels,
        "Inaccurate",
        params,
        weak_dir_name="inaccurate_labels",
    )


def _process_inexact(
    dataset_name: str,
    dataset: dict[str, Any],
    clean_labels: np.ndarray,
    params: dict[str, Any],
) -> None:
    output_dir = _dataset_output_dir(dataset_name)
    scribble_labels = generate_scribble_level_labels(clean_labels, **params)
    point_labels = generate_point_level_labels(clean_labels, **params)
    save_supervision_dataset(
        output_dir,
        dataset,
        clean_labels,
        {
            "scribble_labels": scribble_labels,
            "point_labels": point_labels,
        },
        "Inexact",
        params,
        weak_dir_name="inexact_labels",
    )


def _extract_images(dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    if "image" in dataset:
        return {"image": np.asarray(dataset["image"])}

    images = {}
    for key in ("reflectance", "radiance"):
        if key in dataset:
            images[key] = np.asarray(dataset[key])

    if not images:
        raise KeyError("Dataset loader must return 'image' or at least one image cube.")
    return images


def _select_primary_image(dataset_name: str, images: dict[str, np.ndarray]) -> np.ndarray:
    if "image" in images:
        return images["image"]
    if "reflectance" in images:
        return images["reflectance"]
    return next(iter(images.values()))


def _dataset_output_dir(dataset_name: str) -> Path:
    return get_dataset_root() / dataset_name


def _format_percent_name(rate: float) -> str:
    percent = float(rate) * 100.0
    if percent.is_integer():
        return f"{int(percent)}percent"
    return f"{percent:g}percent".replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate incomplete, inaccurate, and inexact weak labels."
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Dataset root path. If omitted, configs/paths.yaml is used.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=tuple(DATASET_LOADERS),
        help="Datasets to process. If omitted, all configured datasets are processed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dataset_root:
        set_dataset_root(args.dataset_root)
    datasets_to_process = tuple(args.datasets) if args.datasets else DEFAULT_DATASETS_TO_PROCESS
    process_datasets(datasets_to_process)
