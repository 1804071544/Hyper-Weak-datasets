"""Loader for the Houston 2018 hyperspectral dataset in GeoTIFF format."""

from __future__ import annotations

import numpy as np
import rasterio

from data.utils.path_config import get_dataset_root
from data.utils._mat_utils import (
    print_dataset_summary,
    validate_image_and_labels,
)
from data.utils.label_mapping import get_or_create_label_mapping


def load_houston2018_data():
    """
    Load Houston2018 data from ``Raw_Data/Houston2018``.

    Expected files:
        - Trian_image_0.5m.tif: hyperspectral image, read as (H, W, C)
        - TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif: label image, read as (H, W)

    Returns:
        dict: ``{"image": image, "labels": labels, "label_mapping": label_mapping}``
    """
    dataset_dir = get_dataset_root() / "Raw_Data" / "Houston2018"
    image_path = dataset_dir / "Trian_image_0.5m.tif"
    labels_path = dataset_dir / "TrainingGT" / "2018_IEEE_GRSS_DFC_GT_TR.tif"

    for path in (image_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {path}")

    print("Loading Houston2018 hyperspectral image from .tif...")
    with rasterio.open(image_path) as src:
        image = np.transpose(src.read(), (1, 2, 0))

    print("Loading Houston2018 ground truth labels from .tif...")
    with rasterio.open(labels_path) as src:
        labels = src.read(1)

    labels = labels.astype("uint8", copy=False)
    validate_image_and_labels(image, labels, "Houston2018")
    print_dataset_summary("Houston2018", image, labels)
    label_mapping = get_or_create_label_mapping(labels, dataset_dir, "Houston2018")

    return {
        "image": image,
        "labels": labels,
        "label_mapping": label_mapping,
        "meta": {
            "dataset_name": "Houston2018",
            "image_format": "tif",
            "label_format": "tif",
            "image_path": str(image_path),
            "labels_path": str(labels_path),
        },
    }


if __name__ == "__main__":
    data = load_houston2018_data()
    features = data["image"]
    ground_truth = data["labels"]
    print("\nReady for training / weak supervision pipeline.")
    print(f"features shape:     {features.shape}")
    print(f"ground_truth shape: {ground_truth.shape}")
    print('end')
