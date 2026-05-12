"""
Loader for the Washington DC Mall hyperspectral dataset.
"""

from __future__ import annotations

import numpy as np
import rasterio

from data.utils.path_config import get_dataset_root
from data.utils.label_mapping import get_or_create_label_mapping


def load_washington_dc_data():
    """
    Load Washington_DC data from ``Raw_Data/Washington_DC``.

    Expected files:
        - dc.tif: hyperspectral image, read as (H, W, C)
        - dc_dctest_label.tif: ground-truth labels, read as (H, W)

    Returns:
        dict: ``{"image": image, "labels": labels, "label_mapping": label_mapping}``
    """
    dataset_dir = get_dataset_root() / "Raw_Data" / "Washington_DC"
    image_path = dataset_dir / "dc.tif"
    labels_path = dataset_dir / "dc_dctest_label.tif"

    for path in (image_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {path}")

    print("Loading Washington_DC hyperspectral image via rasterio...")
    image = _read_tiff_as_hwc(image_path)

    print("Loading Washington_DC ground truth labels via rasterio...")
    labels = _read_label_tiff(labels_path).astype("uint8", copy=False)

    if image.shape[:2] != labels.shape:
        raise ValueError(
            "Washington_DC image/label spatial shape mismatch: "
            f"{image.shape[:2]} vs {labels.shape}."
        )

    print("\n[Washington_DC Dataset Loaded Successfully]")
    print(f"Image Shape: {image.shape}")
    print(f"Label Shape: {labels.shape}")
    print(f"Image dtype:  {image.dtype}")
    print(f"Label dtype:  {labels.dtype}")

    label_mapping = get_or_create_label_mapping(labels, dataset_dir, "Washington_DC")

    return {
        "image": image,
        "labels": labels,
        "label_mapping": label_mapping,
        "meta": {
            "dataset_name": "Washington_DC",
            "image_format": "tif",
            "label_format": "tif",
            "image_path": str(image_path),
            "labels_path": str(labels_path),
        },
    }


def _read_tiff_as_hwc(filepath):
    with rasterio.open(filepath) as src:
        image_chw = src.read()
    return np.transpose(image_chw, (1, 2, 0))


def _read_label_tiff(filepath):
    with rasterio.open(filepath) as src:
        label_data = src.read()

    if label_data.shape[0] == 1:
        return label_data[0]

    labels_hwc = np.transpose(label_data, (1, 2, 0))
    height, width, channels = labels_hwc.shape
    flat_labels = labels_hwc.reshape(-1, channels)
    _, inverse_indices = np.unique(flat_labels, axis=0, return_inverse=True)
    labels = inverse_indices.reshape(height, width).astype(np.uint8)

    print("Warning: Washington_DC label image appears to be multi-band.")
    print("Converted multi-band labels into class indices.")
    return labels


if __name__ == "__main__":
    data = load_washington_dc_data()
    features = data["image"]
    ground_truth = data["labels"]
    print("\nReady for training / weak supervision pipeline.")
    print(f"features shape:     {features.shape}")
    print(f"ground_truth shape: {ground_truth.shape}")
