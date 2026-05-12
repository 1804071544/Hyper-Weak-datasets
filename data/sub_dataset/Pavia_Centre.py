"""
Loader for the Pavia Centre hyperspectral dataset in MATLAB ``.mat`` format.
"""

from __future__ import annotations

from data.utils.path_config import get_dataset_root
from data.utils._mat_utils import (
    load_mat_variable,
    print_dataset_summary,
    validate_image_and_labels,
)
from data.utils.label_mapping import get_or_create_label_mapping


def load_pavia_centre_data():
    """
    Load Pavia_Centre data from ``Raw_Data/Pavia_Centre``.

    Expected files:
        - Pavia.mat, variable ``pavia``: (H, W, C)
        - Pavia_gt.mat, variable ``pavia_gt``: (H, W)

    Returns:
        dict: ``{"image": image, "labels": labels, "label_mapping": label_mapping}``
    """
    dataset_dir = get_dataset_root() / "Raw_Data" / "Pavia_Centre"
    image_path = dataset_dir / "Pavia.mat"
    labels_path = dataset_dir / "Pavia_gt.mat"

    for path in (image_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {path}")

    print("Loading Pavia_Centre hyperspectral image from .mat...")
    image = load_mat_variable(image_path, ("pavia", "Pavia"))

    print("Loading Pavia_Centre ground truth labels from .mat...")
    labels = load_mat_variable(labels_path, ("pavia_gt", "Pavia_gt"))

    labels = labels.astype("uint8", copy=False)
    validate_image_and_labels(image, labels, "Pavia_Centre")
    print_dataset_summary("Pavia_Centre", image, labels)
    label_mapping = get_or_create_label_mapping(labels, dataset_dir, "Pavia_Centre")

    return {
        "image": image,
        "labels": labels,
        "label_mapping": label_mapping,
        "meta": {
            "dataset_name": "Pavia_Centre",
            "image_format": "mat",
            "label_format": "mat",
            "image_path": str(image_path),
            "labels_path": str(labels_path),
        },
    }


if __name__ == "__main__":
    data = load_pavia_centre_data()
    features = data["image"]
    ground_truth = data["labels"]
    print("\nReady for training / weak supervision pipeline.")
    print(f"features shape:     {features.shape}")
    print(f"ground_truth shape: {ground_truth.shape}")
