"""
Loader for the Pavia University hyperspectral dataset in MATLAB ``.mat`` format.
"""

from __future__ import annotations

from data.utils.path_config import get_dataset_root
from data.utils._mat_utils import (
    load_mat_variable,
    print_dataset_summary,
    validate_image_and_labels,
)
from data.utils.label_mapping import get_or_create_label_mapping


def load_pavia_university_data():
    """
    Load Pavia_University data from ``Raw_Data/Pavia_University``.

    Expected files:
        - PaviaU.mat, variable ``paviaU``: (H, W, C)
        - PaviaU_gt.mat, variable ``paviaU_gt``: (H, W)

    Returns:
        dict: ``{"image": image, "labels": labels, "label_mapping": label_mapping}``
    """
    dataset_dir = get_dataset_root() / "Raw_Data" / "Pavia_University"
    image_path = dataset_dir / "PaviaU.mat"
    labels_path = dataset_dir / "PaviaU_gt.mat"

    for path in (image_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {path}")

    print("Loading Pavia_University hyperspectral image from .mat...")
    image = load_mat_variable(image_path, ("paviaU", "PaviaU"))

    print("Loading Pavia_University ground truth labels from .mat...")
    labels = load_mat_variable(labels_path, ("paviaU_gt", "PaviaU_gt"))

    labels = labels.astype("uint8", copy=False)
    validate_image_and_labels(image, labels, "Pavia_University")
    print_dataset_summary("Pavia_University", image, labels)
    label_mapping = get_or_create_label_mapping(labels, dataset_dir, "Pavia_University")

    return {
        "image": image,
        "labels": labels,
        "label_mapping": label_mapping,
        "meta": {
            "dataset_name": "Pavia_University",
            "image_format": "mat",
            "label_format": "mat",
            "image_path": str(image_path),
            "labels_path": str(labels_path),
        },
    }


if __name__ == "__main__":
    data = load_pavia_university_data()
    features = data["image"]
    ground_truth = data["labels"]
    print("\nReady for training / weak supervision pipeline.")
    print(f"features shape:     {features.shape}")
    print(f"ground_truth shape: {ground_truth.shape}")
