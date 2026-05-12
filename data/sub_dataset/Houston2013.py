"""
Loader for the Houston 2013 hyperspectral dataset in MATLAB ``.mat`` format.
"""

from __future__ import annotations

from data.utils.path_config import get_dataset_root
from data.utils._mat_utils import (
    load_mat_variable,
    print_dataset_summary,
    validate_image_and_labels,
)
from data.utils.label_mapping import get_or_create_label_mapping


def load_houston2013_data():
    """
    Load Houston2013 data from ``Raw_Data/Houston2013``.

    Expected files:
        - Houstondata.mat, variable ``Houstondata``: (H, W, C)
        - Houstonlabel.mat, variable ``Houstonlabel``: (H, W)

    Returns:
        dict: ``{"image": image, "labels": labels, "label_mapping": label_mapping}``
    """
    dataset_dir = get_dataset_root() / "Raw_Data" / "Houston2013"
    image_path = dataset_dir / "Houstondata.mat"
    labels_path = dataset_dir / "Houstonlabel.mat"

    for path in (image_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {path}")

    print("Loading Houston2013 hyperspectral image from .mat...")
    image = load_mat_variable(image_path, ("Houstondata", "houstondata"))

    print("Loading Houston2013 ground truth labels from .mat...")
    labels = load_mat_variable(labels_path, ("Houstonlabel", "houstonlabel"))

    labels = labels.astype("uint8", copy=False)
    validate_image_and_labels(image, labels, "Houston2013")
    print_dataset_summary("Houston2013", image, labels)
    label_mapping = get_or_create_label_mapping(labels, dataset_dir, "Houston2013")

    return {
        "image": image,
        "labels": labels,
        "label_mapping": label_mapping,
        "meta": {
            "dataset_name": "Houston2013",
            "image_format": "mat",
            "label_format": "mat",
            "image_path": str(image_path),
            "labels_path": str(labels_path),
        },
    }


if __name__ == "__main__":
    data = load_houston2013_data()
    features = data["image"]
    ground_truth = data["labels"]
    print("\nReady for training / weak supervision pipeline.")
    print(f"features shape:     {features.shape}")
    print(f"ground_truth shape: {ground_truth.shape}")
    print('end')
