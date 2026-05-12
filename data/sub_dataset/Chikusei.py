"""
=============================================================================
File Name:     chikusei_loader_rasterio.py
Description:   Utility script to load the Chikusei hyperspectral dataset using
               rasterio. Automatically converts image data from (C, H, W) to
               (H, W, C). The ground-truth label TIFF is loaded as a standard
               2D class-index matrix.

Author:        Enzhao
Date:          2026-04-28
Dependencies:  numpy, rasterio
=============================================================================
"""

import os
import numpy as np
import rasterio

from data.utils.path_config import get_dataset_root
from data.utils.label_mapping import get_or_create_label_mapping


def load_chikusei_data():
    """
    Loads the Chikusei hyperspectral dataset.

    Expected directory structure:

        <dataset_root>/
            Raw_Data/
                Chikusei/
                    HyperspecVNIR_Chikusei_20140729.tif
                    HyperspecVNIR_Chikusei_20140729_Ground_Truth.tif

    Returns:
        dict: A dictionary containing:
            - "image": (H, W, C) numpy array.
            - "labels": (H, W) numpy array with integer class indices.
            - "label_mapping": Label values, land-cover names, and pixel counts.
    """

    # 1. 获取根目录并拼接 Chikusei 数据目录
    root_dir = get_dataset_root()
    dataset_dir = os.path.join(root_dir, "Raw_Data", "Chikusei")

    # 2. 定义数据和标签路径
    image_path = os.path.join(
        dataset_dir,
        "HyperspecVNIR_Chikusei_20140729.tif"
    )

    labels_path = os.path.join(
        dataset_dir,
        "HyperspecVNIR_Chikusei_20140729_Ground_Truth.tif"
    )

    # 3. 安全性检查
    for path in [image_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected dataset file: {path}")

    # 4. 读取高光谱影像，转换为 H, W, C
    def read_tiff_as_hwc(filepath):
        with rasterio.open(filepath) as src:
            img_chw = src.read()
            img_hwc = np.transpose(img_chw, (1, 2, 0))
        return img_hwc

    # 5. 读取标签
    def read_label_tiff(filepath):
        with rasterio.open(filepath) as src:
            label_data = src.read()

        # 如果标签是单波段: (1, H, W) -> (H, W)
        if label_data.shape[0] == 1:
            labels = label_data[0]

        # 如果标签意外是多波段 RGB: (3, H, W) -> (H, W, 3)
        else:
            labels = np.transpose(label_data, (1, 2, 0))

            # 对 RGB 标签做唯一颜色编码
            H, W, C = labels.shape
            flat_labels = labels.reshape(-1, C)
            unique_colors, inverse_indices = np.unique(
                flat_labels,
                axis=0,
                return_inverse=True
            )
            labels = inverse_indices.reshape(H, W).astype(np.uint8)

            print("Warning: Chikusei label image appears to be multi-band.")
            print("Converted RGB/multi-band labels into class indices.")
            print("Found label color map:")
            for idx, color in enumerate(unique_colors):
                print(f"  Class {idx}: {tuple(color)}")

        return labels

    # 6. 实际读取
    print("Loading Chikusei hyperspectral image via rasterio...")
    image = read_tiff_as_hwc(image_path)

    print("Loading Chikusei ground truth labels via rasterio...")
    labels = read_label_tiff(labels_path)

    # 7. 标签类型整理
    labels = labels.astype(np.uint8)

    print("\n[Chikusei Dataset Loaded Successfully]")
    print(f"Image Shape:  {image.shape}")
    print(f"Label Shape:  {labels.shape}")

    label_mapping = get_or_create_label_mapping(labels, dataset_dir, "Chikusei")

    return {
        "image": image,
        "labels": labels,
        "label_mapping": label_mapping,
        "meta": {
            "dataset_name": "Chikusei",
            "image_format": "tif",
            "label_format": "tif",
            "image_path": image_path,
            "labels_path": labels_path,
        },
    }


# ==========================================
# 测试代码 (Test execution)
# ==========================================
if __name__ == "__main__":
    chikusei_data = load_chikusei_data()

    features = chikusei_data["image"]
    ground_truth = chikusei_data["labels"]

    print("\nReady for training / weak supervision pipeline.")
    print(f"features shape:     {features.shape}")
    print(f"ground_truth shape: {ground_truth.shape}")
