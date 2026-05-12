"""
=============================================================================
File Name:     aerorit_loader_rasterio.py
Description:   Utility script to load the AeroRIT hyperspectral dataset using
               the `rasterio` library. Automatically handles the transposition
               from (C, H, W) to (H, W, C), and converts the 3-band RGB label
               image into a 1-band class index matrix for deep learning.
               Specifically maps RGB [153, 0, 0] to background index 0.

Author:        Enzhao
Date:          2026-04-27
Dependencies:  numpy, rasterio
=============================================================================
"""

import os
import numpy as np
import rasterio

# 按照你的需求导入路径配置函数
from data.utils.path_config import get_dataset_root
from data.utils.label_mapping import get_or_create_label_mapping


def load_aerorit_data():
    """
    Loads the AeroRIT hyperspectral dataset and processes the labels.

    Returns:
        dict: A dictionary containing:
            - "radiance": (H, W, C) numpy array.
            - "reflectance": (H, W, C) numpy array.
            - "labels": (H, W) 1D numpy array with integer class indices.
            - "color_map": Dictionary mapping RGB tuples to class indices.
            - "label_mapping": Label values, land-cover names, and pixel counts.
    """
    # 1. 获取根目录并拼接目标文件夹路径
    root_dir = get_dataset_root()
    dataset_dir = os.path.join(root_dir, "Raw_Data", "AeroRIT")

    # 2. 定义具体文件路径
    radiance_path = os.path.join(dataset_dir, "image_hsi_radiance.tif")
    reflectance_path = os.path.join(dataset_dir, "image_hsi_reflectance.tif")
    labels_path = os.path.join(dataset_dir, "image_labels.tif")

    # 3. 安全性检查
    for path in [radiance_path, reflectance_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected dataset file: {path}")

    # 4. 定义辅助读取函数 (转换为 H, W, C)
    def read_tiff_as_hwc(filepath):
        with rasterio.open(filepath) as src:
            img_chw = src.read()
            return np.transpose(img_chw, (1, 2, 0))

    # 5. 读取数据
    print("Loading AeroRIT Radiance data via rasterio...")
    radiance_img = read_tiff_as_hwc(radiance_path)

    print("Loading AeroRIT Reflectance data via rasterio...")
    reflectance_img = read_tiff_as_hwc(reflectance_path)

    print("Loading AeroRIT RGB Labels via rasterio...")
    labels_rgb = read_tiff_as_hwc(labels_path)

    # ==========================================
    # 核心新增逻辑：RGB 3波段 转 单波段 Index，并将 [153, 0, 0] 设为 0
    # ==========================================
    print("Converting RGB labels to 1D class indices...")
    H, W, C = labels_rgb.shape
    if C != 3:
        print(f"Warning: Label image has {C} bands. Expected 3 (RGB).")

    # 将空间维度展平，变为 (N, 3) 的二维矩阵
    flat_labels = labels_rgb.reshape(-1, C)

    # 利用 np.unique 获取唯一的颜色组合，并返回逆向索引
    unique_colors, inverse_indices = np.unique(flat_labels, axis=0, return_inverse=True)

    # 定义目标背景颜色
    target_bg = np.array([153, 0, 0])
    bg_match = np.all(unique_colors == target_bg, axis=1)

    # 初始化映射表 (默认不改变)
    remap = np.arange(len(unique_colors))

    if np.any(bg_match):
        old_bg_idx = np.where(bg_match)[0][0]

        # 如果目标背景的索引不是 0，则进行对调
        if old_bg_idx != 0:
            # 交换 unique_colors 数组中的位置，方便后续生成 color_map
            temp_color = unique_colors[0].copy()
            unique_colors[0] = unique_colors[old_bg_idx]
            unique_colors[old_bg_idx] = temp_color

            # 更新映射表：遇到 old_bg_idx 映射为 0，遇到 0 映射为 old_bg_idx
            remap[old_bg_idx] = 0
            remap[0] = old_bg_idx

        print(f"Successfully mapped target RGB {target_bg} to Class 0 (Background).")
    else:
        print(f"Warning: Target background RGB {target_bg} not found in label image.")

    # 利用映射表更新所有像素的索引，并 reshape 回图像的空间维度 (H, W)
    labels_1d = remap[inverse_indices].reshape(H, W).astype(np.uint8)

    # 生成颜色字典，方便用户知道对应关系
    color_map = {tuple(color): int(idx) for idx, color in enumerate(unique_colors)}

    print("\n[AeroRIT Dataset Loaded Successfully]")
    print(f"Radiance Shape:    {radiance_img.shape}")
    print(f"Reflectance Shape: {reflectance_img.shape}")
    print(f"Final Label Shape: {labels_1d.shape}  <-- 已转换为单波段")

    label_mapping = get_or_create_label_mapping(labels_1d, dataset_dir, "AeroRIT")

    return {
        "radiance": radiance_img,
        "reflectance": reflectance_img,
        "labels": labels_1d,
        "color_map": color_map,
        "label_mapping": label_mapping,
        "meta": {
            "dataset_name": "AeroRIT",
            "image_format": "tif",
            "label_format": "tif",
            "image_paths": {
                "radiance": radiance_path,
                "reflectance": reflectance_path,
            },
            "labels_path": labels_path,
            "label_reference_path": reflectance_path,
        },
    }


# ==========================================
# 测试代码 (Test execution)
# ==========================================
if __name__ == "__main__":
    # 调用函数获取数据
    aerorit_data = load_aerorit_data()

    # 提取我们需要的特征和 1D 标签
    features = aerorit_data["reflectance"]
    ground_truth = aerorit_data["labels"]

    # 现在的 ground_truth 就是标准的 (H, W) 矩阵，且背景 [153, 0, 0] 绝对是 0。
    # 可以直接传入弱监督函数中使用了！
