"""Save weak_label_gen datasets as GeoTIFF files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from data.utils.progress import progress


def save_supervision_dataset(
    output_dir: str | Path,
    dataset: dict[str, Any],
    clean_labels: np.ndarray,
    weak_arrays: dict[str, np.ndarray],
    supervision_type: str,
    parameters: dict[str, Any],
    weak_dir_name: str = "weak_labels",
) -> None:
    """
    Save one generated weak-supervision dataset.

    Output layout:
        output_dir/images
        output_dir/clean_labels
        output_dir/<weak_dir_name>
    """
    output_path = Path(output_dir)
    meta = dataset.get("meta", {})
    label_mapping = dataset.get("label_mapping")

    _save_images(output_path / "images", dataset, meta)
    _save_label_array(
        output_path / "clean_labels",
        "labels",
        clean_labels,
        meta,
        overwrite=False,
    )

    if label_mapping is not None:
        _save_json(output_path / "clean_labels" / "label_mapping.json", label_mapping)

    weak_dir = output_path / weak_dir_name
    for name, array in weak_arrays.items():
        _save_label_array(weak_dir, name, array, meta)

    metadata = {
        "dataset": meta.get("dataset_name"),
        "supervision_type": supervision_type,
        "parameters": parameters,
        "label_format": "tif",
        "weak_label_dir": weak_dir_name,
    }
    _save_json(output_path / f"{weak_dir_name}_metadata.json", metadata)


def _save_images(images_dir: Path, dataset: dict[str, Any], meta: dict[str, Any]) -> None:
    image_paths = _image_paths(meta)
    for name, array in _image_arrays(dataset).items():
        target_path = images_dir / f"{name}.tif"
        if target_path.exists():
            print(f"Exists, skipped: {target_path}")
            continue
        _save_tif_array(target_path, array, reference_path=image_paths.get(name))


def _save_label_array(
    output_dir: Path,
    stem: str,
    array: np.ndarray,
    meta: dict[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    target_path = output_dir / f"{stem}.tif"
    reference_path = _label_reference_path(meta)
    if target_path.exists() and not overwrite:
        if _matches_label_reference(target_path, reference_path, array):
            print(f"Exists, skipped: {target_path}")
            return
        print(f"Exists but spatial reference changed, rewriting: {target_path}")
    _save_tif_array(target_path, array, reference_path=reference_path)


def _save_tif_array(
    output_path: str | Path,
    array: np.ndarray,
    reference_path: str | Path | None = None,
) -> None:
    import rasterio
    from rasterio.windows import Window

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array)
    if data.dtype == np.bool_:
        data = data.astype(np.uint8)

    if data.ndim == 2:
        write_data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        write_data = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Only 2D labels or 3D HWC images can be saved as tif, got shape {data.shape}.")

    count, height, width = write_data.shape
    profile = _tif_profile(reference_path, width, height, count, write_data.dtype)

    with rasterio.open(output, "w", **profile) as dst:
        block_height, block_width = _write_block_shape(height, width)
        row_count = math.ceil(height / block_height)
        col_count = math.ceil(width / block_width)
        total_blocks = count * row_count * col_count
        block_indices = (
            (band_idx, row, col)
            for band_idx in range(count)
            for row in range(0, height, block_height)
            for col in range(0, width, block_width)
        )
        for band_idx, row, col in progress(
            block_indices,
            desc=f"Saving {output.name}",
            total=total_blocks,
            unit="block",
            leave=True,
        ):
            window_height = min(block_height, height - row)
            window_width = min(block_width, width - col)
            window = Window(col, row, window_width, window_height)
            dst.write(
                write_data[
                    band_idx,
                    row : row + window_height,
                    col : col + window_width,
                ],
                band_idx + 1,
                window=window,
            )

    print(f"Saved: {output}")


def _write_block_shape(height: int, width: int) -> tuple[int, int]:
    """Return a moderate write window size for visible progress on large arrays."""
    return min(512, height), min(512, width)


def _tif_profile(
    reference_path: str | Path | None,
    width: int,
    height: int,
    count: int,
    dtype: np.dtype,
) -> dict[str, Any]:
    import rasterio

    profile = None
    if reference_path:
        reference = Path(reference_path)
        if reference.exists() and reference.suffix.lower() in {".tif", ".tiff"}:
            with rasterio.open(reference) as src:
                profile = src.profile.copy()

    if profile is None:
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": count,
            "dtype": str(np.dtype(dtype)),
        }

    profile.pop("nodata", None)
    profile.update(
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(np.dtype(dtype)),
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    return profile


def _label_reference_path(meta: dict[str, Any]) -> str | Path | None:
    return meta.get("label_reference_path") or meta.get("labels_path")


def _matches_label_reference(
    target_path: Path,
    reference_path: str | Path | None,
    array: np.ndarray,
) -> bool:
    reference = Path(reference_path) if reference_path else None
    if reference is None or not reference.exists() or reference.suffix.lower() not in {".tif", ".tiff"}:
        return True

    import rasterio

    data = np.asarray(array)
    height, width = data.shape[:2]
    with rasterio.open(reference) as ref, rasterio.open(target_path) as target:
        return (
            target.width == width
            and target.height == height
            and target.crs == ref.crs
            and np.allclose(tuple(target.transform), tuple(ref.transform))
        )


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Saved: {path}")


def _image_arrays(dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(dataset[key])
        for key in ("image", "reflectance", "radiance")
        if key in dataset
    }


def _image_paths(meta: dict[str, Any]) -> dict[str, str]:
    image_paths = meta.get("image_paths")
    if isinstance(image_paths, dict):
        return {str(key): str(value) for key, value in image_paths.items()}

    image_path = meta.get("image_path")
    if image_path:
        return {"image": str(image_path)}

    return {}
