"""
Shared helpers for MATLAB ``.mat`` based hyperspectral datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio


def load_mat_variable(
    mat_path: str | Path,
    variable_names: tuple[str, ...] | list[str] | None = None,
    *,
    transpose_hdf5_chw: bool = False,
) -> np.ndarray:
    """
    Load one array variable from a MATLAB ``.mat`` file.

    SciPy handles classic MAT files. MATLAB v7.3 files are HDF5 based and are
    read via h5py, where 3D hyperspectral cubes may need CHW -> HWC conversion.
    """
    path = Path(mat_path)
    names = tuple(variable_names or ())

    try:
        mat_data = sio.loadmat(path)
        variables = {
            key: value
            for key, value in mat_data.items()
            if not key.startswith("__")
        }
        key = _select_variable_key(variables, names, path)
        return np.asarray(variables[key]).squeeze()
    except NotImplementedError:
        return _load_hdf5_mat_variable(
            path,
            names,
            transpose_hdf5_chw=transpose_hdf5_chw,
        )


def _load_hdf5_mat_variable(
    path: Path,
    variable_names: tuple[str, ...],
    *,
    transpose_hdf5_chw: bool,
) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"{path} is a MATLAB v7.3/HDF5 file. Install h5py to read it."
        ) from exc

    with h5py.File(path, "r") as h5_file:
        datasets: dict[str, object] = {}

        def collect_dataset(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj

        h5_file.visititems(collect_dataset)
        key = _select_variable_key(datasets, variable_names, path)
        array = np.asarray(datasets[key][()]).squeeze()

    if transpose_hdf5_chw and array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))

    return array


def _select_variable_key(
    variables: dict[str, object],
    variable_names: tuple[str, ...],
    path: Path,
) -> str:
    if not variables:
        raise ValueError(f"No readable array variables found in {path}.")

    for expected_name in variable_names:
        if expected_name in variables:
            return expected_name

    lower_lookup = {key.lower(): key for key in variables}
    for expected_name in variable_names:
        key = lower_lookup.get(expected_name.lower())
        if key is not None:
            return key

    if len(variables) == 1:
        return next(iter(variables))

    available = ", ".join(sorted(variables))
    expected = ", ".join(variable_names) or "<single variable>"
    raise KeyError(
        f"Could not select variable from {path}. "
        f"Expected {expected}; available variables: {available}."
    )


def validate_image_and_labels(image: np.ndarray, labels: np.ndarray, name: str) -> None:
    """Ensure image and label spatial dimensions match."""
    if image.ndim != 3:
        raise ValueError(f"{name} image must be 3D (H, W, C), got {image.shape}.")
    if labels.ndim != 2:
        raise ValueError(f"{name} labels must be 2D (H, W), got {labels.shape}.")
    if image.shape[:2] != labels.shape:
        raise ValueError(
            f"{name} image/label spatial shape mismatch: "
            f"{image.shape[:2]} vs {labels.shape}."
        )


def print_dataset_summary(name: str, image: np.ndarray, labels: np.ndarray) -> None:
    print(f"\n[{name} Dataset Loaded Successfully]")
    print(f"Image Shape: {image.shape}")
    print(f"Label Shape: {labels.shape}")
    print(f"Image dtype:  {image.dtype}")
    print(f"Label dtype:  {labels.dtype}")

    print("\nFound Classes:")
    for cls in np.unique(labels):
        label_type = "(Background / Unlabeled)" if cls == 0 else ""
        pixel_count = int(np.sum(labels == cls))
        print(f"  Class {int(cls)} {label_type}: Pixels = {pixel_count}")
