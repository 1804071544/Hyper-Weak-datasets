"""
Generate inexact weak annotations from dense label matrices.

Provides:
    - generate_scribble_level_labels
    - generate_point_level_labels
"""

from __future__ import annotations

import os
import tempfile
from collections import deque
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("SKIMAGE_DATADIR", str(Path(tempfile.gettempdir()) / "skimage_data"))

from skimage.measure import label as connected_components
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from data.utils.progress import progress


def generate_scribble_level_labels(label_matrix, bg_value=0, show_progress=True):
    label_matrix = np.asarray(label_matrix)
    scribble_matrix = np.full_like(label_matrix, fill_value=bg_value)
    classes = [cls for cls in np.unique(label_matrix) if cls != bg_value]

    for cls in progress(classes, desc="Inexact scribble", unit="class", enabled=show_progress):
        class_mask = label_matrix == cls
        y_slice, x_slice = _mask_bbox(class_mask)
        if y_slice.stop == y_slice.start:
            continue
        skeleton = skeletonize(np.ascontiguousarray(class_mask[y_slice, x_slice]))
        scribble_matrix[y_slice, x_slice][skeleton] = cls

    return scribble_matrix


def generate_point_level_labels(label_matrix, bg_value=0, show_progress=True):
    label_matrix = np.asarray(label_matrix)
    point_matrix = np.full_like(label_matrix, fill_value=bg_value)
    classes = [cls for cls in np.unique(label_matrix) if cls != bg_value]

    for cls in progress(classes, desc="Inexact point classes", unit="class", enabled=show_progress):
        components, _ = connected_components(label_matrix == cls, connectivity=2, return_num=True)
        regions = regionprops(components)
        for region in progress(
            regions,
            desc=f"Inexact point class {cls}",
            unit="region",
            leave=False,
            enabled=show_progress,
        ):
            y_min, x_min, y_max, x_max = region.bbox
            component = components[y_min:y_max, x_min:x_max] == region.label
            center = _get_longest_path_midpoint(component)
            if center is not None:
                point_matrix[y_min + center[0], x_min + center[1]] = cls

    return point_matrix


def _mask_bbox(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return slice(0, 0), slice(0, 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    return slice(y_min, y_max), slice(x_min, x_max)


def _get_longest_path_midpoint(binary_mask):
    skeleton = skeletonize(np.ascontiguousarray(binary_mask))
    coords = np.argwhere(skeleton)

    if len(coords) == 0:
        fallback = np.argwhere(binary_mask)
        if len(fallback) == 0:
            return None
        return tuple(np.round(np.mean(fallback, axis=0)).astype(int))
    if len(coords) == 1:
        return tuple(coords[0])

    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords)}
    idx_to_coord = {idx: tuple(coord) for idx, coord in enumerate(coords)}
    graph = {idx: [] for idx in range(len(coords))}
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for coord in coords:
        idx = coord_to_idx[tuple(coord)]
        for dy, dx in directions:
            neighbor = (coord[0] + dy, coord[1] + dx)
            if neighbor in coord_to_idx:
                graph[idx].append(coord_to_idx[neighbor])

    def bfs(start_idx):
        queue = deque([start_idx])
        parent = {start_idx: None}
        depth = {start_idx: 0}
        farthest = start_idx

        while queue:
            node = queue.popleft()
            if depth[node] > depth[farthest]:
                farthest = node
            for neighbor in graph[node]:
                if neighbor not in parent:
                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1
                    queue.append(neighbor)

        return farthest, parent

    def trace_path(parent, end_idx):
        path = []
        node = end_idx
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]

    end_a, _ = bfs(0)
    end_b, parent = bfs(end_a)
    path = trace_path(parent, end_b)
    return idx_to_coord[path[len(path) // 2]]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.utils.demo_matrix import build_demo_label_matrix

    print("Generating shared demo matrix and extracting weak annotations...")
    labels = build_demo_label_matrix()
    scribble_labels = generate_scribble_level_labels(labels, bg_value=0)
    point_labels = generate_point_level_labels(labels, bg_value=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(labels, cmap="viridis", origin="upper")
    axes[0].set_title("Dense labels")

    axes[1].imshow(labels, cmap="viridis", origin="upper", alpha=0.2)
    for cls, color in [(1, "red"), (2, "magenta"), (3, "cyan"), (4, "orange")]:
        y, x = np.where(scribble_labels == cls)
        if len(x) > 0:
            axes[1].scatter(x, y, c=color, s=5, label=f"Class {cls}")
    axes[1].set_title("Scribble labels")
    axes[1].legend(loc="upper right")

    axes[2].imshow(labels, cmap="viridis", origin="upper", alpha=0.2)
    for cls, color in [(1, "red"), (2, "magenta"), (3, "cyan"), (4, "orange")]:
        y, x = np.where(point_labels == cls)
        if len(x) > 0:
            axes[2].scatter(x, y, c=color, marker="*", s=220, edgecolor="white",
                            linewidths=1.2, label=f"Class {cls}")
    axes[2].set_title("Point labels")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.set_xlim(0, labels.shape[1] - 1)
        ax.set_ylim(labels.shape[0] - 1, 0)
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")

    plt.tight_layout()
    plt.show()
