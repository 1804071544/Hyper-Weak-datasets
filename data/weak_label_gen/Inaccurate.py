"""Generate inaccurate weak labels for hyperspectral classification."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label as connected_components,
)
from scipy.spatial.distance import cdist

from data.utils.progress import progress


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def generate_inaccurate_labels(
    hsi_data,
    label_matrix,
    noise_ratio=0.30,
    noise_weights=[1, 1, 1],
    max_shift=5,
    smooth_sigma=5.0,
    bg_value=0,
    seed=None,
    adaptive_shift_ratio=0.15,
    asym_chunk_size=200000,
    n_jobs=1,
    show_progress=True,
):
    """
    Return combined noisy labels, a uint8 noise mask, and labels with each
    inaccurate noise type applied separately.

    noise_type_mask encoding:
        0 = clean, 1 = symmetric, 2 = asymmetric, 3 = boundary.
    """
    if seed is not None:
        np.random.seed(seed)

    hsi_data = np.asarray(hsi_data)
    labels = np.asarray(label_matrix)
    h, w, channels = hsi_data.shape
    valid_mask = labels != bg_value
    classes = np.unique(labels[valid_mask])
    if len(classes) < 2:
        clean = labels.copy()
        separate_labels = {
            "symmetric_noise_labels": clean.copy(),
            "asymmetric_noise_labels": clean.copy(),
            "boundary_noise_labels": clean.copy(),
        }
        return clean, np.zeros((h, w), dtype=np.uint8), separate_labels

    total_noise = int(np.sum(valid_mask) * noise_ratio)
    weights = np.asarray(noise_weights, dtype=float)
    weights = weights / weights.sum()
    target_sym = int(total_noise * weights[0])
    target_asym = int(total_noise * weights[1])
    target_bound = total_noise - target_sym - target_asym

    noisy = labels.copy()
    noise_mask = np.zeros((h, w), dtype=np.uint8)
    boundary_only = labels.copy()
    symmetric_only = labels.copy()
    asymmetric_only = labels.copy()

    if target_bound > 0:
        boundary_labels, boundary_mask = _generate_boundary_noise(
            labels,
            target_bound,
            max_shift=max_shift,
            smooth_sigma=smooth_sigma,
            adaptive_shift_ratio=adaptive_shift_ratio,
            bg_value=bg_value,
            show_progress=show_progress,
        )
        noisy[boundary_mask] = boundary_labels[boundary_mask]
        boundary_only[boundary_mask] = boundary_labels[boundary_mask]
        noise_mask[boundary_mask] = 3

        shortfall = target_bound - int(np.sum(boundary_mask))
        if shortfall > 0 and weights[0] + weights[1] > 0:
            target_sym += int(shortfall * weights[0] / (weights[0] + weights[1]))
            target_asym = total_noise - int(np.sum(boundary_mask)) - target_sym

    available = np.flatnonzero(valid_mask & (noise_mask == 0))
    np.random.shuffle(available)
    if target_sym + target_asym > len(available):
        target_sym = min(target_sym, len(available))
        target_asym = len(available) - target_sym

    sym_indices = available[:target_sym]
    asym_indices = available[target_sym:target_sym + target_asym]
    flat_noisy = noisy.ravel()
    flat_labels = labels.ravel()
    flat_symmetric = symmetric_only.ravel()
    flat_asymmetric = asymmetric_only.ravel()

    sym_values = _symmetric_noise_values(flat_labels, sym_indices, classes)
    flat_noisy[sym_indices] = sym_values
    flat_symmetric[sym_indices] = sym_values
    noise_mask.ravel()[sym_indices] = 1

    asym_values = _asymmetric_noise_values(
        flat_labels,
        hsi_data.reshape(-1, channels),
        asym_indices,
        classes,
        chunk_size=asym_chunk_size,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    flat_noisy[asym_indices] = asym_values
    flat_asymmetric[asym_indices] = asym_values
    noise_mask.ravel()[asym_indices] = 2

    separate_labels = {
        "symmetric_noise_labels": symmetric_only,
        "asymmetric_noise_labels": asymmetric_only,
        "boundary_noise_labels": boundary_only,
    }
    return noisy, noise_mask, separate_labels


def _generate_boundary_noise(
    labels,
    target_count,
    max_shift,
    smooth_sigma,
    adaptive_shift_ratio,
    bg_value,
    show_progress=True,
):
    target_labels = labels.copy()
    empty_mask = np.zeros(labels.shape, dtype=bool)
    if target_count <= 0 or max_shift <= 0:
        return target_labels, empty_mask

    boundary_segments = _find_boundary_segments(
        labels,
        max_shift,
        adaptive_shift_ratio,
        bg_value,
        show_progress=show_progress,
    )
    if not boundary_segments:
        return target_labels, empty_mask

    remaining = target_count
    max_rounds = max(1, int(np.ceil(target_count / max(1, len(boundary_segments) * 20))))
    blocked = np.zeros(labels.shape, dtype=bool)

    # Background boundaries use one direction per class pair to avoid simultaneous
    # expansion and shrinkage around the same object/background contour.
    pair_directions = {
        pair: np.random.choice([-1.0, 1.0])
        for pair, *_ in boundary_segments
        if bg_value in pair
    }

    for _ in range(max_rounds):
        segment_order = np.arange(len(boundary_segments))
        np.random.shuffle(segment_order)

        added_this_round = 0
        for index in progress(
            segment_order,
            desc=f"Inaccurate boundary round {_ + 1}/{max_rounds}",
            unit="segment",
            leave=False,
            enabled=show_progress,
        ):
            pair, y_slice, x_slice, skeleton, radius = boundary_segments[int(index)]
            patch_labels, changed = _deform_boundary_segment(
                labels,
                pair,
                y_slice,
                x_slice,
                skeleton,
                radius,
                smooth_sigma,
                bg_value,
                pair_directions.get(pair),
            )
            labels_crop = labels[y_slice, x_slice]
            target_crop = target_labels[y_slice, x_slice]
            blocked_crop = blocked[y_slice, x_slice]
            changed &= target_crop == labels_crop
            changed &= ~blocked_crop
            if not np.any(changed):
                continue

            # Keep the boundary budget strict; any shortfall is later reassigned
            # to pixel-level inaccurate noise.
            if int(np.sum(changed)) > remaining:
                changed = _smooth_subset(changed, remaining, smooth_sigma)
            target_crop[changed] = patch_labels[changed]
            changed_count = int(np.sum(changed))
            remaining -= changed_count
            added_this_round += changed_count
            blocked_crop |= binary_dilation(changed, structure=CONNECTIVITY_8, iterations=max(1, int(radius)))
            if remaining <= 0:
                return target_labels, target_labels != labels

        if added_this_round == 0:
            break

    return target_labels, target_labels != labels


def _find_boundary_segments(labels, max_shift, adaptive_shift_ratio, bg_value, show_progress=True):
    segment_masks = {}
    component_labels, component_short_sides = _foreground_component_shapes(labels, bg_value)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for dy, dx in progress(directions, desc="Inaccurate boundary scan", unit="dir", enabled=show_progress):
        src_y, dst_y = _slice_pair(labels.shape[0], dy)
        src_x, dst_x = _slice_pair(labels.shape[1], dx)
        center = labels[dst_y, dst_x]
        neighbor = labels[src_y, src_x]
        different = center != neighbor
        if not np.any(different):
            continue

        for value_a, value_b in np.unique(np.stack([center[different], neighbor[different]], axis=1), axis=0):
            pair = tuple(sorted((value_a.item(), value_b.item())))
            if pair not in segment_masks:
                segment_masks[pair] = np.zeros(labels.shape, dtype=bool)
            pair_mask = different & (
                ((center == pair[0]) & (neighbor == pair[1]))
                | ((center == pair[1]) & (neighbor == pair[0]))
            )
            segment_masks[pair][dst_y, dst_x] |= pair_mask

    segments = []
    for pair, mask in progress(
        segment_masks.items(),
        desc="Inaccurate boundary components",
        total=len(segment_masks),
        unit="pair",
        enabled=show_progress,
    ):
        components, count = connected_components(mask, structure=CONNECTIVITY_8)
        slices = find_objects(components)
        for component_id, obj in enumerate(slices, start=1):
            if obj is None:
                continue
            radius_y, radius_x = _pad_slices(obj[0], obj[1], labels.shape, padding=1)
            radius_crop = components[radius_y, radius_x] == component_id
            if not np.any(radius_crop):
                continue
            radius = _segment_radius(
                labels,
                pair,
                radius_crop,
                radius_y,
                radius_x,
                component_labels,
                component_short_sides,
                max_shift,
                adaptive_shift_ratio,
                bg_value,
            )
            skeleton_y, skeleton_x = _pad_slices(obj[0], obj[1], labels.shape, padding=radius)
            skeleton_crop = components[skeleton_y, skeleton_x] == component_id
            segments.append((pair, skeleton_y, skeleton_x, skeleton_crop, radius))
    return segments


def _foreground_component_shapes(labels, bg_value):
    component_labels = {}
    component_short_sides = {}
    for cls in np.unique(labels):
        if cls == bg_value:
            continue
        components, _ = connected_components(labels == cls, structure=CONNECTIVITY_8)
        slices = find_objects(components)
        component_labels[cls.item()] = components
        component_short_sides[cls.item()] = {
            idx: min(obj[0].stop - obj[0].start, obj[1].stop - obj[1].start)
            for idx, obj in enumerate(slices, start=1)
            if obj is not None
        }
    return component_labels, component_short_sides


def _segment_radius(
    labels,
    pair,
    boundary_component,
    y_slice,
    x_slice,
    component_labels,
    component_short_sides,
    max_shift,
    adaptive_shift_ratio,
    bg_value,
):
    labels_crop = labels[y_slice, x_slice]
    touching = binary_dilation(boundary_component, structure=CONNECTIVITY_8, iterations=1)
    foreground_sides = []
    for value in pair:
        if value == bg_value or value not in component_labels:
            continue
        component_ids = component_labels[value][y_slice, x_slice]
        touched_ids = np.unique(component_ids[touching & (labels_crop == value)])
        touched_ids = touched_ids[touched_ids > 0]
        foreground_sides.extend(
            component_short_sides[value][component_id]
            for component_id in touched_ids
            if component_id in component_short_sides[value]
        )
    if not foreground_sides:
        return 1
    min_side = min(foreground_sides)
    adaptive_limit = max(1, int(np.ceil(min_side * adaptive_shift_ratio)))
    half_side_limit = max(1, int(np.floor((min_side - 1) / 2)))
    return int(min(max(1, int(max_shift)), adaptive_limit, half_side_limit))


def _deform_boundary_segment(
    labels,
    pair,
    y_slice,
    x_slice,
    skeleton,
    radius,
    smooth_sigma,
    bg_value,
    forced_direction=None,
):
    side_a, side_b = pair
    labels_crop = labels[y_slice, x_slice]

    # Anchor the displacement on one side of the boundary, then let nearby pixels
    # inherit the nearest boundary anchor's movement direction and magnitude.
    boundary_seed = skeleton & (labels_crop == side_a)
    if not np.any(boundary_seed):
        boundary_seed = skeleton
    distance_to_seed, displacement = _smooth_boundary_displacement(boundary_seed, radius, smooth_sigma)
    if forced_direction is not None:
        displacement = float(forced_direction) * np.abs(displacement)

    threshold = max(0.5, float(radius) * 0.15)
    grow_a = (displacement > threshold) & (labels_crop == side_b) & (distance_to_seed <= displacement)
    grow_b = (displacement < -threshold) & (labels_crop == side_a) & (distance_to_seed <= -displacement)
    changed_crop = grow_a | grow_b

    # Non-background boundaries may bend both ways, but not at immediately
    # adjacent anchors; transition zones stay unchanged to avoid double edges.
    if forced_direction is None:
        changed_crop &= ~_boundary_direction_transition_zone(boundary_seed, displacement, radius, threshold)
    if not np.any(changed_crop):
        return labels_crop.copy(), changed_crop

    proposed_crop = labels_crop.copy()
    proposed_crop[grow_a] = side_a
    proposed_crop[grow_b] = side_b
    return proposed_crop, changed_crop


def _smooth_boundary_displacement(skeleton, radius, smooth_sigma):
    # Smooth random values on boundary anchors, then propagate each anchor's
    # displacement to its nearest surrounding pixels.
    displacement = np.random.uniform(-1.0, 1.0, size=skeleton.shape)
    if smooth_sigma > 0:
        sigma = min(float(smooth_sigma), max(skeleton.shape) / 4.0)
        displacement = gaussian_filter(displacement, sigma=sigma, mode="nearest")

    boundary_values = displacement[skeleton]
    if boundary_values.size == 0:
        return np.zeros(skeleton.shape, dtype=float)

    displacement -= float(np.median(boundary_values))
    scale = float(np.max(np.abs(displacement[skeleton])))
    if scale <= 1e-8:
        return np.full(skeleton.shape, np.inf), np.zeros(skeleton.shape, dtype=float)

    distance_to_seed, nearest = distance_transform_edt(~skeleton, return_indices=True)
    boundary_displacement = displacement[nearest[0], nearest[1]]
    return distance_to_seed, boundary_displacement / scale * max(1, int(radius))


def _boundary_direction_transition_zone(boundary_seed, displacement, radius, threshold):
    positive_seed = boundary_seed & (displacement > threshold)
    negative_seed = boundary_seed & (displacement < -threshold)
    if not np.any(positive_seed) or not np.any(negative_seed):
        return np.zeros(boundary_seed.shape, dtype=bool)

    transition_seed = (
        binary_dilation(positive_seed, structure=CONNECTIVITY_8, iterations=1)
        & binary_dilation(negative_seed, structure=CONNECTIVITY_8, iterations=1)
    )
    if not np.any(transition_seed):
        return np.zeros(boundary_seed.shape, dtype=bool)

    transition_radius = max(1, min(3, int(radius)))
    return distance_transform_edt(~transition_seed) <= transition_radius


def _pad_slices(y_slice, x_slice, shape, padding):
    padding = int(padding)
    return (
        slice(max(0, y_slice.start - padding), min(shape[0], y_slice.stop + padding)),
        slice(max(0, x_slice.start - padding), min(shape[1], x_slice.stop + padding)),
    )


def _smooth_subset(component, target_count, smooth_sigma):
    selected = np.zeros_like(component, dtype=bool)
    coords = np.argwhere(component)
    if target_count <= 0 or len(coords) == 0:
        return selected
    if len(coords) <= target_count:
        selected[component] = True
        return selected

    activation = np.random.random(component.shape)
    if smooth_sigma > 0:
        activation = gaussian_filter(activation, sigma=float(smooth_sigma), mode="nearest")

    values = activation[component]
    order = np.argsort(-values, kind="stable")
    chosen = coords[order[:target_count]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def _slice_pair(size, offset):
    if offset < 0:
        return slice(0, size + offset), slice(-offset, size)
    if offset > 0:
        return slice(offset, size), slice(0, size - offset)
    return slice(0, size), slice(0, size)


def _symmetric_noise_values(flat_labels, indices, classes):
    values = flat_labels[indices].copy()
    for cls in classes:
        cls_mask = flat_labels[indices] == cls
        other_classes = np.setdiff1d(classes, cls)
        if np.any(cls_mask) and len(other_classes) > 0:
            values[cls_mask] = np.random.choice(other_classes, size=int(np.sum(cls_mask)))
    return values


def _asymmetric_noise_values(
    flat_labels,
    flat_hsi,
    indices,
    classes,
    chunk_size=200000,
    n_jobs=1,
    show_progress=True,
):
    if len(indices) == 0:
        return flat_labels[indices].copy()
    centroids = np.asarray([np.mean(flat_hsi[flat_labels == cls], axis=0) for cls in classes])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def solve_chunk(chunk_indices):
        distances = cdist(flat_hsi[chunk_indices], centroids, metric="euclidean")
        true_indices = np.array([class_to_idx[cls] for cls in flat_labels[chunk_indices]])
        distances[np.arange(len(chunk_indices)), true_indices] = np.inf
        return np.asarray([classes[idx] for idx in np.argmin(distances, axis=1)])

    chunk_size = max(1, int(chunk_size))
    chunks = [indices[start:start + chunk_size] for start in range(0, len(indices), chunk_size)]
    if int(n_jobs) > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            results = progress(
                executor.map(solve_chunk, chunks),
                desc="Inaccurate asymmetric chunks",
                total=len(chunks),
                unit="chunk",
                enabled=show_progress,
            )
            return np.concatenate(list(results))
    chunks = progress(
        chunks,
        desc="Inaccurate asymmetric chunks",
        total=len(chunks),
        unit="chunk",
        enabled=show_progress,
    )
    return np.concatenate([solve_chunk(chunk) for chunk in chunks])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.utils.demo_matrix import build_demo_hsi_cube, build_demo_label_matrix

    labels = build_demo_label_matrix()
    hsi = build_demo_hsi_cube(labels)
    noisy, mask, separate_labels = generate_inaccurate_labels(hsi, labels, noise_ratio=0.30, seed=42)
    boundary_only = separate_labels["boundary_noise_labels"]
    boundary_only_mask = boundary_only != labels

    print("Noisy labels:", noisy.shape)
    print("Noise mask values:", np.unique(mask, return_counts=True))
    print("Changed pixels:", int(np.sum(noisy != labels)))

    def label_boundary(label_map):
        boundary = np.zeros(label_map.shape, dtype=bool)
        boundary[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
        boundary[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
        return boundary

    original_boundary = label_boundary(labels)
    deformed_boundary = label_boundary(boundary_only)
    boundary_window = binary_dilation(boundary_only_mask, structure=CONNECTIVITY_8, iterations=3)
    original_boundary_near_noise = original_boundary & boundary_window
    deformed_boundary_near_noise = deformed_boundary & boundary_window

    label_cmap = ListedColormap(["#111111", "#e41a1c", "#4daf4a", "#377eb8", "#ff7f00"])
    noise_cmap = ListedColormap(["#111111", "#ffd92f", "#00c5ff", "#d627ff"])
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes = axes.ravel()

    axes[0].imshow(labels, cmap=label_cmap, vmin=0, vmax=4, origin="upper")
    axes[0].set_title("Clean labels")
    axes[1].imshow(noisy, cmap=label_cmap, vmin=0, vmax=4, origin="upper")
    axes[1].set_title("Inaccurate labels")
    axes[2].imshow(mask, cmap=noise_cmap, vmin=0, vmax=3, origin="upper")
    axes[2].set_title("Noise type mask")
    axes[3].imshow(labels, cmap=label_cmap, vmin=0, vmax=4, origin="upper", alpha=0.3)
    y, x = np.where(boundary_only_mask)
    axes[3].scatter(x, y, s=2, c="#d627ff")
    axes[3].set_title("Boundary noise")
    axes[4].imshow(labels, cmap=label_cmap, vmin=0, vmax=4, origin="upper", alpha=0.35)
    y0, x0 = np.where(original_boundary_near_noise)
    y1, x1 = np.where(deformed_boundary_near_noise)
    axes[4].scatter(x0, y0, s=3, c="#ffffff", label="Original boundary")
    axes[4].scatter(x1, y1, s=2, c="#d627ff", label="Deformed boundary")
    axes[4].set_title("Boundary before/after")
    axes[4].legend(loc="upper right", markerscale=3, frameon=False)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()
