"""
=============================================================================
File Name:     generate_incomplete_labels.py
Description:   Performs stratified random sampling on a 2D label matrix to
               simulate the "Incomplete" weak supervision scenario in
               hyperspectral urban remote sensing classification. Utilizes
               NumPy vectorized operations for efficient processing of large
               matrices, avoiding explicit for-loops over the pixel space.

Author:        Enzhao
Date:          2026-04-24
Version:       1.1
Dependencies:  numpy, matplotlib
=============================================================================
"""

import numpy as np

from data.utils.progress import progress


def generate_incomplete_labels(label_matrix, sampling_rate=0.01, bg_value=0, seed=None, show_progress=True):
    """
    Performs stratified random sampling on a 2D label matrix to simulate
    an Incomplete weak supervision scenario.

    Parameters:
        label_matrix (np.ndarray): The original 2D label matrix (Absolute Ground Truth).
        sampling_rate (float): The sampling ratio, default is 0.01 (1%).
        bg_value (int): The background value (unselected pixels will be assigned this value). Default is 0.
        seed (int, optional): Random seed to ensure reproducibility of the generated weak labels.

    Returns:
        np.ndarray: The new label matrix after sampling, where unselected pixels are set to bg_value.
    """
    return generate_nested_incomplete_labels(
        label_matrix,
        [sampling_rate],
        bg_value=bg_value,
        seed=seed,
        show_progress=show_progress,
    )[float(sampling_rate)]


def generate_nested_incomplete_labels(label_matrix, sampling_rates, bg_value=0, seed=None, show_progress=True):
    """
    Generate nested stratified samples. For each class, one random permutation is
    shared by all rates, so smaller-rate labels are subsets of larger-rate labels.
    """
    rng = np.random.default_rng(seed)
    label_matrix = np.asarray(label_matrix)
    rates = sorted(float(rate) for rate in sampling_rates)

    weak_labels = {
        rate: np.full_like(label_matrix, fill_value=bg_value)
        for rate in rates
    }

    classes = [cls for cls in np.unique(label_matrix) if cls != bg_value]
    for cls in progress(classes, desc="Incomplete nested", unit="class", enabled=show_progress):
        cls_indices = np.flatnonzero(label_matrix == cls)
        if len(cls_indices) == 0:
            continue
        shuffled = rng.permutation(cls_indices)
        for rate in rates:
            num_samples = max(1, int(len(shuffled) * rate))
            np.put(weak_labels[rate], shuffled[:num_samples], cls)

    return weak_labels


# ==========================================
# Testing and Visualization (256x256 Matrix)
# ==========================================
if __name__ == "__main__":
    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.utils.demo_matrix import build_demo_label_matrix


    def test_and_visualize_random_sampling():
        """
        Simulates a 256x256 label matrix and visualizes the effect of
        stratified random sampling at different rates (10% and 1%).
        """
        print("Generating shared demo matrix and extracting random samples...")

        test_matrix = build_demo_label_matrix()
        matrix_size = test_matrix.shape[0]

        weak_10_percent = generate_incomplete_labels(test_matrix, sampling_rate=0.10, bg_value=0, seed=42)
        weak_1_percent = generate_incomplete_labels(test_matrix, sampling_rate=0.01, bg_value=0, seed=42)

        # 6. Plotting the results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot 1: Original Ground Truth
        ax1 = axes[0]
        ax1.imshow(test_matrix, cmap='viridis', origin='upper')
        ax1.set_title("Original 256x256 (Dense GT)", fontsize=14)

        # Helper function to plot sparse points clearly
        def plot_sparse_labels(ax, label_matrix, title):
            ax.imshow(test_matrix, cmap='viridis', origin='upper', alpha=0.2)

            # Extract and plot each class explicitly with distinct colors
            classes_colors = [(1, 'red', 'Class 1'),
                              (2, 'magenta', 'Class 2'),
                              (3, 'cyan', 'Class 3'),
                              (4, 'orange', 'Class 4')]

            for cls, color, label in classes_colors:
                y_coords, x_coords = np.where(label_matrix == cls)
                if len(x_coords) > 0:
                    # s=5 controls the dot size. Slightly larger for visibility.
                    ax.scatter(x_coords, y_coords, c=color, s=5, label=label)

            ax.set_title(title, fontsize=14)
            ax.legend(loc='upper right', markerscale=3)  # Make legend markers bigger

        # Subplot 2: 10% Sampling
        ax2 = axes[1]
        plot_sparse_labels(ax2, weak_10_percent, "10% Random Sampling (Incomplete)")

        # Subplot 3: 1% Sampling
        ax3 = axes[2]
        plot_sparse_labels(ax3, weak_1_percent, "1% Random Sampling (Extreme Incomplete)")

        # Format axes
        for ax in axes:
            ax.set_xlim(0, matrix_size - 1)
            ax.set_ylim(matrix_size - 1, 0)
            ax.set_xlabel("X Pixel")
            ax.set_ylabel("Y Pixel")
            ax.grid(color='white', linestyle='-', linewidth=0.2, alpha=0.5)

        plt.tight_layout()
        plt.show()


    test_and_visualize_random_sampling()
