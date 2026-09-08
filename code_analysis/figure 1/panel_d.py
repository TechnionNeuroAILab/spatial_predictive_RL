"""PCA cognitive-map panel: 3D embedding coloured by x, y, and heading."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

from utils import apply_nature_style

apply_nature_style()

PDF_DPI = 600
PNG_DPI = 300

# Shared colormap for allocentric x and y.
XY_CMAP = "viridis"
# Cyclic colormap for heading angle.
THETA_CMAP = "hsv"


def x_tertile_labels(x: np.ndarray) -> np.ndarray:
    """Return 0/1/2 labels based on x-position tertiles (fixed at 1/3, 2/3 of [0,1])."""
    return np.digitize(np.asarray(x, dtype=float), [1.0 / 3.0, 2.0 / 3.0])


def load_phase_probe(data_dir: Path):
    beh = pd.read_pickle(data_dir / "df_behavior_before.pkl")
    act = pd.read_pickle(data_dir / "df_activations_before.pkl")
    n = len(beh)
    positions = beh[["x", "y"]].to_numpy(dtype=float)
    theta = beh["theta"].to_numpy(dtype=float)
    activations = act["activity"].values.reshape(n, -1)
    return positions, theta, activations


def _subsample(n: int, max_points: int, seed: int = 42) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def render_panel_d(data_dir: Path, stem, reward_pos=(0.5, 0.5), max_points: int = 8000):
    """Three 3D PCA views: colour by x, y (same cmap), and theta (cyclic)."""
    positions, theta, activations = load_phase_probe(data_dir)
    x = positions[:, 0]
    y = positions[:, 1]

    act_c = activations - activations.mean(axis=0, keepdims=True)
    pca = PCA(n_components=3, random_state=42)
    z = pca.fit_transform(act_c)
    evr = pca.explained_variance_ratio_

    idx = _subsample(len(z), max_points)
    z, x, y, theta = z[idx], x[idx], y[idx], theta[idx]

    # Map heading to [0, 1) for a cyclic colormap.
    theta_norm = (theta + np.pi) / (2.0 * np.pi)

    fig = plt.figure(figsize=(7.8, 2.55))
    specs = [
        (x, XY_CMAP, "Allocentric x", (0.0, 1.0), "x"),
        (y, XY_CMAP, "Allocentric y", (0.0, 1.0), "y"),
        (theta_norm, THETA_CMAP, "Heading θ", (0.0, 1.0), "θ"),
    ]

    axes = []
    for i, (color, cmap, label, (vmin, vmax), short) in enumerate(specs):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        sc = ax.scatter(
            z[:, 0], z[:, 1], z[:, 2],
            c=color, s=2, alpha=0.35, cmap=cmap,
            vmin=vmin, vmax=vmax, linewidths=0, rasterized=True,  # points; axes/text stay vector
        )
        ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)", fontsize=5, labelpad=-2)
        ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)", fontsize=5, labelpad=-2)
        ax.set_zlabel(f"PC3 ({evr[2]*100:.0f}%)", fontsize=5, labelpad=-4)
        ax.tick_params(labelsize=4, pad=-2)
        ax.set_title(short, fontsize=6, pad=2)
        ax.view_init(elev=18, azim=-55)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, fraction=0.04)
        cbar.set_label(label, fontsize=5)
        cbar.ax.tick_params(labelsize=4)
        if i == 2:
            # Label cyclic colorbar in radians at ends.
            cbar.set_ticks([0.0, 0.5, 1.0])
            cbar.set_ticklabels(["−π", "0", "π"])
        axes.append(ax)

    fig.suptitle("Phase 1 manifold (random walk)", fontsize=6, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.08)

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, format="png", dpi=PNG_DPI, facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(pdf_path, format="pdf", dpi=PDF_DPI, facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path
