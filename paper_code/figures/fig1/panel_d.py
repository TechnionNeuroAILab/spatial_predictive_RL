"""PCA cognitive-map panel: 3D embedding coloured by x, y, and heading."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

from utils import FS_BASE, FS_SMALL, apply_nature_style

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
    """Load one probe period as (positions, heading, activations)."""
    beh = pd.read_pickle(data_dir / "df_behavior_before.pkl")
    act = pd.read_pickle(data_dir / "df_activations_before.pkl")
    n = len(beh)
    positions = beh[["x", "y"]].to_numpy(dtype=float)
    theta = beh["theta"].to_numpy(dtype=float)
    activations = act["activity"].values.reshape(n, -1)
    return positions, theta, activations


def _subsample(n: int, max_points: int, seed: int = 42) -> np.ndarray:
    """Sorted random subset of `n` row indices, capped at `max_points`."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def render_panel_d(data_dir: Path, stem, reward_pos=(0.5, 0.5), max_points: int = 8000,
                   width_in: float | None = None):
    """Three 3D PCA views: colour by x, y (same cmap), and theta (cyclic).

    Rendered at `width_in` (its final on-page width) so the type scale survives
    assembly. Numeric ticks are dropped: PC units are arbitrary, they cost most
    of the width at this size, and the panel's claim is that colour varies
    smoothly along the manifold, not what any coordinate equals. The variance
    explained is stated once beneath the strip instead of in nine axis labels.
    """
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

    w = width_in if width_in else 7.8
    fig = plt.figure(figsize=(w, w * 0.46))
    specs = [
        (x, XY_CMAP, (0.0, 1.0), "Arena x"),
        (y, XY_CMAP, (0.0, 1.0), "Arena y"),
        (theta_norm, THETA_CMAP, (0.0, 1.0), "Heading"),
    ]

    axes, mappables = [], []
    for i, (color, cmap, (vmin, vmax), short) in enumerate(specs):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        sc = ax.scatter(
            z[:, 0], z[:, 1], z[:, 2],
            c=color, s=2, alpha=0.35, cmap=cmap,
            vmin=vmin, vmax=vmax, linewidths=0, rasterized=True,  # points; axes/text stay vector
        )
        mappables.append(sc)
        ax.set_title(short, fontsize=FS_SMALL, pad=-2, loc="center")
        ax.view_init(elev=18, azim=-55)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(length=0)
        ax.set_xlabel("PC1", fontsize=4.6, labelpad=-11)
        ax.set_ylabel("PC2", fontsize=4.6, labelpad=-11)
        ax.set_zlabel("PC3", fontsize=4.6, labelpad=-11)

        # Clear the grey panes and heavy 3D gridlines: they add a lot of ink,
        # date the figure, and compete with the point cloud that carries the
        # actual result.
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor("#DDDDDD")
            axis.pane.set_linewidth(0.4)
            axis._axinfo["grid"].update(color="#EDEDED", linewidth=0.25)
            axis.line.set_color("#AAAAAA")
            axis.line.set_linewidth(0.4)
        axes.append(ax)

    # Two short horizontal bars under the strip: x and y share one scale
    # (normalised arena position), so one bar serves both; heading is cyclic and
    # needs its own. Previously three identical vertical 0-1 bars ate ~25% of
    # the width and their labels collided with the PC3 axis label.
    cax1 = fig.add_axes([0.09, 0.07, 0.24, 0.026])
    cb1 = fig.colorbar(mappables[0], cax=cax1, orientation="horizontal")
    cb1.set_ticks([0, 1]); cb1.set_ticklabels(["0", "1"])
    cb1.set_label("Arena x, y", fontsize=FS_SMALL, labelpad=1)

    cax2 = fig.add_axes([0.63, 0.07, 0.24, 0.026])
    cb2 = fig.colorbar(mappables[2], cax=cax2, orientation="horizontal")
    cb2.set_ticks([0, 0.5, 1]); cb2.set_ticklabels(["−π", "0", "π"])
    cb2.set_label("Heading", fontsize=FS_SMALL, labelpad=1)
    for cb in (cb1, cb2):
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=FS_SMALL, size=1.0, width=0.4, pad=1.0)

    # Variance explained, stated once instead of in nine axis labels. Top-right
    # of the strip: centred at the bottom it ran straight through both colourbars.
    fig.text(0.005, 0.995,
             f"PC1 {evr[0]*100:.0f}% · PC2 {evr[1]*100:.0f}% · PC3 {evr[2]*100:.0f}%",
             ha="left", va="top", fontsize=FS_SMALL, color="#6E6E6E")

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.18, wspace=0.0)

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
