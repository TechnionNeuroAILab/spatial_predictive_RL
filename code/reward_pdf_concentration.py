"""
reward_pdf_concentration.py

Reward-centered PDF concentration analysis for place cells.

This script:
- Defines a Gaussian around the reward, truncated to the arena, and its complement.
- Normalizes them to p_near(x) and p_far(x) PDFs over the arena.
- Uses both PDFs to compute a "posterior near-reward probability" per place-field center:
    post_near(x) = p_near(x) / (p_near(x) + p_far(x))
- Compares these between BEFORE and AFTER reinforcement learning, using:
    * Independent population analysis
    * Paired analysis (same neurons tracked)
- Generates figures and prints statistical test results.

Requires:
- shift_toward_reward.py in the same directory (for filter_place_cells, REWARD_POINT).
- data/df_skaggs_before.pkl, data/df_skaggs_after.pkl
- data/df_centers_before.pkl, data/df_centers_after.pkl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from shift_toward_reward import filter_place_cells, REWARD_POINT

# Arena bounds in normalized coordinates (x,y in [0,1])
ARENA_BOUNDS = (0.0, 1.0, 0.0, 1.0)
DEFAULT_SIGMA = 0.1  # Gaussian width around reward
FIG_DIR = "figures"


# ---------------------------------------------------------------------
# 1. Core Gaussian + complementary function + normalized PDFs
# ---------------------------------------------------------------------
def reward_gaussian_kernel(
    x,
    y,
    reward=REWARD_POINT,
    sigma=DEFAULT_SIGMA,
    arena=ARENA_BOUNDS,
):
    """
    Truncated Gaussian kernel centered at reward.
    x, y: arrays or scalars of positions in arena coordinates.
    arena: (xmin, xmax, ymin, ymax).
    Returns g(x,y) in [0,1], with 0 outside arena.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.zeros_like(x, dtype=float)

    xmin, xmax, ymin, ymax = arena
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    if not np.any(inside):
        return g

    dx = x[inside] - reward[0]
    dy = y[inside] - reward[1]
    g[inside] = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    return g


def compute_reward_pdfs(
    sigma=DEFAULT_SIGMA,
    reward=REWARD_POINT,
    arena=ARENA_BOUNDS,
    n_grid=256,
):
    """
    Numerically compute normalization constants and return
    functions p_near(x,y) and p_far(x,y), plus (Z_near, Z_far).

    g(x) = exp(-||x - reward||^2 / (2 sigma^2)) inside arena, 0 outside
    h(x) = 1 - g(x)

    p_near(x) = g(x) / Z_near
    p_far(x)  = h(x) / Z_far
    """
    xmin, xmax, ymin, ymax = arena
    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    g = reward_gaussian_kernel(X, Y, reward=reward, sigma=sigma, arena=arena)
    area = (xmax - xmin) * (ymax - ymin)

    # Integrals ≈ mean * area (uniform grid)
    Z_near = g.mean() * area
    h = 1.0 - g
    Z_far = h.mean() * area

    def p_near(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        g_xy = reward_gaussian_kernel(x, y, reward=reward, sigma=sigma, arena=arena)
        return g_xy / Z_near

    def p_far(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        g_xy = reward_gaussian_kernel(x, y, reward=reward, sigma=sigma, arena=arena)
        h_xy = 1.0 - g_xy
        return h_xy / Z_far

    return p_near, p_far, Z_near, Z_far


# ---------------------------------------------------------------------
# 2. Utilities for centers and scores
# ---------------------------------------------------------------------
def extract_xy_from_center(center):
    """
    Robustly extract a single (x,y) from the 'center' field.

    Handles:
    - (x, y)
    - [x, y]
    - np.array([x, y])
    - np.array([[x, y]])
    - [[x, y], ...] -> takes the first one (appropriate when k == 1).
    Returns (x, y) as floats, or None if it fails.
    """
    if center is None:
        return None

    arr = np.asarray(center, dtype=float)

    # (2,) -> single point
    if arr.ndim == 1 and arr.size == 2:
        return float(arr[0]), float(arr[1])

    # (N, 2) -> take first point (should be N=1 when k==1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return float(arr[0, 0]), float(arr[0, 1])

    # Fallback: if it's a list-like, try to interpret as list of centers
    try:
        pts = np.array([c for c in center if c is not None], dtype=float)
        if pts.ndim == 1 and pts.size == 2:
            return float(pts[0]), float(pts[1])
        if pts.ndim == 2 and pts.shape[1] == 2:
            return float(pts[0, 0]), float(pts[0, 1])
    except Exception:
        pass

    return None


def add_pdf_scores_to_centers(
    df_pc_centers,
    p_near,
    p_far,
    sigma=DEFAULT_SIGMA,
    reward=REWARD_POINT,
    arena=ARENA_BOUNDS,
):
    """
    Given a DataFrame of place-cell centers (must contain 'neuron_id' and 'center'),
    compute for each center:

    - x, y: coordinates
    - g_raw: unnormalized Gaussian value g(x)
    - p_near: normalized near-reward PDF
    - p_far: normalized far-from-reward PDF
    - posterior_near: p_near / (p_near + p_far)
    - llr_near_far: log (p_near / p_far)

    Returns a new DataFrame with these columns added.
    """
    rows = []
    eps = 1e-12

    for _, row in df_pc_centers.iterrows():
        nid = row["neuron_id"]
        center = row["center"]

        xy = extract_xy_from_center(center)
        if xy is None:
            continue
        x, y = xy

        g_val = reward_gaussian_kernel(x, y, reward=reward, sigma=sigma, arena=arena)
        pn = p_near(x, y)
        pf = p_far(x, y)

        posterior_near = pn / (pn + pf + eps)
        llr = np.log((pn + eps) / (pf + eps))

        rows.append(
            {
                "neuron_id": nid,
                "x": x,
                "y": y,
                "g_raw": float(g_val),
                "p_near": float(pn),
                "p_far": float(pf),
                "posterior_near": float(posterior_near),
                "llr_near_far": float(llr),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Visualization helpers
# ---------------------------------------------------------------------
def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_reward_pdfs(
    p_near,
    p_far,
    reward=REWARD_POINT,
    arena=ARENA_BOUNDS,
    n_grid=256,
    output_path=None,
):
    """
    Visualize p_near(x,y) and p_far(x,y) as 2D heatmaps over the arena.
    """
    xmin, xmax, ymin, ymax = arena
    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    p_near_grid = p_near(X, Y)
    p_far_grid = p_far(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(
        p_near_grid,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        aspect="equal",
    )
    axes[0].scatter(
        [reward[0]], [reward[1]], marker="*", s=80, edgecolors="k", facecolors="none"
    )
    axes[0].set_title("p_near(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        p_far_grid,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        aspect="equal",
    )
    axes[1].scatter(
        [reward[0]], [reward[1]], marker="*", s=80, edgecolors="k", facecolors="none"
    )
    axes[1].set_title("p_far(x, y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Saved PDF heatmaps to {output_path}")
    plt.close(fig)


def plot_posterior_histograms(
    post_before,
    post_after,
    title,
    output_path=None,
):
    """
    Overlaid histograms of posterior_near for before vs after populations.
    """
    if len(post_before) == 0 or len(post_after) == 0:
        print("No data to plot posterior histograms.")
        return

    df_plot = pd.DataFrame(
        {
            "posterior_near": np.concatenate([post_before, post_after]),
            "condition": (["Before"] * len(post_before) + ["After"] * len(post_after)),
        }
    )

    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df_plot,
        x="posterior_near",
        hue="condition",
        kde=True,
        stat="density",
        common_norm=False,
        bins=20,
        alpha=0.4,
    )
    plt.xlabel("Posterior near-reward probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Saved posterior histogram to {output_path}")
    plt.close()


def plot_paired_posterior_scatter(
    post_before,
    post_after,
    title,
    output_path=None,
):
    """
    Scatter plot of posterior_near: Before vs After for paired neurons.
    """
    if len(post_before) == 0:
        print("No paired data to plot.")
        return

    post_before = np.asarray(post_before)
    post_after = np.asarray(post_after)

    vmin = min(post_before.min(), post_after.min())
    vmax = max(post_before.max(), post_after.max())
    pad = 0.05 * (vmax - vmin + 1e-6)
    vmin = max(0.0, vmin - pad)
    vmax = min(1.0, vmax + pad)

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(post_before, post_after, alpha=0.7, s=25)
    plt.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2, label="y = x")
    plt.xlabel("Posterior near-reward (Before)")
    plt.ylabel("Posterior near-reward (After)")
    plt.title(title)
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Saved paired posterior scatter to {output_path}")
    plt.close()


# ---------------------------------------------------------------------
# 4. Statistical tests
# ---------------------------------------------------------------------
def independent_t_test_posterior(post_before, post_after, alpha=0.05):
    """
    Welch's t-test for independent populations on posterior_near.
    H0: mean(post_before) >= mean(post_after)
    H1: mean(post_before) <  mean(post_after)
    (i.e., AFTER is more concentrated near the reward)
    """
    post_before = np.asarray(post_before)
    post_after = np.asarray(post_after)

    n_b = len(post_before)
    n_a = len(post_after)

    if n_b < 2 or n_a < 2:
        print("Not enough data for independent t-test (N<2 in one group).")
        return

    t_stat, p_value = stats.ttest_ind(
        post_before,
        post_after,
        equal_var=False,
        alternative="less",  # H1: mean(before) < mean(after)
    )

    print("\n--- Independent t-test on posterior_near ---")
    print("H0: mean(post_before) >= mean(post_after)")
    print("H1: mean(post_before) <  mean(post_after)")
    print(f"N_before: {n_b}, mean_before: {post_before.mean():.4f}")
    print(f"N_after:  {n_a}, mean_after:  {post_after.mean():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (one-tailed): {p_value:.4e}")
    if p_value < alpha:
        print(f"Result: Significant increase after RL (p < {alpha}).")
    else:
        print(f"Result: Not significant (p >= {alpha}).")
    print("-------------------------------------------\n")


def paired_t_test_posterior(post_before, post_after, alpha=0.05):
    """
    Paired t-test for same neurons' posterior_near before vs after.
    H0: mean(post_before) >= mean(post_after)
    H1: mean(post_before) <  mean(post_after)
    """
    post_before = np.asarray(post_before)
    post_after = np.asarray(post_after)

    n = len(post_before)
    if n < 2:
        print("Not enough paired data for t-test (N<2).")
        return

    t_stat, p_value = stats.ttest_rel(
        post_before,
        post_after,
        alternative="less",  # H1: mean(before) < mean(after)
    )

    print("\n--- Paired t-test on posterior_near ---")
    print("H0: mean(post_before) >= mean(post_after)")
    print("H1: mean(post_before) <  mean(post_after)")
    print(f"N_paired: {n}")
    print(f"mean_before: {post_before.mean():.4f}")
    print(f"mean_after:  {post_after.mean():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (one-tailed): {p_value:.4e}")
    if p_value < alpha:
        print(f"Result: Significant increase after RL (p < {alpha}).")
    else:
        print(f"Result: Not significant (p >= {alpha}).")
    print("----------------------------------------\n")


# ---------------------------------------------------------------------
# 5. Main analyses
# ---------------------------------------------------------------------
def run_independent_pdf_analysis(
    df_skaggs_before,
    df_centers_before,
    df_skaggs_after,
    df_centers_after,
    sigma=DEFAULT_SIGMA,
    skaggs_prec=None,
):
    """
    Independent population analysis: treat BEFORE and AFTER place-cell
    populations as independent samples.

    Uses the same filter_place_cells as shift_toward_reward to define place cells,
    then computes posterior_near for each center, and compares distributions.
    """
    print("\n" + "=" * 60)
    print("Reward PDF concentration: Independent population analysis")
    print(f"(sigma = {sigma})")
    print("=" * 60)

    # Build PDFs once (common for before & after)
    p_near, p_far, Z_near, Z_far = compute_reward_pdfs(
        sigma=sigma, reward=REWARD_POINT, arena=ARENA_BOUNDS
    )
    print(f"Z_near ≈ {Z_near:.4f}, Z_far ≈ {Z_far:.4f} (should both be ~1.0)")

    # Place cells (using the same criteria as shift_toward_reward)
    df_pc_ids_before = filter_place_cells(
        df_skaggs_before, df_centers_before, skaggs_prec=skaggs_prec
    )
    df_pc_ids_after = filter_place_cells(
        df_skaggs_after, df_centers_after, skaggs_prec=skaggs_prec
    )

    # Restrict to single-bump neurons (k == 1), consistent with that filter
    df_centers_before_k1 = df_centers_before[df_centers_before["k"] == 1]
    df_centers_after_k1 = df_centers_after[df_centers_after["k"] == 1]

    df_before_pc_centers = df_pc_ids_before.merge(
        df_centers_before_k1[["neuron_id", "center"]],
        on="neuron_id",
        how="inner",
    )
    df_after_pc_centers = df_pc_ids_after.merge(
        df_centers_after_k1[["neuron_id", "center"]],
        on="neuron_id",
        how="inner",
    )

    print(
        f"N place-cell centers (independent): "
        f"{len(df_before_pc_centers)} before, {len(df_after_pc_centers)} after"
    )

    df_before_scores = add_pdf_scores_to_centers(
        df_before_pc_centers, p_near, p_far, sigma=sigma
    )
    df_after_scores = add_pdf_scores_to_centers(
        df_after_pc_centers, p_near, p_far, sigma=sigma
    )

    post_before = df_before_scores["posterior_near"].to_numpy()
    post_after = df_after_scores["posterior_near"].to_numpy()

    # Fraction of cells more likely "near" than "far"
    frac_near_before = np.mean(post_before > 0.5) if len(post_before) > 0 else np.nan
    frac_near_after = np.mean(post_after > 0.5) if len(post_after) > 0 else np.nan

    print(
        f"Fraction of cells with posterior_near > 0.5 (independent): "
        f"{frac_near_before:.3f} before, {frac_near_after:.3f} after"
    )

    # Histograms
    plot_posterior_histograms(
        post_before,
        post_after,
        title="Posterior near-reward probability (independent PCs)",
        output_path=os.path.join(
            FIG_DIR, "posterior_near_hist_independent.png"
        ),
    )

    # Statistical test
    independent_t_test_posterior(post_before, post_after, alpha=0.05)


def run_paired_pdf_analysis(
    df_skaggs_before,
    df_centers_before,
    df_centers_after,
    sigma=DEFAULT_SIGMA,
    skaggs_prec=None,
):
    """
    Paired analysis: track the SAME place cells before & after.
    Place cells are defined from BEFORE data (same as in shift_toward_reward).
    """
    print("\n" + "=" * 60)
    print("Reward PDF concentration: Paired analysis")
    print(f"(sigma = {sigma})")
    print("=" * 60)

    p_near, p_far, Z_near, Z_far = compute_reward_pdfs(
        sigma=sigma, reward=REWARD_POINT, arena=ARENA_BOUNDS
    )
    print(f"Z_near ≈ {Z_near:.4f}, Z_far ≈ {Z_far:.4f} (should both be ~1.0)")

    # Place cells defined based on BEFORE data
    df_pc_before_ids = filter_place_cells(
        df_skaggs_before, df_centers_before, skaggs_prec=skaggs_prec
    )

    df_before_k1 = df_centers_before[df_centers_before["k"] == 1]
    df_after_k1 = df_centers_after[df_centers_after["k"] == 1]

    place_ids = set(df_pc_before_ids["neuron_id"])
    before_ids = set(df_before_k1["neuron_id"])
    after_ids = set(df_after_k1["neuron_id"])

    common_ids = sorted(place_ids.intersection(before_ids).intersection(after_ids))
    print(f"N common place cells (paired): {len(common_ids)}")

    post_before_list = []
    post_after_list = []

    eps = 1e-12
    for nid in common_ids:
        row_b = df_before_k1[df_before_k1["neuron_id"] == nid]
        row_a = df_after_k1[df_after_k1["neuron_id"] == nid]
        if row_b.empty or row_a.empty:
            continue

        center_b = row_b["center"].iloc[0]
        center_a = row_a["center"].iloc[0]

        xy_b = extract_xy_from_center(center_b)
        xy_a = extract_xy_from_center(center_a)
        if xy_b is None or xy_a is None:
            continue

        xb, yb = xy_b
        xa, ya = xy_a

        pn_b = p_near(xb, yb)
        pf_b = p_far(xb, yb)
        pn_a = p_near(xa, ya)
        pf_a = p_far(xa, ya)

        post_b = pn_b / (pn_b + pf_b + eps)
        post_a = pn_a / (pn_a + pf_a + eps)

        post_before_list.append(post_b)
        post_after_list.append(post_a)

    post_before = np.asarray(post_before_list, dtype=float)
    post_after = np.asarray(post_after_list, dtype=float)

    print(
        f"Using {len(post_before)} neurons for paired posterior analysis "
        "(after filtering and valid centers)."
    )

    # Fraction of cells with higher posterior_near after RL
    if len(post_before) > 0:
        frac_increase = np.mean(post_after > post_before)
        print(
            f"Fraction of neurons with posterior_near(after) > posterior_near(before): "
            f"{frac_increase:.3f}"
        )

    # Scatter: before vs after
    plot_paired_posterior_scatter(
        post_before,
        post_after,
        title="Posterior near-reward probability (paired PCs)",
        output_path=os.path.join(
            FIG_DIR, "posterior_near_paired_scatter.png"
        ),
    )

    # Paired t-test
    paired_t_test_posterior(post_before, post_after, alpha=0.05)


# ---------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------
def main():
    ensure_fig_dir()
    sns.set(style="whitegrid")

    print("Loading data...")
    df_skaggs_before = pd.read_pickle("data/df_skaggs_before.pkl")
    df_skaggs_after = pd.read_pickle("data/df_skaggs_after.pkl")
    df_centers_before = pd.read_pickle("data/df_centers_before.pkl")
    df_centers_after = pd.read_pickle("data/df_centers_after.pkl")

    sigma = DEFAULT_SIGMA
    skaggs_prec = None  # or e.g. 0.8 if you want quantile-based filtering

    # Visualize the PDFs once
    p_near, p_far, Z_near, Z_far = compute_reward_pdfs(
        sigma=sigma, reward=REWARD_POINT, arena=ARENA_BOUNDS
    )
    plot_reward_pdfs(
        p_near,
        p_far,
        reward=REWARD_POINT,
        arena=ARENA_BOUNDS,
        output_path=os.path.join(FIG_DIR, "reward_pdfs.png"),
    )

    # Independent population analysis
    run_independent_pdf_analysis(
        df_skaggs_before,
        df_centers_before,
        df_skaggs_after,
        df_centers_after,
        sigma=sigma,
        skaggs_prec=skaggs_prec,
    )

    # Paired analysis (same cells before vs after)
    run_paired_pdf_analysis(
        df_skaggs_before,
        df_centers_before,
        df_centers_after,
        sigma=sigma,
        skaggs_prec=skaggs_prec,
    )

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
