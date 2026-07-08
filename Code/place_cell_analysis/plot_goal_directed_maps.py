"""
Plot goal-directed cognitive maps from an existing training run.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

from pc_common import append_stat, resolve_reward_pos, write_stats_report

sns.set_theme(style="ticks")
def save_figure(fig, save_path, no_show=False, despine=True):
    """Despine, then save the figure as .png, .jpg, and .pdf"""
    if despine:
        try:
            sns.despine(fig=fig)
        except Exception:
            pass
    if save_path:
        base, _ = os.path.splitext(save_path)
        for ext in (".png", ".jpg", ".pdf"):
            fig.savefig(base + ext, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Saved: {base + ext}")
    if not no_show:
        plt.show()
    plt.close(fig)

def build_rate_maps(df_beh, df_act, m=32, hidden_dim=512):
    n_steps = len(df_beh)
    act_mat = df_act["activity"].values.reshape(n_steps, hidden_dim)
    x_bin = np.clip((df_beh["x"].values * m).astype(int), 0, m - 1)
    y_bin = np.clip((df_beh["y"].values * m).astype(int), 0, m - 1)
    gc = x_bin * m + y_bin
    n_bins = m * m
    bin_sum = np.zeros((n_bins, hidden_dim), dtype=np.float64)
    np.add.at(bin_sum, gc, act_mat)
    bin_count = np.bincount(gc, minlength=n_bins).astype(float)

    rate_flat = np.divide(bin_sum, bin_count[:, None],out=np.zeros_like(bin_sum),where=bin_count[:, None] > 0)
    rate_maps = rate_flat.T.reshape(hidden_dim, m, m)  # (N, m, m)
    occupancy = (bin_count / n_steps).reshape(m, m)
    return rate_maps, occupancy


def compute_skaggs(rate_maps, occupancy):
    hidden_dim = rate_maps.shape[0]
    occ_flat = occupancy.reshape(-1)                    # (m*m,)
    rate_flat = rate_maps.reshape(hidden_dim, -1).T     # (m*m, N)
    mean_rate = rate_flat.mean(axis=0)                  # (N,)
    ratio = np.where(mean_rate[None, :] > 1e-10,rate_flat / mean_rate[None, :], 0.0)
    log_ratio = np.where(ratio > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
    return (occ_flat[:, None] * ratio * log_ratio).sum(axis=0)


def smooth_and_norm(rm, sigma=1.5):
    s = gaussian_filter(rm, sigma=sigma)
    if s.max() > 0:
        s = (s - s.min()) / (s.max() - s.min())
    return s


def field_center_of_mass(rm):
    m = rm.shape[0]
    total = rm.sum()
    if total <= 0:
        return (0.5, 0.5)
    xs = np.arange(m) / m + 0.5 / m
    ys = np.arange(m) / m + 0.5 / m
    cx = (rm * xs[None, :]).sum() / total
    cy = (rm * ys[:, None]).sum() / total
    return (cx, cy)

def fig_comparison(rm_rw, skaggs_rw, rm_gc, skaggs_gc,top_idx, reward_bin, title_suffix="", save_path=None, no_show=False):
    n = len(top_idx)
    ncols = min(10, n)
    nrows_per = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows_per * 2, ncols,
                             figsize=(ncols * 1.9, nrows_per * 2 * 2.1))
    fig.patch.set_facecolor("#1a1a1a")
    rx, ry = reward_bin
    for i, nid in enumerate(top_idx):
        row_rw = i // ncols
        row_gc = row_rw + nrows_per
        col = i % ncols

        for ax, rm_src, sk in [
            (axes[row_rw, col], rm_rw, skaggs_rw),
            (axes[row_gc, col], rm_gc, skaggs_gc),
        ]:
            ax.imshow(smooth_and_norm(rm_src[nid]), origin="lower",
                      cmap="jet", vmin=0, vmax=1)
            ax.plot(rx, ry, "w*", markersize=5)
            ax.axis("off")
            ax.set_facecolor("black")
    # Row labels
    for row in range(nrows_per):
        axes[row, 0].set_ylabel("Random\nWalk", fontsize=7,rotation=0, labelpad=38, color="white")
    for row in range(nrows_per, nrows_per * 2):
        axes[row, 0].set_ylabel("Goal-\nDirected", fontsize=7,rotation=0, labelpad=38, color="white")
    for i in range(n, ncols * nrows_per):
        axes[i // ncols, i % ncols].axis("off")
        axes[i // ncols + nrows_per, i % ncols].axis("off")
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_occupancy(occ_rw, occ_gc, reward_bin, save_path=None, no_show=False,
                   label_a="Random Walk", label_b="Goal-Directed"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rx, ry = reward_bin
    for ax, occ, label in zip(axes, [occ_rw, occ_gc], [label_a, label_b]):
        im = ax.imshow(occ.T, origin="lower", cmap="inferno", aspect="equal")
        ax.plot(rx * occ.shape[0] / occ.shape[0],
                ry * occ.shape[1] / occ.shape[1], "c*", markersize=16)
        ax.axis("off")
        ax.set_title(label, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Occupancy fraction")

    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_ori(rm_rw, skaggs_rw, rm_gc, skaggs_gc, reward_pos, threshold=0.05, save_path=None, no_show=False,
            label_a="Random walk", label_b="Goal-directed"):
    def centres(rm, sk):
        place_ids = np.where(sk >= threshold)[0]
        if len(place_ids) == 0:
            return np.empty((0, 2))
        return np.array([field_center_of_mass(rm[nid]) for nid in place_ids])
    c_rw = centres(rm_rw, skaggs_rw)
    c_gc = centres(rm_gc, skaggs_gc)
    d_rw = np.linalg.norm(c_rw - reward_pos, axis=1) if len(c_rw) else np.array([])
    d_gc = np.linalg.norm(c_gc - reward_pos, axis=1) if len(c_gc) else np.array([])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ax = axes[0]
    bins = np.linspace(0, 0.8, 22)
    if len(d_rw):
        ax.hist(d_rw, bins=bins, alpha=0.55, color="steelblue",label=f"{label_a} (n={len(d_rw)})", density=True)
    if len(d_gc):
        ax.hist(d_gc, bins=bins, alpha=0.55, color="tomato",label=f"{label_b} (n={len(d_gc)})", density=True)
    ax.axvline(0, color="gold", linewidth=2.5, linestyle="--", label="Reward location")
    ax.set_xlabel("Distance of field centre from reward", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax = axes[1]
    if len(c_rw):
        ax.scatter(c_rw[:, 0], c_rw[:, 1], alpha=0.3, s=9, c="steelblue")
    ax.plot(*reward_pos, "r*", markersize=16)
    ax.add_patch(plt.Circle(reward_pos, 0.15, color="red", fill=False,linewidth=1.5, linestyle="--"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(label_a, fontsize=12)
    ax = axes[2]
    if len(c_gc):
        ax.scatter(c_gc[:, 0], c_gc[:, 1], alpha=0.3, s=9, c="tomato")
    ax.plot(*reward_pos, "r*", markersize=16)
    ax.add_patch(plt.Circle(reward_pos, 0.15, color="red", fill=False,linewidth=1.5, linestyle="--"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(label_b, fontsize=12)

    plt.tight_layout()
    save_figure(fig, save_path, no_show)


def fig_100cells(rate_maps, skaggs, top_idx, reward_bin, title="Cognitive Map", save_path=None, no_show=False):
    n_show = min(100, len(top_idx))
    rx, ry = reward_bin

    fig = plt.figure(figsize=(15, 15), dpi=150)

    for i in range(n_show):
        nid = top_idx[i]
        ax = fig.add_subplot(10, 10, i + 1)
        rm = smooth_and_norm(rate_maps[nid])
        ax.imshow(rm, origin="lower", cmap="jet", interpolation="gaussian", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)

def _load_cycle_rate_maps(data_dir, m=32, hidden_dim=512, probe_type="goal"):
    if probe_type == "goal":
        cycle_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("df_activations_cycle_") and f.endswith("_goal.pkl")
        ])
    else:
        cycle_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("df_activations_cycle_") and f.endswith(".pkl")
            and "goal" not in f
        ])
    cycle_data = {}
    for fname in cycle_files:
        label = fname.replace("df_activations_", "").replace(".pkl", "")
        df_a = pd.read_pickle(os.path.join(data_dir, fname))
        beh_fname = fname.replace("df_activations_", "df_behavior_")
        df_b = pd.read_pickle(os.path.join(data_dir, beh_fname))
        rm, occ = build_rate_maps(df_b, df_a, m, hidden_dim)
        sk = compute_skaggs(rm, occ)
        cycle_data[label] = (rm, occ, sk)
    return cycle_data


def _cycle_num(label):
    """Extract the integer cycle index from a label like 'cycle_3' or 'cycle_3_goal'."""
    mobj = re.search(r"cycle_(\d+)", label)
    return int(mobj.group(1)) if mobj else -1

def fig_drift_across_cycles(data_dir, reward_pos, m=32, hidden_dim=512,
                            skaggs_threshold=0.05, max_neurons=60,
                            save_path=None, no_show=False):
    """Track field-centre trajectory of active neurons across cycles."""
    cycle_data = _load_cycle_rate_maps(data_dir, m, hidden_dim, probe_type="goal")
    if len(cycle_data) < 2:
        print("  Need >=2 cycles for drift figure; skipping.")
        return
    labels = sorted(cycle_data.keys())
    rm_last, _, sk_last = cycle_data[labels[-1]]
    active = np.where(sk_last >= skaggs_threshold)[0]
    if len(active) == 0:
        print("  No active neurons above Skaggs threshold; skipping drift.")
        return
    if len(active) > max_neurons:
        rng = np.random.default_rng(0)
        active = rng.choice(active, max_neurons, replace=False)
    centres = np.zeros((len(labels), len(active), 2))
    for ci, lab in enumerate(labels):
        rm = cycle_data[lab][0]
        for ni, nid in enumerate(active):
            centres[ci, ni] = field_center_of_mass(smooth_and_norm(rm[nid]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    for ni in range(len(active)):
        ax.plot(centres[:, ni, 0], centres[:, ni, 1],
                color="gray", alpha=0.3, linewidth=0.6, zorder=2)
    for ci, lab in enumerate(labels):
        ax.scatter(centres[ci, :, 0], centres[ci, :, 1],
                   c=[cmap[ci]], s=18, alpha=0.75, zorder=3,
                   label=lab.replace("_", " ").title())
    ax.plot(*reward_pos, "r*", markersize=22, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[1]
    disp = np.linalg.norm(centres - centres[0:1], axis=2)
    mean_disp = disp.mean(axis=1)
    std_disp = disp.std(axis=1)
    xs = np.arange(len(labels))
    ax.plot(xs, mean_disp, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.fill_between(xs, mean_disp - std_disp, mean_disp + std_disp,color="steelblue", alpha=0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace("cycle_", "C").replace("_goal", "") for l in labels])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean displacement from cycle-1 centre")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path, no_show)


def fig_ori_across_cycles(data_dir, reward_pos, m=32, hidden_dim=512,skaggs_threshold=0.05, save_path=None, no_show=False):
    """Mean field-centre distance to reward (ORI) per cycle."""
    cycle_data = _load_cycle_rate_maps(data_dir, m, hidden_dim, probe_type="goal")
    if len(cycle_data) == 0:
        print("  No cycle data; skipping ORI-per-cycle figure.")
        return
    labels = sorted(cycle_data.keys())
    means, stds, n_cells = [], [], []
    for lab in labels:
        rm, occ, sk = cycle_data[lab]
        active = np.where(sk >= skaggs_threshold)[0]
        if len(active) == 0:
            means.append(np.nan); stds.append(np.nan); n_cells.append(0)
            continue
        centres = np.array([field_center_of_mass(smooth_and_norm(rm[nid]))
                            for nid in active])
        dists = np.linalg.norm(centres - reward_pos, axis=1)
        means.append(dists.mean()); stds.append(dists.std()); n_cells.append(len(active))

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(labels))
    ax.errorbar(xs, means, yerr=stds, fmt="o-", color="tomato",linewidth=2, capsize=4, markersize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace("cycle_", "C").replace("_goal", "") for l in labels])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean |field-centre − reward|")
    for x, n, mu in zip(xs, n_cells, means):
        if not np.isnan(mu):
            ax.annotate(f"n={n}", (x, mu), textcoords="offset points",
                        xytext=(6, 8), fontsize=8, color="gray")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path, no_show)

def fig_100cells_random(rate_maps, reward_bin, mean_rate_threshold=0.01,n_show=100, seed=0, title="Random active neurons",save_path=None, no_show=False):
    """100 randomly-chosen active neurons."""
    mean_rates = rate_maps.mean(axis=(1, 2))
    active = np.where(mean_rates >= mean_rate_threshold)[0]
    if len(active) == 0:
        print("  No active neurons; skipping random-100 figure.")
        return
    rng = np.random.default_rng(seed)
    sel = rng.choice(active, size=min(n_show, len(active)), replace=False)
    rx, ry = reward_bin
    fig = plt.figure(figsize=(15, 15), dpi=150)
    for i, nid in enumerate(sel):
        ax = fig.add_subplot(10, 10, i + 1)
        rm = smooth_and_norm(rate_maps[nid])
        ax.imshow(rm, origin="lower", cmap="jet",
                  interpolation="gaussian", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_cycle_progression(data_dir, top_idx_gc, reward_bin, m, hidden_dim, save_path=None, no_show=False):
    cycle_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith("df_activations_cycle_") and f.endswith(".pkl")
        and "goal" not in f
    ])
    if not cycle_files:
        print("  No per-cycle activation files found, skipping progression figure.")
        return
    n_show = min(10, len(top_idx_gc))
    sel = top_idx_gc[:n_show]
    n_cyc = len(cycle_files)
    rx, ry = reward_bin
    fig, axes = plt.subplots(n_cyc, n_show, figsize=(n_show * 1.9, n_cyc * 2.1))
    fig.patch.set_facecolor("#1a1a1a")
    if n_cyc == 1:
        axes = [axes]

    for row, fname in enumerate(cycle_files):
        label = fname.replace("df_activations_", "").replace(".pkl", "")
        df_a = pd.read_pickle(os.path.join(data_dir, fname))
        beh_fname = fname.replace("df_activations_", "df_behavior_")
        df_b = pd.read_pickle(os.path.join(data_dir, beh_fname))
        rm_c, occ_c = build_rate_maps(df_b, df_a, m, hidden_dim)
        for col, nid in enumerate(sel):
            ax = axes[row][col]
            ax.imshow(smooth_and_norm(rm_c[nid]), origin="lower",cmap="jet", vmin=0, vmax=1)
            ax.plot(rx, ry, "w*", markersize=5)
            ax.axis("off")
            ax.set_facecolor("black")
            if col == 0:
                ax.set_ylabel(label.replace("_", " ").title(),
                              fontsize=7, rotation=0, labelpad=50, color="white")

    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_pred_vs_rl_across_cycles(rw_cycles, goal_cycles, neuron_id, reward_bin,save_path=None, no_show=False):
    """For ONE neuron: top row = prediction (random-walk probe), bottom row = RL (goal-directed probe); one column per cycle."""
    rw_labels = sorted(rw_cycles.keys(), key=_cycle_num)
    goal_labels = sorted(goal_cycles.keys(), key=_cycle_num)
    n = max(len(rw_labels), len(goal_labels))
    if n == 0:
        print("  No per-cycle data; skipping pred-vs-RL figure.")
        return
    rx, ry = reward_bin
    fig, axes = plt.subplots(2, n, figsize=(n * 1.9, 2 * 2.1), squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")
    for col in range(n):
        for row, cycles, labels in [(0, rw_cycles, rw_labels),
                                     (1, goal_cycles, goal_labels)]:
            ax = axes[row][col]
            if col < len(labels):
                rm = cycles[labels[col]][0]
                ax.imshow(smooth_and_norm(rm[neuron_id]), origin="lower",
                          cmap="jet", vmin=0, vmax=1)
                ax.plot(rx, ry, "w*", markersize=5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            for spine in ax.spines.values():
                spine.set_visible(False)
    # row labels (axis labels, not headers)
    axes[0][0].set_ylabel("Prediction", fontsize=10, color="white")
    axes[1][0].set_ylabel("RL", fontsize=10, color="white")
    # cycle labels along the bottom
    for col in range(n):
        lab = goal_labels[col] if col < len(goal_labels) else rw_labels[col]
        axes[1][col].set_xlabel(f"C{_cycle_num(lab)}", fontsize=9, color="white")
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_drift_matrix(goal_cycles, skaggs_gc, n_cells=100,save_path=None, no_show=False):
    """Neuron x cycle heatmap of field-centre drift: each row is one place cell,
    each column a training cycle, colour = displacement of that cell's field
    centre from its cycle-1 position. Rows sorted by final drift, so stable
    cells (dark) and the few drifting cells (bright) are immediately visible."""
    labels = sorted(goal_cycles.keys(), key=_cycle_num)
    if len(labels) < 2:
        print("  Need >=2 cycles for drift matrix; skipping.")
        return
    top = np.argsort(skaggs_gc)[::-1][:n_cells]
    centres = np.zeros((len(labels), len(top), 2))
    for ci, lab in enumerate(labels):
        rm = goal_cycles[lab][0]
        for ni, nid in enumerate(top):
            centres[ci, ni] = field_center_of_mass(smooth_and_norm(rm[nid]))
    # displacement of each neuron from its cycle-1 centre, at every cycle
    disp = np.linalg.norm(centres - centres[0:1], axis=2).T# (n_cells, n_cycles)
    disp = disp[np.argsort(disp[:, -1])[::-1]]# sort rows by final drift
    fig, ax = plt.subplots(figsize=(5.5, 8))
    im = ax.imshow(disp, aspect="auto", cmap="magma", origin="upper",
                   vmin=0, vmax=np.nanpercentile(disp, 99))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"C{_cycle_num(l)}" for l in labels], fontsize=11)
    ax.set_yticks([0, len(top) - 1])
    ax.set_yticklabels(["1", str(len(top))], fontsize=10)
    ax.set_xlabel("Training cycle", fontsize=12)
    ax.set_ylabel(f"Place cells (sorted by drift, n={len(top)})", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Field-centre displacement from cycle 1", fontsize=11)
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_drift_rdm(goal_cycles, skaggs_gc, n_cells=100, save_path=None, no_show=False):
    """Cycle-to-cycle dissimilarity matrix (annotated). Entry (i,j) = mean
    field-centre distance between cycle i and cycle j across the top n_cells
    place cells. Diagonal = 0; small off-diagonals = the map is stable."""
    labels = sorted(goal_cycles.keys(), key=_cycle_num)
    if len(labels) < 2:
        print("  Need >=2 cycles for drift RDM; skipping.")
        return
    top = np.argsort(skaggs_gc)[::-1][:n_cells]
    nc = len(labels)
    centres = np.zeros((nc, len(top), 2))
    for ci, lab in enumerate(labels):
        rm = goal_cycles[lab][0]
        for ni, nid in enumerate(top):
            centres[ci, ni] = field_center_of_mass(smooth_and_norm(rm[nid]))
    mat = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            mat[i, j] = np.linalg.norm(centres[i] - centres[j], axis=1).mean()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="magma", origin="upper", vmin=0)
    ticks = [f"C{_cycle_num(l)}" for l in labels]
    ax.set_xticks(range(nc)); ax.set_xticklabels(ticks, fontsize=11)
    ax.set_yticks(range(nc)); ax.set_yticklabels(ticks, fontsize=11)
    ax.set_xlabel("Training cycle", fontsize=12)
    ax.set_ylabel("Training cycle", fontsize=12)
    thr = mat.max() * 0.6 if mat.max() > 0 else 1.0
    for i in range(nc):
        for j in range(nc):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < thr else "black", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean field-centre distance between cycles", fontsize=11)
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_drift_rdm_10(rw_cycles, goal_cycles, skaggs_gc, n_cells=100,save_path=None, no_show=False):
    """10x10 dissimilarity matrix over 10 timepoints = 5 cycles x
    {Prediction (random-walk probe), RL (goal-directed probe)}, interleaved per
    cycle. Entry (i,j) = mean field-centre distance between timepoints i and j
    across the top n_cells. Captures both the consolidated map and the
    displacement after the RL part of each cycle."""
    rw_by_num = {_cycle_num(l): l for l in rw_cycles}
    goal_by_num = {_cycle_num(l): l for l in goal_cycles}
    cycles = sorted(set(rw_by_num) & set(goal_by_num))
    if len(cycles) < 2:
        print("  Need >=2 cycles for 10x10 drift matrix; skipping.")
        return
    top = np.argsort(skaggs_gc)[::-1][:n_cells]
    def centres_of(cyc_dict, lab):
        rm = cyc_dict[lab][0]
        return np.array([field_center_of_mass(smooth_and_norm(rm[nid])) for nid in top])
    series = []   # (tick label, centres (n_cells, 2))
    for c in cycles:
        series.append((f"C{c}·P", centres_of(rw_cycles, rw_by_num[c])))    # Prediction
        series.append((f"C{c}·R", centres_of(goal_cycles, goal_by_num[c])))  # RL
    nt = len(series)
    mat = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            mat[i, j] = np.linalg.norm(series[i][1] - series[j][1], axis=1).mean()

    ticks = [s[0] for s in series]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(mat, cmap="magma", origin="upper", vmin=0)
    ax.set_xticks(range(nt)); ax.set_xticklabels(ticks, fontsize=9, rotation=90)
    ax.set_yticks(range(nt)); ax.set_yticklabels(ticks, fontsize=9)
    ax.set_xlabel("Cycle · phase  (P = prediction, R = RL)", fontsize=11)
    thr = mat.max() * 0.6 if mat.max() > 0 else 1.0
    for i in range(nt):
        for j in range(nt):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < thr else "black", fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean field-centre distance", fontsize=11)
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def main():
    parser = argparse.ArgumentParser(
        description="Re-plot cognitive maps from saved pkl data (no retraining needed)."
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory, e.g. New_Results_v2/per_group_lr_v3")
    parser.add_argument("--reward_pos", type=float, nargs=2, default=None,
                        metavar=("X", "Y"), help="Reward (x, y). Defaults to run_dir/config.json's reward_pos.")
    parser.add_argument("--n_cells", type=int, default=20,
                        help="Number of cells to show in comparison figure (default: 20)")
    parser.add_argument("--m", type=int, default=32, help="Grid resolution (default: 32)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="RNN hidden size (default: 512)")
    parser.add_argument("--skaggs_threshold", type=float, default=1.0,
                        help="Place-cell cutoff in bits/spike. Visually validated: 0.15-0.5 includes "
                             "diffuse/noisy cells, >=1.0 reliably gives clean localized fields.")
    parser.add_argument("--no_show", action="store_true",
                        help="Save figures without displaying them")
    parser.add_argument("--fig_subdir", type=str, default="",
                        help="Optional subfolder under <run_dir>/figures/ to save into")
    parser.add_argument("--n_example_neurons", type=int, default=5,
                        help="How many top neurons to render in the pred-vs-RL-across-cycles figure")
    parser.add_argument("--drift_cells", type=int, default=100,
                        help="Number of neurons in the drift matrix (default: 100 -> 10x10)")
    parser.add_argument("--panel_neuron", type=int, default=425,
                        help="Neuron rendered as the named pred-vs-RL panel figure (default: 425)")
    args = parser.parse_args()
    data_dir = os.path.join(args.run_dir, "data")
    fig_dir = os.path.join(args.run_dir, "figures", args.fig_subdir) if args.fig_subdir \
              else os.path.join(args.run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    reward_pos = resolve_reward_pos(args.run_dir, args.reward_pos)
    m = args.m
    hidden_dim = args.hidden_dim
    reward_bin = (int(reward_pos[0] * m), int(reward_pos[1] * m))
    print("Loading probe data from:", data_dir)
    df_beh_rw = pd.read_pickle(os.path.join(data_dir, "df_behavior_before.pkl"))
    df_act_rw = pd.read_pickle(os.path.join(data_dir, "df_activations_before.pkl"))
    df_beh_gc = pd.read_pickle(os.path.join(data_dir, "df_behavior_goal_conditioned.pkl"))
    df_act_gc = pd.read_pickle(os.path.join(data_dir, "df_activations_goal_conditioned.pkl"))
    print(f"  Random-walk probe : {len(df_beh_rw):,} steps")
    print(f"  Goal-directed probe: {len(df_beh_gc):,} steps")
    print("Computing rate maps...")
    rm_rw, occ_rw = build_rate_maps(df_beh_rw, df_act_rw, m, hidden_dim)
    rm_gc, occ_gc = build_rate_maps(df_beh_gc, df_act_gc, m, hidden_dim)
    skaggs_rw = compute_skaggs(rm_rw, occ_rw)
    skaggs_gc = compute_skaggs(rm_gc, occ_gc)
    pc_rw = (skaggs_rw >= args.skaggs_threshold).sum()
    pc_gc = (skaggs_gc >= args.skaggs_threshold).sum()
    print(f"  Skaggs random-walk   : mean={skaggs_rw.mean():.3f}, place cells={pc_rw}/{hidden_dim}")
    print(f"  Skaggs goal-directed : mean={skaggs_gc.mean():.3f}, place cells={pc_gc}/{hidden_dim}")
    top_idx_100 = np.argsort(skaggs_gc)[::-1][:100]
    top_idx = top_idx_100[:args.n_cells]
    print("\nFigure 0a: 100-cell grid (random walk probe)...")
    fig_100cells(
        rm_rw, skaggs_rw, top_idx_100, reward_bin,
        save_path=os.path.join(fig_dir, "100cells_randomwalk.png"),
        no_show=args.no_show,
    )
    print("Figure 0b: 100-cell grid (goal-directed probe)...")
    fig_100cells(
        rm_gc, skaggs_gc, top_idx_100, reward_bin,
        save_path=os.path.join(fig_dir, "100cells_goaldirected.png"),
        no_show=args.no_show,
    )
    print("\nFigure 1: Cell comparison (random walk vs goal-directed)...")
    fig_comparison(
        rm_rw, skaggs_rw, rm_gc, skaggs_gc,
        top_idx=top_idx,
        reward_bin=reward_bin,
        save_path=os.path.join(fig_dir, "goal_directed_comparison.png"),
        no_show=args.no_show,
    )
    print("Figure 2: Trajectory density...")
    fig_occupancy(
        occ_rw, occ_gc,
        reward_bin=reward_bin,
        save_path=os.path.join(fig_dir, "occupancy_comparison.png"),
        no_show=args.no_show,
    )
    print("Figure 3: Over-representation index...")
    fig_ori(
        rm_rw, skaggs_rw, rm_gc, skaggs_gc,
        reward_pos=reward_pos,
        threshold=args.skaggs_threshold,
        save_path=os.path.join(fig_dir, "ori_analysis.png"),
        no_show=args.no_show,
    )
    ori_rw = np.where(skaggs_rw >= args.skaggs_threshold)[0]
    ori_gc = np.where(skaggs_gc >= args.skaggs_threshold)[0]
    d_ori_rw = np.linalg.norm(np.array([field_center_of_mass(rm_rw[n]) for n in ori_rw]) - reward_pos, axis=1) if len(ori_rw) else np.array([])
    d_ori_gc = np.linalg.norm(np.array([field_center_of_mass(rm_gc[n]) for n in ori_gc]) - reward_pos, axis=1) if len(ori_gc) else np.array([])
    ori_record = {
        "test": "ori_random_walk_vs_goal_directed",
        "skaggs_threshold": float(args.skaggs_threshold),
        "n_place_random_walk": int(len(ori_rw)),
        "n_place_goal_directed": int(len(ori_gc)),
        "mean_dist_random_walk": float(d_ori_rw.mean()) if len(d_ori_rw) else None,
        "mean_dist_goal_directed": float(d_ori_gc.mean()) if len(d_ori_gc) else None,
    }
    try:
        from scipy.stats import mannwhitneyu
        if len(d_ori_rw) >= 2 and len(d_ori_gc) >= 2:
            u, p_mw = mannwhitneyu(d_ori_gc, d_ori_rw, alternative="less")
            print(f"  Mann-Whitney (goal-directed < random-walk distance-to-reward): U={u:.0f}  p={p_mw:.2e}")
            ori_record["mannwhitney_u"] = float(u)
            ori_record["mannwhitney_p_less"] = float(p_mw)
    except Exception as e:
        print(f"  (scipy mannwhitneyu unavailable: {e})")
    append_stat(args.run_dir, "plot_goal_directed_maps.py", ori_record)

    print("Figure 4: Per-cycle progression...")
    fig_cycle_progression(
        data_dir=data_dir,
        top_idx_gc=top_idx,
        reward_bin=reward_bin,
        m=m,
        hidden_dim=hidden_dim,
        save_path=os.path.join(fig_dir, "cycle_progression_goaltop.png"),
        no_show=args.no_show,
    )
    print("Figure 5: Drift across cycles...")
    fig_drift_across_cycles(
        data_dir=data_dir,
        reward_pos=reward_pos,
        m=m, hidden_dim=hidden_dim,
        skaggs_threshold=args.skaggs_threshold,
        save_path=os.path.join(fig_dir, "drift_across_cycles.png"),
        no_show=args.no_show,
    )
    print("Figure 6: ORI across cycles...")
    fig_ori_across_cycles(
        data_dir=data_dir,
        reward_pos=reward_pos,
        m=m, hidden_dim=hidden_dim,
        skaggs_threshold=args.skaggs_threshold,
        save_path=os.path.join(fig_dir, "ori_across_cycles.png"),
        no_show=args.no_show,
    )
    print("Figure 7: 100 random active neurons — random walk probe...")
    fig_100cells_random(
        rm_rw, reward_bin,
        mean_rate_threshold=0.01,
        save_path=os.path.join(fig_dir, "100cells_randomwalk_random.png"),
        no_show=args.no_show,
    )
    print("Figure 8: 100 random active neurons — goal-directed probe...")
    fig_100cells_random(
        rm_gc, reward_bin,
        mean_rate_threshold=0.01,
        save_path=os.path.join(fig_dir, "100cells_goaldirected_random.png"),
        no_show=args.no_show,
    )
    #per-cycle data (random-walk = prediction, goal = RL)
    print("\nLoading per-cycle probe data...")
    rw_cycles = _load_cycle_rate_maps(data_dir, m, hidden_dim, probe_type="rw")
    goal_cycles = _load_cycle_rate_maps(data_dir, m, hidden_dim, probe_type="goal")
    print("Figure 9: Prediction vs RL across cycles (per neuron)...")
    for nid in top_idx_100[:args.n_example_neurons]:
        fig_pred_vs_rl_across_cycles(
            rw_cycles, goal_cycles, int(nid), reward_bin,
            save_path=os.path.join(fig_dir, f"pred_vs_rl_cycles_neuron_{int(nid)}.png"),
            no_show=args.no_show,
        )
    if args.panel_neuron is not None:
        print(f"Figure 9b: Pred-vs-RL panel figure (neuron {args.panel_neuron})...")
        fig_pred_vs_rl_across_cycles(
            rw_cycles, goal_cycles, int(args.panel_neuron), reward_bin,
            save_path=os.path.join(fig_dir, f"pred_vs_rl_panel_neuron_{int(args.panel_neuron)}.png"),
            no_show=args.no_show,
        )

    print("Figure 10: Drift matrix (10x10)...")
    fig_drift_matrix(
        goal_cycles, skaggs_gc,
        n_cells=args.drift_cells,
        save_path=os.path.join(fig_dir, "drift_matrix.png"),
        no_show=args.no_show,
    )
    print("Figure 10b: Drift cycle-to-cycle matrix (annotated)...")
    fig_drift_rdm(
        goal_cycles, skaggs_gc,
        n_cells=args.drift_cells,
        save_path=os.path.join(fig_dir, "drift_matrix_numbers.png"),
        no_show=args.no_show,
    )
    print("Figure 10c: Drift 10x10 matrix (cycle x {prediction, RL})...")
    fig_drift_rdm_10(
        rw_cycles, goal_cycles, skaggs_gc,
        n_cells=args.drift_cells,
        save_path=os.path.join(fig_dir, "drift_matrix_10x10.png"),
        no_show=args.no_show,
    )
    # clean re-render of cognitive maps from saved data
    print("Figure 11: Clean cognitive maps (phase1 = before RL, phase2 = after RL, each cycle)...")
    # phase1 = before RL (random-walk probe)
    fig_100cells(
        rm_rw, skaggs_rw, top_idx_100, reward_bin,
        save_path=os.path.join(fig_dir, "phase1_cognitive_map.png"),
        no_show=args.no_show,
    )
    # phase2 = after RL (final, fully-consolidated random-walk probe)
    after_beh = os.path.join(data_dir, "df_behavior_after.pkl")
    after_act = os.path.join(data_dir, "df_activations_after.pkl")
    if os.path.exists(after_beh) and os.path.exists(after_act):
        rm_after, occ_after = build_rate_maps(
            pd.read_pickle(after_beh), pd.read_pickle(after_act), m, hidden_dim)
        skaggs_after = compute_skaggs(rm_after, occ_after)
        fig_100cells(
            rm_after, skaggs_after, top_idx_100, reward_bin,
            save_path=os.path.join(fig_dir, "phase2_cognitive_map.png"),
            no_show=args.no_show,
        )
    else:
        print("  (no df_*_after.pkl found; skipping phase2 cognitive map)")
    # intermediate snapshots per cycle
    for lab in sorted(rw_cycles.keys(), key=_cycle_num):
        rm_c, _, sk_c = rw_cycles[lab]
        fig_100cells(
            rm_c, sk_c, top_idx_100, reward_bin,
            save_path=os.path.join(fig_dir, f"{lab}_cognitive_map.png"),
            no_show=args.no_show,
        )

    report_path = write_stats_report(args.run_dir)
    print(f"Wrote statistics report: {report_path}")

    print("\nAll done. Figures saved to:", fig_dir)

if __name__ == "__main__":
    main()
