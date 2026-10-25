"""Panel b: egocentric FoV arena + RGB observation strip."""
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from utils import (
    DIST_RAY_COLOR,
    FS_BASE,
    FS_SMALL,
    FS_TICK,
    GRID_CENTERS,
    HDS,
    M,
    OBJECT_TYPE_COLORS,
    apply_nature_style,
    save_figure,
)

apply_nature_style()

FOV_EXAMPLE_POSES = [
    ((16, 16), 0, "open field"),
    ((3, 16), 0, "wall ahead"),
    ((16, 16), 2, "heading north"),
    ((28, 28), 4, "corner"),
    ((10, 8), 1, "object in view"),
    ((20, 5), 6, "near boundary"),
]


def build_v10_fov_env(seed: int = 42):
    """Build the schematic arena for this panel: wall-mounted landmark objects plus the agent's field-of-view cells. Returns (env, agent, fov_cells)."""
    from ratinabox import utils
    from ratinabox.Agent import Agent
    from ratinabox.Environment import Environment
    from ratinabox.Neurons import FieldOfViewOVCs

    random.seed(seed)
    np.random.seed(seed)

    class WallFixedFOVc(FieldOfViewOVCs):
        default_params = FieldOfViewOVCs.default_params.copy()

        def __init__(self, agent, params=None):
            super().__init__(agent, params or {})
            self.dists = []

        def ray_distances_to_walls(self, agent_pos, head_direction, thetas, ray_length, walls):
            heading = np.arctan2(head_direction[1], head_direction[0])
            ends = np.stack(
                [
                    agent_pos + ray_length * np.array([np.cos(heading + th), np.sin(heading + th)])
                    for th in thetas
                ],
                axis=0,
            )
            starts = np.tile(agent_pos[np.newaxis, :], (len(thetas), 1))
            segs = np.zeros((len(thetas), 2, 2))
            segs[:, 0, :], segs[:, 1, :] = starts, ends
            intercepts = utils.vector_intercepts(segs, walls, return_collisions=False)
            la, lb = intercepts[..., 0], intercepts[..., 1]
            valid = (la > 0) & (la < 1) & (lb > 0) & (lb < 1)
            la_valid = np.where(valid, la, np.inf)
            return np.minimum(la_valid.min(axis=1) * ray_length, ray_length)

        def update_cell_locations(self):
            thetas = self.Agent.Neurons[0].tuning_angles
            self.dists = self.ray_distances_to_walls(
                self.Agent.pos,
                self.Agent.head_direction,
                thetas,
                100.0,
                np.array(env.walls),
            )
            s = self.dists * 16.5
            n = len(thetas)
            self.tuning_distances = np.ones(n) * 0.06 * s
            self.sigma_distances = np.ones(n) * 0.05
            self.sigma_angles = np.ones(n) * 0.83333333 / s

    env = Environment()
    for type_id in range(3):
        # Schematic: landmarks on all four walls only (none in the interior).
        # 12 per type → 3 per wall, evenly covering left/right/bottom/top.
        locations = []
        for wall in range(4):
            for t in np.random.uniform(0.05, 0.95, size=3):
                if wall == 0:    # left
                    locations.append([0.0, float(t)])
                elif wall == 1:  # right
                    locations.append([1.0, float(t)])
                elif wall == 2:  # bottom
                    locations.append([float(t), 0.0])
                else:            # top
                    locations.append([float(t), 1.0])
        for loc in locations:
            env.add_object(loc, type=type_id)

    agent = Agent(env)
    fov_params = {
        "distance_range": [0.01, 0.11],
        "spatial_resolution": 0.05,
        "cell_arrangement": "uniform_manifold",
        "angle_range": [0, 125],
    }
    fov_cells = [
        WallFixedFOVc(agent, params={**fov_params, "object_tuning_type": i}) for i in range(3)
    ]
    for cell in fov_cells:
        cell.tuning_angles /= 3

    return env, agent, fov_cells


def get_observation_vector(fov_cells) -> np.ndarray:
    """The agent's observation at its current pose: normalised FoV firing rates concatenated with the wall distances."""
    for cell in fov_cells:
        cell.update_cell_locations()
        cell.update()
    fr = np.concatenate([c.firingrate for c in fov_cells]) / 10.0
    dist = fov_cells[0].dists / 1.5
    return np.nan_to_num(np.concatenate([fr, dist]), nan=0.0, posinf=1.0, neginf=0.0)


# Display-only gain for object FoV rows (rates after /10 are typically << 1).
OBJECT_FOV_DISPLAY_GAIN = 6.0


def observation_rgb_image(obs: np.ndarray) -> np.ndarray:
    """4×6 RGB grid: 3 object-type FoV rows + 1 wall-distance row."""
    fr = obs[:18].reshape(3, 6)
    dist = obs[18:24]
    img = np.ones((4, 6, 3), dtype=float)
    for t in range(3):
        color = np.array(OBJECT_TYPE_COLORS[t])
        for i in range(6):
            v = float(np.clip(fr[t, i] * OBJECT_FOV_DISPLAY_GAIN, 0.0, 1.0))
            img[t, i] = (1.0 - v) * 0.97 + v * color
    dcolor = np.array(DIST_RAY_COLOR)
    dmax = max(float(dist.max()), 0.05)
    for i in range(6):
        v = float(np.clip(dist[i] / dmax, 0.0, 1.0))
        img[3, i] = (1.0 - v) * 0.97 + v * dcolor
    return img


def plot_observation_rgb(ax, obs: np.ndarray) -> None:
    """Draw one observation vector as a labelled image, three object-type rows over a wall row."""
    img = observation_rgb_image(obs)
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([f"φ{i}" for i in range(6)], fontsize=FS_SMALL)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["type 0", "type 1", "type 2", "walls"], fontsize=FS_SMALL)
    ax.set_xlabel("FoV cell", fontsize=FS_TICK)
    for y in (0.5, 1.5, 2.5):
        ax.axhline(y, color="0.75", lw=0.6)
    ax.axhline(2.5, color="0.35", lw=1.0)


def render_fov_pose(ax_arena, ax_obs, env, agent, fov_cells, coords, hd_idx, title: str) -> None:
    """Place the agent at one grid cell and heading, then draw the arena with its receptive fields beside the resulting observation."""
    agent.reset_history()
    for cell in fov_cells:
        cell.reset_history()
    agent.pos = GRID_CENTERS[coords[0], coords[1]].copy()
    agent.head_direction = HDS[hd_idx]
    agent.update()
    for cell in fov_cells:
        cell.update_cell_locations()
        cell.update()

    env.plot_environment(fig=ax_arena.figure, ax=ax_arena, plot_objects=True)
    for cell in fov_cells:
        cell.display_vector_cells(ax=ax_arena)
    ax_arena.plot(agent.pos[0], agent.pos[1], "ko", markersize=4, zorder=5)
    hd = agent.head_direction / (np.linalg.norm(agent.head_direction) + 1e-9)
    ax_arena.arrow(
        agent.pos[0], agent.pos[1], 0.04 * hd[0], 0.04 * hd[1],
        head_width=0.02, head_length=0.02, fc="k", ec="k", zorder=6,
    )
    ax_arena.set_title(title, fontsize=FS_TICK)
    ax_arena.set_aspect("equal")
    ax_arena.set_xticks([])
    ax_arena.set_yticks([])

    obs = get_observation_vector(fov_cells)
    plot_observation_rgb(ax_obs, obs)


def render_panel_b(stem_examples, stem_compact, compact_pose_idx: int = 0, seed: int = 42,
                   size_in: tuple[float, float] | None = None):
    """Render panel b: one arena/observation row per example pose, plus a single-pose compact version."""
    env, agent, fov_cells = build_v10_fov_env(seed=seed)

    legend_handles = [
        Patch(facecolor="none", edgecolor=OBJECT_TYPE_COLORS[0], label="landmark object type 0"),
        Patch(facecolor="none", edgecolor=OBJECT_TYPE_COLORS[1], label="landmark object type 1"),
        Patch(facecolor="none", edgecolor=OBJECT_TYPE_COLORS[2], label="landmark object type 2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="k",
               markersize=5, label="FoV receptive field (object-tuned)"),
    ]

    n = len(FOV_EXAMPLE_POSES)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 1.9 * n), squeeze=False)
    for row, (coords, hd_idx, title) in enumerate(FOV_EXAMPLE_POSES):
        render_fov_pose(axes[row, 0], axes[row, 1], env, agent, fov_cells, coords, hd_idx, title)
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=FS_SMALL, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    save_figure(fig, stem_examples)

    coords, hd_idx, title = FOV_EXAMPLE_POSES[compact_pose_idx]
    fig2, axes2 = plt.subplots(1, 2, figsize=size_in or (4.8, 2.0), squeeze=False,
                               gridspec_kw={"width_ratios": [1.15, 1.0]})
    render_fov_pose(axes2[0, 0], axes2[0, 1], env, agent, fov_cells, coords, hd_idx, title)
    # Two columns, inside the axes area: at ncol=2 below the figure the legend
    # was what set this panel's height, and at fontsize=5 it printed at 4.2 pt.
    fig2.legend(handles=legend_handles, loc="lower center", ncol=2,
                fontsize=FS_SMALL, frameon=False, bbox_to_anchor=(0.5, 0.0))
    # Top inset leaves room for the pose title above the arena; bottom for the
    # two-column legend, which is inside the canvas now rather than below it.
    fig2.tight_layout(rect=(0, 0.10, 1, 0.96), pad=0.3)
    return save_figure(fig2, stem_compact, exact_size=size_in is not None)
