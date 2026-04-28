#!pip install ratinabox
import argparse
import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import FieldOfViewOVCs
from ratinabox import utils
import warnings
from scipy.ndimage import gaussian_filter
import random
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train world model + RL controller and save checkpoints/artifacts."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="Results/ratinabox_v9")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--phase1_epochs", type=int, default=10)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--no_show", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


ARGS = parse_args()

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
RNN_RL_LEARNING_RATE = 5e-5
PRED_LOSS_WEIGHT = 0.03
PROBE_REWARD_POS = np.array([0.5, 0.5])   # fixed reward for goal-conditioned probe
SEED = int(ARGS.seed)
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

RUN_DIR = ARGS.output_dir if ARGS.run_name == "" else os.path.join(ARGS.output_dir, ARGS.run_name)
FIGURES_DIR = ensure_dir(os.path.join(RUN_DIR, "figures"))
CHECKPOINT_DIR = ensure_dir(os.path.join(RUN_DIR, "checkpoints"))
DATA_DIR = ensure_dir(os.path.join(RUN_DIR, "data"))

save_json(
    os.path.join(RUN_DIR, "config.json"),
    {
        "seed": SEED,
        "device": device,
        "output_dir": RUN_DIR,
        "n_trials": ARGS.n_trials,
        "timesteps": ARGS.timesteps,
        "phase1_epochs": ARGS.phase1_epochs,
        "max_episodes": ARGS.max_episodes,
        "max_steps": ARGS.max_steps,
        "checkpoint_every": ARGS.checkpoint_every,
        "rnn_rl_learning_rate_W_rec": 5e-7,
        "rnn_rl_learning_rate_W_in_act": 5e-6,
        "rnn_rl_learning_rate_W_out": 5e-5,
        "pred_loss_weight": PRED_LOSS_WEIGHT,
    },
)
print(f"Saving artifacts to: {RUN_DIR}")

m = 32
grid_centers = np.array([[(x/m, y/m) for y in range(m)] for x in range(m)])
hds = np.array([(1,0),(np.sqrt(2)/2,np.sqrt(2)/2),(0,1),(-np.sqrt(2)/2,np.sqrt(2)/2),(-1,0),(-np.sqrt(2)/2,-np.sqrt(2)/2),(0,-1),(np.sqrt(2)/2,-np.sqrt(2)/2)])
coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])

GOAL_RADIUS = 0.05
VISUAL_GOAL_R = 0.05

class WallFixedFOVc(FieldOfViewOVCs):
    def __init__(self, Agent, params={}):
        super().__init__(Agent,params)
        self.dists = []
    def ray_distances_to_walls(self, agent_pos, head_direction, thetas, ray_length, walls):
        n = len(thetas)
        heading = np.arctan2(head_direction[1], head_direction[0])
        ends = np.stack([agent_pos + ray_length * np.array([np.cos(heading + θ),np.sin(heading + θ)]) for θ in thetas], axis=0)
        starts = np.tile(agent_pos[np.newaxis, :], (n, 1))
        segs = np.zeros((n, 2, 2)); segs[:, 0, :], segs[:, 1, :] = starts, ends
        intercepts = utils.vector_intercepts(segs, walls, return_collisions=False)
        la, lb = intercepts[..., 0], intercepts[..., 1]
        valid = (la > 0) & (la < 1) & (lb > 0) & (lb < 1)
        la_valid = np.where(valid, la, np.inf)
        min_la = la_valid.min(axis=1)
        distances = np.minimum(min_la * ray_length, ray_length)
        return distances
    def update_cell_locations(self):
        thetas, n = self.Agent.Neurons[0].tuning_angles, len(self.Agent.Neurons[0].tuning_angles)
        self.dists = self.ray_distances_to_walls(self.Agent.pos, self.Agent.head_direction, thetas, 100.0, np.array(Env.walls))
        s = self.dists * 16.5
        self.tuning_distances, self.sigma_distances, self.sigma_angles = np.ones(n)*0.06*s, np.ones(n)*0.05, np.ones(n)*0.83333333/s

class NormReLU(nn.Module):
    def __init__(self, hidden_size, epsilon=1e-2):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(hidden_size) * -0.5)
        self.epsilon = epsilon
    def forward(self, x):
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.epsilon)
        return F.relu(x_norm + self.bias)

class NextStepRNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, lambda_b=0.1, eta_b=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_in = nn.Linear(obs_dim, hidden_dim, bias=False)
        self.W_act = nn.Linear(act_dim, hidden_dim, bias=False)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, obs_dim)

        self.lambda_b = lambda_b
        self.eta_b = eta_b
        self.c_readout = nn.Linear(hidden_dim, 1, bias=False)
        self.g_strength = nn.Parameter(torch.zeros(1))
        self.g = NormReLU(hidden_dim)

        self.init_weights()

    def init_weights(self):
        tau = 2.0
        k_in = 1.0 / np.sqrt(self.W_in.in_features)
        k_out = 1.0 / np.sqrt(self.hidden_dim)
        torch.nn.init.uniform_(self.W_in.weight, -k_in, k_in)
        torch.nn.init.uniform_(self.W_act.weight, -k_in, k_in)
        torch.nn.init.uniform_(self.W_out.weight, -k_out, k_out)
        W_rec_data = torch.empty(self.hidden_dim, self.hidden_dim)
        torch.nn.init.uniform_(W_rec_data, -k_out, k_out)
        self.W_rec.weight.data = W_rec_data + torch.eye(self.hidden_dim) * (1 - 1 / tau)

    def forward(self, obs_seq, act_seq):
        T, B, _ = obs_seq.size()
        h = torch.zeros(B, self.hidden_dim, device=obs_seq.device)
        b = torch.zeros(B, self.hidden_dim, device=obs_seq.device)
        outputs, hiddens = [], []
        for t in range(T):
            o_in, a_in, h_in = self.W_in(obs_seq[t]), self.W_act(act_seq[t]), self.W_rec(h)
            b = (1 - self.lambda_b) * b + self.eta_b * o_in
            inhib_readout = self.c_readout(h)
            denominator = 1 + F.softplus(self.g_strength) * inhib_readout
            numerator = h_in + o_in + a_in + b
            h = self.g(numerator / denominator)
            outputs.append(torch.sigmoid(self.W_out(h)))
            hiddens.append(h)
        return torch.stack(outputs), torch.stack(hiddens)

    def forward_step(self, obs, act, h, b):
        o_in, a_in, h_in = self.W_in(obs), self.W_act(act), self.W_rec(h)
        b = (1 - self.lambda_b) * b + self.eta_b * o_in
        inhib_readout = self.c_readout(h)
        denominator = 1 + F.softplus(self.g_strength) * inhib_readout
        numerator = h_in + o_in + a_in + b
        h = self.g(numerator / denominator)
        return h, b

class Controller(nn.Module):
    def __init__(self, hidden_dim, act_dim, num_heads=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.goal_encoder = nn.Linear(2, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.actor = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, act_dim))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None: torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, h, goal_vec):
        h = self.input_norm(h)
        g_emb = self.goal_encoder(goal_vec).unsqueeze(1)
        h_seq = h.unsqueeze(1)
        attn_out, _ = self.attention(h_seq, g_emb, g_emb)
        x = h_seq + attn_out
        x = x.squeeze(1)
        x = x + self.fc(x)
        logits = self.actor(x)
        logits = torch.clamp(logits, min=-20, max=20)
        value = self.critic(x)
        return logits, value

def extract_hidden_activity_frozen_weights(model, obs_seq, act_seq):
    model.eval()
    if obs_seq.dim() == 3 and obs_seq.size(0) != obs_seq.size(1):
        if obs_seq.size(1) > obs_seq.size(0):
            obs_seq = obs_seq.permute(1, 0, 2)
            act_seq = act_seq.permute(1, 0, 2)
    with torch.no_grad():
        pred, hiddens = model(obs_seq.to(device), act_seq.to(device))
    return pred.cpu(), hiddens.cpu().numpy()


def run_probe_walk(rnn, period, n_steps=50000, save_dir=None):
    """
    Run a random probe walk with the RNN in eval mode and save the results
    as long-format DataFrames that all downstream analysis scripts expect.

    Outputs (saved to save_dir if provided):
      df_behavior_{period}.pkl  -- columns: [timestep, x, y, theta]
      df_activations_{period}.pkl -- columns: [timestep, neuron_id, activity]
    """
    print(f"\n=== Probe Walk for '{period}' representations ===")
    rnn.eval()
    Ag.reset_history()
    coords = np.random.randint(2, m - 2, size=2)
    Ag.pos = grid_centers[coords[0], coords[1]]
    action_hd_idx = np.random.randint(8)
    Ag.head_direction = hds[action_hd_idx]

    h = torch.zeros(1, HIDDEN_DIM, device=device)
    b = torch.zeros(1, HIDDEN_DIM, device=device)

    beh_x, beh_y, beh_theta = [], [], []
    hidden_list = []

    for t in tqdm(range(n_steps), desc=f"Probe ({period})"):
        obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
        act_vec = torch.zeros(1, 8, device=device)
        act_vec[0, action_hd_idx] = 1.0

        with torch.no_grad():
            h, b = rnn.forward_step(obs, act_vec, h, b)

        beh_x.append(float(Ag.pos[0]))
        beh_y.append(float(Ag.pos[1]))
        beh_theta.append(float(np.arctan2(Ag.head_direction[1], Ag.head_direction[0])))
        hidden_list.append(h.detach().cpu().numpy()[0])   # shape (HIDDEN_DIM,)

        action_hd_idx = np.random.randint(8)
        coords = np.clip(coords + coord_updates[action_hd_idx], 1, m - 2)
        Ag.pos = grid_centers[coords[0], coords[1]]
        Ag.head_direction = hds[action_hd_idx]

    # --- Build behaviour DataFrame ---
    timesteps = np.arange(n_steps)
    df_behavior = pd.DataFrame({"timestep": timesteps, "x": beh_x, "y": beh_y, "theta": beh_theta})

    # --- Build activations DataFrame (vectorised, avoids 2.5M-row loop) ---
    hidden_arr = np.vstack(hidden_list)                        # (n_steps, HIDDEN_DIM)
    ts_repeated = np.repeat(timesteps, HIDDEN_DIM)             # [0,0,...,0, 1,1,...]
    nid_tiled   = np.tile(np.arange(HIDDEN_DIM), n_steps)     # [0,1,...,511, 0,1,...]
    act_flat    = hidden_arr.reshape(-1)                       # flattened activations
    df_activations = pd.DataFrame({"timestep": ts_repeated, "neuron_id": nid_tiled, "activity": act_flat})

    if save_dir is not None:
        beh_path = os.path.join(save_dir, f"df_behavior_{period}.pkl")
        act_path = os.path.join(save_dir, f"df_activations_{period}.pkl")
        df_behavior.to_pickle(beh_path)
        df_activations.to_pickle(act_path)
        print(f"  Saved behavior  : {beh_path}  ({len(df_behavior):,} rows)")
        print(f"  Saved activations: {act_path}  ({len(df_activations):,} rows)")

    return df_behavior, df_activations


def run_goal_conditioned_probe(rnn, ctrl, reward_pos, n_steps=50000, save_dir=None):
    """
    Goal-conditioned probe: run the RL controller with a fixed reward position.
    Uses 50 short episodes from random starts to ensure full spatial coverage.
    Records hidden states under active reward-seeking to expose RL-shaped representations.
    """
    print(f"\n=== Goal-Conditioned Probe (reward at {reward_pos}) ===")
    rnn.eval()
    ctrl.eval()

    goal_pos = np.asarray(reward_pos, dtype=float)
    n_episodes = 50
    steps_per_ep = n_steps // n_episodes

    beh_x, beh_y, beh_theta = [], [], []
    hidden_list = []
    total_steps = 0

    for ep in tqdm(range(n_episodes), desc="Goal-conditioned probe"):
        coords = np.random.randint(2, m - 2, size=2)
        Ag.pos = grid_centers[coords[0], coords[1]]
        action_hd_idx = np.random.randint(8)
        Ag.head_direction = hds[action_hd_idx]
        h = torch.zeros(1, HIDDEN_DIM, device=device)
        b = torch.zeros(1, HIDDEN_DIM, device=device)
        prev_act = torch.zeros(1, 8, device=device)

        with torch.no_grad():
            for _ in range(WARMUP_STEPS):
                obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
                widx = np.random.randint(8)
                wv = torch.zeros(1, 8, device=device); wv[0, widx] = 1.0
                h, b = rnn.forward_step(obs, wv, h, b)
                coords = np.clip(coords + coord_updates[widx], 1, m - 2)
                Ag.pos = grid_centers[coords[0], coords[1]]
                Ag.head_direction = hds[widx]
                prev_act = wv

        for t in range(steps_per_ep):
            obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
            goal_vec = torch.tensor(goal_pos - Ag.pos, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                h, b = rnn.forward_step(obs, prev_act, h, b)
                logits, _ = ctrl(h, goal_vec)
                action = torch.argmax(logits, dim=1)

            beh_x.append(float(Ag.pos[0]))
            beh_y.append(float(Ag.pos[1]))
            beh_theta.append(float(np.arctan2(Ag.head_direction[1], Ag.head_direction[0])))
            hidden_list.append(h.detach().cpu().numpy()[0])

            action_idx = action.item()
            coords = np.clip(coords + coord_updates[action_idx], 1, m - 2)
            Ag.pos = grid_centers[coords[0], coords[1]]
            Ag.head_direction = hds[action_idx]
            prev_act = torch.zeros(1, 8, device=device); prev_act[0, action_idx] = 1.0

            total_steps += 1
            if total_steps >= n_steps:
                break
        if total_steps >= n_steps:
            break

    n_rec = len(hidden_list)
    timesteps = np.arange(n_rec)
    df_behavior = pd.DataFrame({"timestep": timesteps, "x": beh_x, "y": beh_y, "theta": beh_theta})

    hidden_arr = np.vstack(hidden_list)
    ts_repeated = np.repeat(timesteps, HIDDEN_DIM)
    nid_tiled   = np.tile(np.arange(HIDDEN_DIM), n_rec)
    act_flat    = hidden_arr.reshape(-1)
    df_activations = pd.DataFrame({"timestep": ts_repeated, "neuron_id": nid_tiled, "activity": act_flat})

    if save_dir is not None:
        beh_path = os.path.join(save_dir, "df_behavior_goal_conditioned.pkl")
        act_path = os.path.join(save_dir, "df_activations_goal_conditioned.pkl")
        df_behavior.to_pickle(beh_path)
        df_activations.to_pickle(act_path)
        print(f"  Saved behavior  : {beh_path}  ({len(df_behavior):,} rows)")
        print(f"  Saved activations: {act_path}  ({len(df_activations):,} rows)")

    return df_behavior, df_activations


def compute_skaggs_report(df_behavior, df_activations, period, m=32,
                           skaggs_threshold=0.05, n_top=38,
                           save_dir=None, figures_dir=None):
    """
    Compute Skaggs information scores from probe walk DataFrames.
    Saves df_skaggs_{period}.pkl and a top-cells figure so place cell quality
    """
    print(f"\n=== Place cell quality report — '{period}' ===")
    n_steps = len(df_behavior)
    n_neurons = HIDDEN_DIM
    n_bins = m * m
    if len(df_activations) == n_steps * n_neurons:
        act_mat = df_activations["activity"].values.reshape(n_steps, n_neurons)
        neuron_ids = np.arange(n_neurons)
        beh_sorted = df_behavior.sort_values("timestep")
    else:
        act_pivot = df_activations.pivot(index="timestep", columns="neuron_id", values="activity").fillna(0)
        beh_sorted = df_behavior[df_behavior["timestep"].isin(act_pivot.index)].sort_values("timestep")
        act_mat = act_pivot.reindex(beh_sorted["timestep"]).values
        neuron_ids = act_pivot.columns.values
        n_steps = act_mat.shape[0]

    # Position → grid cell
    x_bin = np.clip((beh_sorted["x"].values * m).astype(int), 0, m - 1)
    y_bin = np.clip((beh_sorted["y"].values * m).astype(int), 0, m - 1)
    gc = x_bin * m + y_bin  # (T,)
    # Scatter-add activity into bins — fully vectorised
    bin_sum = np.zeros((n_bins, n_neurons), dtype=float)
    np.add.at(bin_sum, gc, act_mat)
    bin_count = np.bincount(gc, minlength=n_bins).astype(float)

    occ = bin_count / n_steps
    mean_rate_per_bin = np.divide(bin_sum, bin_count[:, None],
                                   out=np.zeros_like(bin_sum),
                                   where=bin_count[:, None] > 0)
    mean_rate = act_mat.mean(axis=0)  # (N,)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean_rate[None, :] > 1e-10,
                         mean_rate_per_bin / mean_rate[None, :], 0.0)
        log_ratio = np.where(ratio > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
    skaggs = (occ[:, None] * ratio * log_ratio).sum(axis=0)  # (N,)

    df_skaggs = pd.DataFrame({"neuron_id": neuron_ids, "skaggs_score": skaggs})
    n_pass = int((df_skaggs["skaggs_score"] >= skaggs_threshold).sum())
    print(f"  Mean Skaggs : {df_skaggs['skaggs_score'].mean():.4f}")
    print(f"  Cells >= {skaggs_threshold} (place cells) : {n_pass} / {n_neurons}")

    if save_dir is not None:
        pkl_path = os.path.join(save_dir, f"df_skaggs_{period}.pkl")
        df_skaggs.to_pickle(pkl_path)
        print(f"  Saved: {pkl_path}")

    # Top cells figure
    if figures_dir is not None:
        top_idx = df_skaggs.nlargest(n_top, "skaggs_score").index.values  # row indices = neuron ids
        top_act = act_mat[:, top_idx]  # (T, n_top)

        rate_maps = np.zeros((n_top, m, m), dtype=float)
        for b in range(n_bins):
            mask = gc == b
            if mask.any():
                rate_maps[:, b // m, b % m] = top_act[mask].mean(axis=0)

        n_cols = 8
        n_rows = int(np.ceil(n_top / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))
        axes_flat = np.array(axes).flatten()
        for i, nid in enumerate(top_idx):
            ax = axes_flat[i]
            rm = gaussian_filter(rate_maps[i], sigma=1.5)
            if rm.max() > 0:
                rm = (rm - rm.min()) / (rm.max() - rm.min())
            ax.imshow(rm, origin="lower", cmap="jet", vmin=0, vmax=1)
            score = df_skaggs.loc[nid, "skaggs_score"] if nid in df_skaggs.index else 0.0
            ax.set_title(f"{nid}  {score:.3f}", fontsize=6)
            ax.axis("off")
        for i in range(n_top, len(axes_flat)):
            axes_flat[i].axis("off")
        fig.suptitle(f"Top {n_top} by Skaggs — {period}  |  {n_pass}/{n_neurons} place cells", fontsize=9)
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"top_skaggs_{period}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved figure: {fig_path}")

    return df_skaggs


def moving_average(values, window):
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals
    window = max(1, min(int(window), vals.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(vals, kernel, mode="valid")


def save_pickle(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def save_checkpoint(
    filename,
    rnn,
    controller,
    optimizer_rnn=None,
    optimizer_controller=None,
    optimizer_rnn_rl=None,
    episode=None,
    history=None,
    extra_metrics=None,
):
    payload = {
        "seed": SEED,
        "episode": episode,
        "rnn_state_dict": rnn.state_dict(),
        "controller_state_dict": controller.state_dict(),
        "optimizer_rnn_state_dict": None if optimizer_rnn is None else optimizer_rnn.state_dict(),
        "optimizer_controller_state_dict": None if optimizer_controller is None else optimizer_controller.state_dict(),
        "optimizer_rnn_rl_state_dict": None if optimizer_rnn_rl is None else optimizer_rnn_rl.state_dict(),
        "history": history,
        "extra_metrics": extra_metrics,
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(payload, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

Env = Environment()
for type_id in range(3):
    locations = np.concatenate((np.round(np.random.uniform(low=0.0, high=1.0, size=(12, 1))), (np.random.uniform(low=0.0, high=1.0, size=(12, 1)))), axis=1)
    [Env.add_object(l, type=type_id) for l in locations]

Ag = Agent(Env)
fov_params = {"distance_range": [0.01, .11], "spatial_resolution": 0.05, "cell_arrangement": "uniform_manifold", "angle_range": [0, 125]}
fov_cells_all = [WallFixedFOVc(Ag, params={**fov_params, "object_tuning_type": i}) for i in range(3)]
for c in fov_cells_all: c.tuning_angles /= 3
def get_next_move(coords, hd_idx, m=32):
    wall_buffer = 1
    border_flags = [coords[0] >= m-wall_buffer, coords[1] >= m-wall_buffer, coords[0] <= wall_buffer, coords[1] <= wall_buffer]
    action_probs = np.roll(np.array([0.5, 0.15, 0.05, 0.04, 0.02, 0.04, 0.05, 0.15]), hd_idx)
    if np.any(border_flags):
        if border_flags[0]: action_probs[[0,1,7]] = 0
        if border_flags[1]: action_probs[[2,1,3]] = 0
        if border_flags[2]: action_probs[[4,3,5]] = 0
        if border_flags[3]: action_probs[[6,5,7]] = 0
        if action_probs.sum() > 0: action_probs /= action_probs.sum()
        else: action_probs = np.ones(8)/8
    next_action_idx = np.random.choice(8, p=action_probs)
    coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])
    next_coords = np.clip(coords + coord_updates[next_action_idx], 0, m-1)
    action_encoding = np.zeros(8); action_encoding[next_action_idx] = 1
    return next_coords, next_action_idx, action_encoding
def get_obs():
    [c.update_cell_locations() for c in fov_cells_all]; [c.update() for c in fov_cells_all]
    fr = np.concatenate([c.firingrate for c in fov_cells_all]) / 10.0
    dist = fov_cells_all[0].dists / 1.5
    obs = np.concatenate([fr, dist])
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

dummy = get_obs()
OBS_DIM = len(dummy)
ACT_DIM = 8
HIDDEN_DIM = 512
rnn_model = NextStepRNN(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(device)
controller = Controller(HIDDEN_DIM, ACT_DIM).to(device)
def visualize_place_fields(rnn, title="Latent Space", save_path=None):
    print(f"Generating {title}...")
    rnn.eval()
    Ag.reset_history()
    coords = np.random.randint(2, m-2, size=2)
    Ag.pos = grid_centers[coords[0], coords[1]]
    h = torch.zeros(1, HIDDEN_DIM).to(device)
    b = torch.zeros(1, HIDDEN_DIM).to(device)
    positions, activations = [], []
    for t in range(5000):
        obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
        action_idx = np.random.randint(8)
        act_vec = torch.zeros(1, 8).to(device); act_vec[0, action_idx] = 1.0
        with torch.no_grad():
            h, b = rnn.forward_step(obs, act_vec, h, b)
        positions.append(Ag.pos.copy())
        activations.append(h.detach().cpu().numpy()[0])
        coords = np.clip(coords + coord_updates[action_idx], 1, m-2)
        Ag.pos = grid_centers[coords[0], coords[1]]
        Ag.head_direction = hds[action_idx]
    positions = np.array(positions)
    activations = np.array(activations)
    nbins = 32
    activity_map = np.zeros((100, nbins, nbins), dtype=np.float32)
    counts = np.zeros((nbins, nbins), dtype=np.float32)

    for t in range(len(positions)):
        x = int(np.clip(positions[t, 0]*nbins, 0, nbins-1))
        y = int(np.clip(positions[t, 1]*nbins, 0, nbins-1))
        activity_map[:, y, x] += activations[t, :100]
        counts[y, x] += 1

    for i in range(100):
        activity_map[i] = np.divide(activity_map[i], counts, out=np.zeros_like(activity_map[i]), where=counts!=0)
    for i in range(100):
        smoothed = gaussian_filter(activity_map[i], sigma=1.5)
        if smoothed.max() > 0:
            activity_map[i] = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        else:
            activity_map[i] = smoothed

    fig = plt.figure(figsize=(15, 15), dpi=150)
    fig.suptitle(title, fontsize=16, y=0.98)

    for i in range(100):
        ax = fig.add_subplot(10, 10, i+1)
        ax.imshow(activity_map[i], origin='lower', cmap='jet', interpolation='gaussian', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if not ARGS.no_show:
        plt.show()
    plt.close(fig)

def controller_latent(controller, h, goal_vec):
    controller.eval()
    with torch.no_grad():
        h_n = controller.input_norm(h)
        g_emb = controller.goal_encoder(goal_vec).unsqueeze(1)
        h_seq = h_n.unsqueeze(1)
        attn_out, _ = controller.attention(h_seq, g_emb, g_emb)
        x = (h_seq + attn_out).squeeze(1)
        x = x + controller.fc(x)
    return x

def visualize_controller_fields(rnn, controller, title="Controller Latent Fields", goal_mode="fixed", save_path=None):
    print(f"Generating {title}...")
    rnn.eval()
    controller.eval()
    Ag.reset_history()
    coords = np.random.randint(2, m-2, size=2)
    Ag.pos = grid_centers[coords[0], coords[1]]
    Ag.head_direction = hds[np.random.randint(8)]
    h = torch.zeros(1, HIDDEN_DIM).to(device)
    b = torch.zeros(1, HIDDEN_DIM).to(device)
    prev_act = torch.zeros(1, 8).to(device)
    if goal_mode == "fixed":
        goal_grid = np.random.randint(2, m-2, size=2)
        goal_pos_fixed = grid_centers[goal_grid[0], goal_grid[1]]
    positions, activations = [], []

    for t in range(5000):
        obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
        action_idx = np.random.randint(8)
        act_vec = torch.zeros(1, 8).to(device); act_vec[0, action_idx] = 1.0
        with torch.no_grad():
            h, b = rnn.forward_step(obs, prev_act, h, b)
        if goal_mode == "random":
            gg = np.random.randint(2, m-2, size=2)
            goal_pos = grid_centers[gg[0], gg[1]]
            goal_vec = torch.tensor(goal_pos - Ag.pos, dtype=torch.float32).unsqueeze(0).to(device)
        elif goal_mode == "fixed":
            goal_vec = torch.tensor(goal_pos_fixed - Ag.pos, dtype=torch.float32).unsqueeze(0).to(device)
        elif goal_mode == "zero":
            goal_vec = torch.zeros(1, 2, device=device)
        else:
            raise ValueError("goal_mode must be one of: 'fixed', 'random', 'zero'")
        x = controller_latent(controller, h, goal_vec)
        positions.append(Ag.pos.copy())
        activations.append(x.detach().cpu().numpy()[0])
        coords = np.clip(coords + coord_updates[action_idx], 1, m-2)
        Ag.pos = grid_centers[coords[0], coords[1]]
        Ag.head_direction = hds[action_idx]
        prev_act = act_vec

    positions = np.array(positions)
    activations = np.array(activations)
    nbins = 32
    activity_map = np.zeros((100, nbins, nbins), dtype=np.float32)
    counts = np.zeros((nbins, nbins), dtype=np.float32)
    for t in range(len(positions)):
        xb = int(np.clip(positions[t, 0] * nbins, 0, nbins - 1))
        yb = int(np.clip(positions[t, 1] * nbins, 0, nbins - 1))
        activity_map[:, yb, xb] += np.abs(activations[t, :100])
        counts[yb, xb] += 1
    for i in range(100):
        activity_map[i] = np.divide(activity_map[i], counts, out=np.zeros_like(activity_map[i]), where=counts!=0)
    for i in range(100):
        smoothed = gaussian_filter(activity_map[i], sigma=1.5)
        if smoothed.max() > 0:
            activity_map[i] = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        else:
            activity_map[i] = smoothed
    fig = plt.figure(figsize=(15, 15), dpi=150)
    fig.suptitle(title, fontsize=16, y=0.98)
    for i in range(100):
        ax = fig.add_subplot(10, 10, i+1)
        ax.imshow(activity_map[i], origin='lower', cmap='jet', interpolation='gaussian', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if not ARGS.no_show:
        plt.show()
    plt.close(fig)

print("\n=== PHASE 1: Training World Model ===")
n_trials = int(ARGS.n_trials)
timesteps = int(ARGS.timesteps)
obs_data, act_data = np.zeros((n_trials, timesteps, OBS_DIM)), np.zeros((n_trials, timesteps, ACT_DIM))

print("Generating Data...")
for i in tqdm(range(n_trials)):
    [c.reset_history() for c in fov_cells_all]; Ag.reset_history()
    coords = np.random.randint(2, m-2, size=2)
    action_hd_idx = np.random.randint(8)
    Ag.pos = grid_centers[coords[0], coords[1]]
    Ag.head_direction = hds[action_hd_idx]
    for t in range(timesteps):
        obs = get_obs()
        obs_data[i, t] = obs
        coords, action_hd_idx, action_encoding = get_next_move(coords, action_hd_idx, m=m)
        act_data[i, t] = action_encoding
        Ag.update(forced_next_position=grid_centers[coords[0], coords[1]])
        Ag.head_direction = hds[action_hd_idx]
obs_seq = torch.tensor(obs_data[:, :-1], dtype=torch.float32)
act_seq = torch.tensor(act_data[:, :-1], dtype=torch.float32)
next_obs_seq = torch.tensor(obs_data[:, 1:], dtype=torch.float32)
dataset = TensorDataset(obs_seq, act_seq, next_obs_seq)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
optimizer_rnn = optim.RMSprop(rnn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()
rnn_model.train()
losses = []
phase1_epochs = int(ARGS.phase1_epochs)
print(f"Training RNN ({phase1_epochs} Epochs)...")
for epoch in range(phase1_epochs):
    epoch_loss = 0
    for o, a, target in loader:
        o, a, target = o.to(device).permute(1,0,2), a.to(device).permute(1,0,2), target.to(device).permute(1,0,2)
        pred, h_seq = rnn_model(o, a)
        mse_loss = criterion(pred, target)
        l1_loss = 1e-4 * torch.mean(torch.abs(h_seq))
        loss = mse_loss + l1_loss

        optimizer_rnn.zero_grad(); loss.backward(); optimizer_rnn.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(loader))
    if epoch % max(1, phase1_epochs // 5) == 0:
        print(f"Epoch {epoch}: Loss {losses[-1]:.5f}")

np.save(os.path.join(DATA_DIR, "phase1_losses.npy"), np.asarray(losses, dtype=float))
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Prediction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase1_prediction_loss_curve.png"), dpi=150)
if not ARGS.no_show:
    plt.show()
plt.close()

save_checkpoint(
    filename="phase1_final.pt",
    rnn=rnn_model,
    controller=controller,
    optimizer_rnn=optimizer_rnn,
    episode=0,
    extra_metrics={"phase1_final_loss": float(losses[-1]) if len(losses) > 0 else None},
)

# --- Probe walk BEFORE RL (uses Phase 1 weights) ---
_beh_before, _act_before = run_probe_walk(rnn_model, period="before", n_steps=50000, save_dir=DATA_DIR)
compute_skaggs_report(_beh_before, _act_before, period="before", save_dir=DATA_DIR, figures_dir=FIGURES_DIR)

visualize_place_fields(
    rnn_model,
    title="Phase 1: Cognitive Map",
    save_path=os.path.join(FIGURES_DIR, "phase1_cognitive_map.png"),
)

print("\n=== Extracting Hidden Activity for Analysis ===")
obs_sample = obs_seq[:10]
act_sample = act_seq[:10]
predictions, hidden_states = extract_hidden_activity_frozen_weights(rnn_model, obs_sample, act_sample)
print(f"Predictions shape: {predictions.shape}")
print(f"Hidden states shape: {hidden_states.shape}")
save_pickle(
    os.path.join(DATA_DIR, "hidden_states_phase1.pkl"),
    {
        "seed": SEED,
        "predictions": predictions.numpy(),
        "hidden_states": hidden_states,
        "obs_sample_shape": tuple(obs_sample.shape),
        "act_sample_shape": tuple(act_sample.shape),
    },
)

print("\n=== PHASE 2: RL Training (With RNN Gradient Flow at Reduced LR) ===")
rnn_model.train()
optimizer_controller = optim.Adam(controller.parameters(), lr=1e-4, eps=1e-5)
optimizer_rnn_rl = optim.Adam([
    {"params": rnn_model.W_rec.parameters(), "lr": 5e-7},# recurrent manifold
    {"params": rnn_model.W_in.parameters(),  "lr": 5e-6},# sensory input
    {"params": rnn_model.W_act.parameters(), "lr": 5e-6},# action input 
    {"params": rnn_model.W_out.parameters(), "lr": 5e-5},# predictive output
    {"params": list(rnn_model.c_readout.parameters()) +
               list(rnn_model.g.parameters()) +
               [rnn_model.g_strength],"lr": 5e-6},# gain control
], eps=1e-5)
MAX_EPISODES = int(ARGS.max_episodes)
MAX_STEPS = int(ARGS.max_steps)
GAMMA = 0.99
CLIP_EPS = 0.1
UPDATE_EPOCHS = 4
WARMUP_STEPS = 10
history = {'rewards': [], 'success': [], 'pred_loss': []}
for episode in tqdm(range(MAX_EPISODES)):
    coords = np.random.randint(2, m-2, size=2)
    Ag.pos = grid_centers[coords[0], coords[1]]
    Ag.head_direction = hds[np.random.randint(8)]
    goal_grid = np.random.randint(2, m-2, size=2)
    while np.linalg.norm(goal_grid - coords) < 5: goal_grid = np.random.randint(2, m-2, size=2)
    goal_pos = grid_centers[goal_grid[0], goal_grid[1]]
    h = torch.zeros(1, HIDDEN_DIM).to(device)
    b = torch.zeros(1, HIDDEN_DIM).to(device)
    prev_act = torch.zeros(1, 8).to(device)
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
            action_idx = np.random.randint(8)
            act_vec = torch.zeros(1, 8).to(device); act_vec[0, action_idx] = 1.0
            h, b = rnn_model.forward_step(obs, act_vec, h, b)
            coords = np.clip(coords + coord_updates[action_idx], 1, m-2)
            Ag.pos = grid_centers[coords[0], coords[1]]
            Ag.head_direction = hds[action_idx]
            prev_act = act_vec
    hs, goals, actions, log_probs, rewards, values, dones, obs_list, prev_acts, bs, next_obs_list = [], [], [], [], [], [], [], [], [], [], []
    ep_reward = 0; success = False
    prev_dist = np.linalg.norm(Ag.pos - goal_pos)

    for t in range(MAX_STEPS):
        obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
        goal_vec = torch.tensor(goal_pos - Ag.pos, dtype=torch.float32).unsqueeze(0).to(device)
        h, b = rnn_model.forward_step(obs, prev_act, h, b)

        logits, value = controller(h, goal_vec)
        dist = Categorical(logits=logits)
        action = dist.sample()
        new_coords = coords + coord_updates[action.item()]
        if (new_coords[0] < 1 or new_coords[0] >= m-1 or new_coords[1] < 1 or new_coords[1] >= m-1): new_coords = coords
        coords = new_coords
        Ag.pos = grid_centers[coords[0], coords[1]]
        Ag.head_direction = hds[action.item()]
        next_obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
        prev_act_old = prev_act.clone()
        prev_act = torch.zeros(1, 8).to(device); prev_act[0, action.item()] = 1.0
        curr_dist = np.linalg.norm(Ag.pos - goal_pos)
        r = 0; done = False
        if curr_dist < GOAL_RADIUS:
            r += 15.0; done = True; success = True
        else:
            progress = prev_dist - curr_dist
            r += progress * 20.0
            r -= 0.05
        prev_dist = curr_dist
        ep_reward += r

        hs.append(h); goals.append(goal_vec); actions.append(action)
        log_probs.append(dist.log_prob(action)); rewards.append(r); values.append(value); dones.append(done)
        obs_list.append(obs); prev_acts.append(prev_act_old); bs.append(b)
        next_obs_list.append(next_obs)
        if done: break
    history['rewards'].append(ep_reward)
    history['success'].append(1 if success else 0)
    if len(rewards) < 2:
        history['pred_loss'].append(np.nan)
        continue
    R = 0
    returns, advantages = [], []
    if not done:
        with torch.no_grad():
            _, next_val = controller(h, goal_vec)
            R = next_val.item()
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)
    b_values = torch.cat(values).squeeze().detach()
    b_log_probs = torch.stack(log_probs).squeeze().detach()
    advantages = returns - b_values
    if advantages.std() > 1e-5: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else: advantages = advantages - advantages.mean()

    b_hs = torch.stack(hs).squeeze(1)
    b_goals = torch.cat(goals)
    b_actions = torch.stack(actions).squeeze()
    b_obs = torch.cat(obs_list)
    b_prev_acts = torch.cat(prev_acts)
    b_bs = torch.stack(bs).squeeze(1)
    b_next_obs = torch.cat(next_obs_list)
    episode_pred_loss = np.nan
    for _ in range(UPDATE_EPOCHS):
        recomputed_hs = []
        h_temp = b_hs[0].unsqueeze(0).detach()
        b_temp = b_bs[0].unsqueeze(0).detach()
        for i in range(len(b_obs)):
            h_temp, b_temp = rnn_model.forward_step(b_obs[i:i+1], b_prev_acts[i:i+1], h_temp, b_temp)
            recomputed_hs.append(h_temp)
        recomputed_hs = torch.cat(recomputed_hs)
        logits, new_vals = controller(recomputed_hs, b_goals)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(b_actions)
        ratio = torch.exp(new_log_probs - b_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
        rl_loss = -torch.min(surr1, surr2).mean() + 0.5 * (new_vals.squeeze() - returns).pow(2).mean() - 0.02 * dist.entropy().mean()
        pred_next_obs = torch.sigmoid(rnn_model.W_out(recomputed_hs))
        pred_loss = criterion(pred_next_obs, b_next_obs)
        total_loss = rl_loss + PRED_LOSS_WEIGHT * pred_loss
        optimizer_controller.zero_grad()
        optimizer_rnn_rl.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 0.5)
        optimizer_controller.step()
        optimizer_rnn_rl.step()
        episode_pred_loss = float(pred_loss.item())

    history['pred_loss'].append(episode_pred_loss)

    if ARGS.checkpoint_every > 0 and (episode + 1) % int(ARGS.checkpoint_every) == 0:
        recent_window = min(10, len(history['rewards']))
        save_checkpoint(
            filename=f"phase2_ep_{episode + 1}.pt",
            rnn=rnn_model,
            controller=controller,
            optimizer_rnn=optimizer_rnn,
            optimizer_controller=optimizer_controller,
            optimizer_rnn_rl=optimizer_rnn_rl,
            episode=episode + 1,
            history=None,
            extra_metrics={
                "mean_reward_last_window": float(np.mean(history['rewards'][-recent_window:])),
                "mean_success_last_window": float(np.mean(history['success'][-recent_window:])),
            },
        )

visualize_place_fields(
    rnn_model,
    title="After RL: Cognitive Map (with gradient flow)",
    save_path=os.path.join(FIGURES_DIR, "phase2_cognitive_map.png"),
)
visualize_controller_fields(
    rnn_model,
    controller,
    title="After RL: Controller Latent Fields",
    goal_mode="fixed",
    save_path=os.path.join(FIGURES_DIR, "phase2_controller_latent_fields.png"),
)

reward_curve = np.asarray(history['rewards'], dtype=float)
success_curve = np.asarray(history['success'], dtype=float)
pred_loss_curve = np.asarray(history['pred_loss'], dtype=float)

smooth_window = min(20, max(1, len(reward_curve)))
reward_smooth = moving_average(reward_curve, smooth_window)
success_smooth = moving_average(success_curve, smooth_window)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
if reward_smooth.size > 0:
    plt.plot(np.arange(smooth_window - 1, smooth_window - 1 + reward_smooth.size), reward_smooth)
plt.title("RL Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
if success_smooth.size > 0:
    plt.plot(np.arange(smooth_window - 1, smooth_window - 1 + success_smooth.size), success_smooth, color='green')
plt.title("Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "phase2_learning_curves.png"), dpi=150)
if not ARGS.no_show:
    plt.show()
plt.close()

np.savez(
    os.path.join(RUN_DIR, "learning_curves.npz"),
    phase1_loss=np.asarray(losses, dtype=float),
    rl_reward=reward_curve,
    rl_success=success_curve,
    rl_pred_loss=pred_loss_curve,
)

recent_window = min(10, len(history['rewards']))
save_checkpoint(
    filename="phase2_final.pt",
    rnn=rnn_model,
    controller=controller,
    optimizer_rnn=optimizer_rnn,
    optimizer_controller=optimizer_controller,
    optimizer_rnn_rl=optimizer_rnn_rl,
    episode=MAX_EPISODES,
    history=history,
    extra_metrics={
        "phase1_final_loss": float(losses[-1]) if len(losses) > 0 else None,
        "phase2_mean_reward_last_window": float(np.mean(history['rewards'][-recent_window:])) if recent_window > 0 else None,
        "phase2_mean_success_last_window": float(np.mean(history['success'][-recent_window:])) if recent_window > 0 else None,
    },
)

# --- Probe walk AFTER RL (uses Phase 2 fine-tuned weights) ---
_beh_after, _act_after = run_probe_walk(rnn_model, period="after", n_steps=50000, save_dir=DATA_DIR)
compute_skaggs_report(_beh_after, _act_after, period="after", save_dir=DATA_DIR, figures_dir=FIGURES_DIR)

# --- Goal-conditioned probe AFTER RL: representations during active reward-seeking ---
_beh_gc, _act_gc = run_goal_conditioned_probe(rnn_model, controller, PROBE_REWARD_POS, n_steps=50000, save_dir=DATA_DIR)
compute_skaggs_report(_beh_gc, _act_gc, period="goal_conditioned", save_dir=DATA_DIR, figures_dir=FIGURES_DIR)

print("\nTesting...")
controller.eval()
rnn_model.eval()
test_goals = [np.array([0.5, 0.5]), np.array([0.75, 0.25])]
titles = ["Center Goal (0.5, 0.5)", "Generalization Goal (0.75, 0.25)"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
test_success_rates = []
for ax, t_goal, title in zip([ax1, ax2], test_goals, titles):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(title, fontsize=14); ax.set_facecolor('#e6e6e6')
    ax.add_patch(Circle(t_goal, VISUAL_GOAL_R, color='gold', zorder=10))
    successes = 0
    for i in range(15):
        coords = np.random.randint(2, m-2, size=2)
        while np.linalg.norm(grid_centers[coords[0], coords[1]] - t_goal) < 0.3: coords = np.random.randint(2, m-2, size=2)
        Ag.pos = grid_centers[coords[0], coords[1]]
        h = torch.zeros(1, HIDDEN_DIM).to(device)
        b = torch.zeros(1, HIDDEN_DIM).to(device)
        prev_act = torch.zeros(1, 8).to(device)
        with torch.no_grad():
            for _ in range(WARMUP_STEPS):
                obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
                action_idx = np.random.randint(8)
                act_vec = torch.zeros(1, 8).to(device); act_vec[0, action_idx] = 1.0
                h, b = rnn_model.forward_step(obs, act_vec, h, b)
                coords = np.clip(coords + coord_updates[action_idx], 1, m-2)
                Ag.pos = grid_centers[coords[0], coords[1]]
                Ag.head_direction = hds[action_idx]
                prev_act = act_vec
        path = [Ag.pos.copy()]
        for t in range(MAX_STEPS):
            obs = torch.tensor(get_obs(), dtype=torch.float32).unsqueeze(0).to(device)
            goal_vec = torch.tensor(t_goal - Ag.pos, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                h, b = rnn_model.forward_step(obs, prev_act, h, b)
                logits, _ = controller(h, goal_vec)
                action = torch.argmax(logits, dim=1)
            prev_act = torch.zeros(1, 8).to(device); prev_act[0, action.item()] = 1.0
            coords = np.clip(coords + coord_updates[action.item()], 1, m-2)
            Ag.pos = grid_centers[coords[0], coords[1]]
            path.append(Ag.pos.copy())
            if np.linalg.norm(Ag.pos - t_goal) < GOAL_RADIUS: successes += 1; break
        path = np.array(path)
        is_success = np.linalg.norm(path[-1] - t_goal) < GOAL_RADIUS
        color = 'green' if is_success else 'red'
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, alpha=0.8)
        ax.scatter(path[0, 0], path[0, 1], color='blue', s=40)
    test_success_rates.append(float(successes / 15.0))
    ax.text(0.05, 0.95, f"Success: {successes}/15", transform=ax.transAxes, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "test_trajectories.png"), dpi=150)
if not ARGS.no_show:
    plt.show()
plt.close(fig)

print("\n=== Final Hidden Activity Extraction for Analysis ===")
obs_sample_final = obs_seq[:50]
act_sample_final = act_seq[:50]
predictions_final, hidden_states_final = extract_hidden_activity_frozen_weights(rnn_model, obs_sample_final, act_sample_final)
print(f"Final Predictions shape: {predictions_final.shape}")
print(f"Final Hidden states shape: {hidden_states_final.shape}")
save_pickle(
    os.path.join(DATA_DIR, "hidden_states_final.pkl"),
    {
        "seed": SEED,
        "predictions": predictions_final.numpy(),
        "hidden_states": hidden_states_final,
        "obs_sample_shape": tuple(obs_sample_final.shape),
        "act_sample_shape": tuple(act_sample_final.shape),
    },
)

metrics_payload = {
    "seed": SEED,
    "phase1_final_loss": float(losses[-1]) if len(losses) > 0 else None,
    "phase2_final_reward": float(history['rewards'][-1]) if len(history['rewards']) > 0 else None,
    "phase2_final_success": float(history['success'][-1]) if len(history['success']) > 0 else None,
    "phase2_mean_reward": float(np.mean(history['rewards'])) if len(history['rewards']) > 0 else None,
    "phase2_mean_success": float(np.mean(history['success'])) if len(history['success']) > 0 else None,
    "phase2_mean_reward_last10": float(np.mean(history['rewards'][-10:])) if len(history['rewards']) > 0 else None,
    "phase2_mean_success_last10": float(np.mean(history['success'][-10:])) if len(history['success']) > 0 else None,
    "test_success_rates": test_success_rates,
    "final_hidden_shape": list(hidden_states_final.shape),
}
save_json(os.path.join(RUN_DIR, "metrics.json"), metrics_payload)
print(f"Saved run metrics: {os.path.join(RUN_DIR, 'metrics.json')}")