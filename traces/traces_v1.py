!pip install ratinabox
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import RMSprop
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Circle
from tqdm import tqdm
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import FieldOfViewBVCs, FieldOfViewOVCs
from ratinabox import utils
import ratinabox
import warnings
from scipy.ndimage import gaussian_filter
import random, math

warnings.filterwarnings('ignore')

# Seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class WallFixedFOVc(FieldOfViewOVCs):
    def __init__(self, Agent, params={}):
        super().__init__(Agent,params)
        self.history["tuning_distances"], self.history["sigma_distances"], self.history["sigma_angles"], self.history["dists"] = [], [], [], []

    def display_vector_cells(self, fig=None, ax=None, t=None, **kwargs):
        if t is None: t = self.Agent.history["t"][-1]
        t_id = np.argmin(np.abs(np.array(self.Agent.history["t"]) - t))
        if fig is None and ax is None: fig, ax = self.Agent.plot_trajectory(t_start=t - 10, t_end=t, **kwargs)
        pos = self.Agent.history["pos"][t_id]
        y_axis_wrt_agent, x_axis_wrt_agent, head_direction_angle = np.array([0, 1]), np.array([1,0]), 0.0
        if self.reference_frame == "egocentric":
            head_direction = self.Agent.history["head_direction"][t_id]
            head_direction_angle = (180 / np.pi) * ratinabox.utils.get_angle(head_direction)
            x_axis_wrt_agent, y_axis_wrt_agent = head_direction / np.linalg.norm(head_direction), utils.rotate(x_axis_wrt_agent, np.pi / 2)
        fr, tuning_distances, sigma_angles, sigma_distances, tuning_angles = \
            np.array(self.history["firingrate"][t_id]), np.array(self.history["tuning_distances"][t_id]), np.array(self.history["sigma_angles"][t_id]), np.array(self.history["sigma_distances"][t_id]), self.tuning_angles
        x, y = tuning_distances * np.cos(tuning_angles), tuning_distances * np.sin(tuning_angles)
        pos_of_cells = pos + np.outer(x, x_axis_wrt_agent) + np.outer(y, y_axis_wrt_agent)
        ww, hh, aa  = sigma_angles * tuning_distances, sigma_distances, 1.0 * head_direction_angle + tuning_angles * 180 / np.pi
        ec = EllipseCollection(ww,hh, aa, units='x', offsets=pos_of_cells, offset_transform=ax.transData, linewidth=0.5, edgecolor="dimgrey", zorder=2.1)
        if self.cell_colors is None: facecolor_array = np.tile(np.array(matplotlib.colors.to_rgba(self.color if self.color is not None else "C1")), (self.n, 1))
        else: facecolor_array = self.cell_colors.copy()
        facecolor_array[:, -1] = 0.7 * np.maximum(0, np.minimum(1, fr / (0.5 * self.max_fr)))
        ec.set_facecolors(facecolor_array)
        ax.add_collection(ec)
        return fig, ax

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

    def save_to_history(self):
        super().save_to_history()
        self.history["tuning_distances"].append(self.tuning_distances.copy())
        self.history["sigma_distances"].append(self.sigma_distances.copy())
        self.history["sigma_angles"].append(self.sigma_angles.copy())
        self.history["dists"].append(self.dists.copy()/np.sqrt(2))

    def reset_history(self):
        super().reset_history()
        self.history["dists"]=[]

class NormReLU(nn.Module):
    def __init__(self, hidden_size, epsilon=1e-2, noise_std=0.01):
        super().__init__()
        self.bias, self.epsilon, self.noise_std = nn.Parameter(torch.zeros(hidden_size)), epsilon, noise_std
    def forward(self, x):
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.epsilon)
        return F.relu(x_norm + self.bias + torch.randn_like(x) * self.noise_std if self.training else x_norm + self.bias)

class NextStepRNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, lambda_b=0.1, eta_b=0.1):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden_dim = obs_dim, act_dim, hidden_dim
        self.W_in = nn.Linear(obs_dim, hidden_dim, bias=False)
        self.W_act = nn.Linear(act_dim, hidden_dim, bias=False)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, obs_dim)

        self.lambda_b = lambda_b
        self.eta_b = eta_b

        self.c_readout = nn.Linear(hidden_dim, 1, bias=False)
        self.g_strength = nn.Parameter(torch.zeros(1))

        self.g = NormReLU(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.init_weights()

    def init_weights(self):
        tau, k_in, k_out = 2.0, 1.0 / np.sqrt(self.obs_dim + self.act_dim), 1.0 / np.sqrt(self.hidden_dim)
        init.uniform_(self.W_in.weight, -k_in, k_in)
        init.uniform_(self.W_act.weight, -k_in, k_in)
        init.uniform_(self.W_out.weight, -k_out, k_out)
        init.uniform_(self.c_readout.weight, -k_out, k_out)
        W_rec_data = torch.empty(self.hidden_dim, self.hidden_dim)
        init.uniform_(W_rec_data, -k_out, k_out)
        self.W_rec.weight.data = W_rec_data + torch.eye(self.hidden_dim) * (1 - 1 / tau)

    def forward(self, obs_seq, act_seq, return_hidden=False):
        obs_seq = self.dropout(obs_seq)
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

            if return_hidden:
                hiddens.append(h.detach().cpu())
            outputs.append(torch.sigmoid(self.W_out(h)))

        outputs = torch.stack(outputs)
        if return_hidden:
            return outputs, torch.stack(hiddens)
        return outputs

    def forward_step(self, obs, act, h_prev, b_prev):
        o_in, a_in, h_in = self.W_in(obs), self.W_act(act), self.W_rec(h_prev)

        b_new = (1 - self.lambda_b) * b_prev + self.eta_b * o_in
        inhib_readout = self.c_readout(h_prev)
        denominator = 1 + F.softplus(self.g_strength) * inhib_readout
        numerator = h_in + o_in + a_in + b_new
        h_new = self.g(numerator / denominator)

        return h_new, b_new

class SpatiallyAwareActor(nn.Module):
    def __init__(self, hidden_dim, act_dim, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.spatial_layer = nn.Linear(hidden_dim, 128)
        if use_attention:
            self.attention = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
            self.attention_norm = nn.LayerNorm(128)
        self.policy_layers = nn.ModuleList([
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        ])
        self.policy_output = nn.Linear(64, act_dim)
        self.dropout = nn.Dropout(0.1)
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(256),
            nn.LayerNorm(256),
            nn.LayerNorm(128)
        ])
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, h):
        batch_size = h.size(0)
        x = F.relu(self.spatial_layer(h))
        if self.use_attention and batch_size > 0:
            x_expanded = x.unsqueeze(1)
            attn_output, _ = self.attention(x_expanded, x_expanded, x_expanded)
            x = self.attention_norm(x + attn_output.squeeze(1))
        for layer, ln in zip(self.policy_layers[:3], self.ln_layers):
            residual = x if x.size(-1) == layer.out_features else None
            x = layer(x); x = ln(x); x = F.relu(x); x = self.dropout(x)
            if residual is not None: x = x + residual
        x = F.relu(self.policy_layers[3](x))
        logits = self.policy_output(x)
        return logits

class SpatiallyAwareCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.spatial_layer = nn.Linear(hidden_dim, 128)
        self.value_layers = nn.ModuleList([
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        ])
        self.value_output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(256),
            nn.LayerNorm(256),
            nn.LayerNorm(128)
        ])
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, h):
        x = F.relu(self.spatial_layer(h))
        for layer, ln in zip(self.value_layers[:3], self.ln_layers):
            residual = x if x.size(-1) == layer.out_features else None
            x = layer(x); x = ln(x); x = F.relu(x); x = self.dropout(x)
            if residual is not None: x = x + residual
        x = F.relu(self.value_layers[3](x))
        return self.value_output(x)

def get_next_move(coords, hd_idx, m=10):
    wall_buffer = 1
    border_flags = [coords[0] >= m-wall_buffer, coords[1] >= m-wall_buffer, coords[0] <= wall_buffer, coords[1] <= wall_buffer]
    action_probs = np.roll(np.array([0.5, 0.15, 0.05, 0.04, 0.02, 0.04, 0.05, 0.15]), hd_idx)
    action_options = np.array(['E','NE','N','NW','W','SW','S','SE'])
    action_idxs = np.arange(8)
    coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])
    if np.any(border_flags):
        if border_flags[0]: action_probs[np.isin(action_options, ['E', 'NE', 'SE'])] = 0
        if border_flags[1]: action_probs[np.isin(action_options, ['N', 'NE', 'NW'])] = 0
        if border_flags[2]: action_probs[np.isin(action_options, ['W', 'NW', 'SW'])] = 0
        if border_flags[3]: action_probs[np.isin(action_options, ['S', 'SE', 'SW'])] = 0
        action_probs /= action_probs.sum()
    next_action_idx = np.random.choice(action_idxs, p=action_probs)
    next_coords = coords + coord_updates[next_action_idx]
    action_encoding = np.zeros(8); action_encoding[next_action_idx] = 1
    return next_coords, next_action_idx, action_encoding

np.random.seed(SEED)
Env = Environment()
number_of_wall_objects = 12
for type_id in range(3):
    locations = np.concatenate((np.round(np.random.uniform(low=0.0, high=1.0, size=(number_of_wall_objects, 1))), (np.random.uniform(low=0.0, high=1.0, size=(number_of_wall_objects, 1)))), axis=1)
    locations = np.where(np.random.rand(len(locations), 1) < 0.5, locations, locations[:, ::-1])
    [Env.add_object(l, type=type_id) for l in locations]

Ag = Agent(Env)
fov_params = {"distance_range": [0.01, .11], "spatial_resolution": 0.05, "cell_arrangement": "uniform_manifold", "angle_range": [0, 125]}
fov_cells_all = [WallFixedFOVc(Ag, params={**fov_params, "object_tuning_type": i}) for i in range(3)]
for fov_cell in fov_cells_all: fov_cell.tuning_angles /= 3

m = 32
grid_centers = np.array([[(x/m, y/m) for y in range(m)] for x in range(m)])
hds = np.array([(1,0),(np.sqrt(2)/2,np.sqrt(2)/2),(0,1),(-np.sqrt(2)/2,np.sqrt(2)/2),(-1,0),(-np.sqrt(2)/2,-np.sqrt(2)/2),(0,-1),(np.sqrt(2)/2,-np.sqrt(2)/2)])

print("Generating Training Data...")
n_trials, timesteps = 1000, 100
act_array, obs_array = np.zeros((n_trials, timesteps, 8)), np.zeros((n_trials, timesteps, 24))
coords = np.random.randint(high=m - 2, low=1, size=(2))

for nt in tqdm(np.arange(n_trials)):
    [c.reset_history() for c in fov_cells_all]; Ag.reset_history()
    action_hd_idx = np.random.randint(high=8,low=0)
    Ag.pos, Ag.head_direction = np.array(grid_centers[coords[0],coords[1]], dtype=float), np.array(hds[action_hd_idx], dtype=float)
    actions_saved = []
    for t in range(int(timesteps)):
        coords, action_hd_idx, action_encoding = get_next_move(coords, action_hd_idx, m=m)
        Ag.update(forced_next_position=np.array(grid_centers[coords[0], coords[1]], dtype=float))
        Ag.head_direction = np.array(hds[action_hd_idx], dtype=float)
        actions_saved.append(action_encoding)
        [c.update_cell_locations() for c in fov_cells_all]; [c.update() for c in fov_cells_all]
    obs_array[nt] = np.concatenate([c.get_history_arrays()['firingrate'] for c in fov_cells_all] + [fov_cells_all[0].get_history_arrays()['dists']], axis=1)
    act_array[nt] = np.array(actions_saved, dtype=float)

print("Generating Test Data...")
tmp = 50000
[c.reset_history() for c in fov_cells_all]; Ag.reset_history()
action_hd_idx = np.random.randint(high=8,low=0)
Ag.pos, Ag.head_direction = np.array(grid_centers[coords[0],coords[1]], dtype=float), np.array(hds[action_hd_idx], dtype=float)
actions_saved = []
for t in tqdm(np.arange(int(tmp))):
    coords, action_hd_idx, action_encoding = get_next_move(coords, action_hd_idx, m=m)
    Ag.update(forced_next_position=np.array(grid_centers[coords[0], coords[1]], dtype=float))
    Ag.head_direction = np.array(hds[action_hd_idx], dtype=float)
    actions_saved.append(action_encoding)
    [c.update_cell_locations() for c in fov_cells_all]; [c.update() for c in fov_cells_all]

obs_array_test = np.concatenate([c.get_history_arrays()['firingrate'] for c in fov_cells_all] + [fov_cells_all[0].get_history_arrays()['dists']], axis=1)
act_array_test, positions = np.array(actions_saved, dtype=float), Ag.get_history_arrays()['pos']
obs_array_test= np.nan_to_num(obs_array_test, nan=0.0)
obs_seq_test, act_seq_test = torch.tensor(obs_array_test, dtype=torch.float32).unsqueeze(1), torch.tensor(act_array_test, dtype=torch.float32).unsqueeze(1)

def train_next_step_rnn(model, train_loader, val_loader, num_epochs, lr):
    model.to(device)
    optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.95, eps=1e-8)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []
    patience, halfing_counter, epochs_no_improve, best_val_loss = 15, 0, 0, float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for obs_seq, act_seq, next_obs_seq in train_loader:
            obs_seq, act_seq, next_obs_seq = [x.permute(1,0,2).to(device) for x in (obs_seq, act_seq, next_obs_seq)]
            optimizer.zero_grad()
            pred = model(obs_seq, act_seq)
            loss = loss_fn(pred, next_obs_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for obs_seq, act_seq, next_obs_seq in val_loader:
                obs_seq, act_seq, next_obs_seq = [x.permute(1,0,2).to(device) for x in (obs_seq, act_seq, next_obs_seq)]
                pred = model(obs_seq, act_seq)
                loss = loss_fn(pred, next_obs_seq)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.6f}, Val Loss={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss - 5e-5:
            best_val_loss, epochs_no_improve = avg_val_loss, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5
                print("Reducing learning rate to", pg['lr'])
                epochs_no_improve, halfing_counter = 0, halfing_counter + 1
        if halfing_counter >= 7:
            print("Stopping early.")
            break
    return train_losses, val_losses

def get_current_observation(agent, fov_cells_list):
    obs_parts = []
    for cells in fov_cells_list:
        cells.update_cell_locations()
        cells.update()
        obs_parts.append(cells.firingrate)
    obs_parts.append(fov_cells_list[0].dists / np.sqrt(2))
    return torch.tensor(np.nan_to_num(np.concatenate(obs_parts)), dtype=torch.float32).to(device)

def plot_latent_space_spatial_activity(model, obs_seq_data, act_seq_data, positions_data, title=""):
    print(f"Visualizing representations for: {title}")
    _, hidden_activity = extract_hidden_activity_frozen_weights(model, obs_seq_data[:-1], act_seq_data[:-1])
    hidden_dim, T, nbins = model.hidden_dim, hidden_activity.shape[0], 32
    activity_map, counts = np.zeros((hidden_dim, nbins, nbins)), np.zeros((nbins, nbins))
    for t in range(T):
        x_bin, y_bin = int(positions_data[t, 0] * nbins), int(positions_data[t, 1] * nbins)
        if 0 <= x_bin < nbins and 0 <= y_bin < nbins:
            activity_map[:, y_bin, x_bin] += hidden_activity[t]
            counts[y_bin, x_bin] += 1
    activity_map /= (counts[np.newaxis, :, :] + 1e-6)
    units_to_plot, grid_size = min(100, hidden_dim), int(np.sqrt(min(100, hidden_dim)))
    fig = plt.figure(figsize=(12, 12)); fig.suptitle(title, fontsize=16)
    for i in range(units_to_plot):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.imshow(gaussian_filter(activity_map[i], sigma=1.0), origin='lower', cmap='jet', interpolation='gaussian')
        ax.set(xticks=[], yticks=[])
    fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def plot_latent_representations_after_rl(model, hidden_activity_data, positions_data, title=""):
    print(f"Visualizing representations for: {title}")
    hidden_dim = model.hidden_dim
    T = hidden_activity_data.shape[0]
    nbins = 32
    activity_map = np.zeros((hidden_dim, nbins, nbins))
    counts = np.zeros((nbins, nbins))
    for t in range(T):
        if t < len(positions_data):
            x_bin = int(np.clip(positions_data[t, 0] * nbins, 0, nbins-1))
            y_bin = int(np.clip(positions_data[t, 1] * nbins, 0, nbins-1))
            activity_map[:, y_bin, x_bin] += np.abs(hidden_activity_data[t])
            counts[y_bin, x_bin] += 1
    for i in range(hidden_dim):
        activity_map[i] = np.divide(activity_map[i], counts, out=np.zeros_like(activity_map[i]), where=counts!=0)
    units_to_plot = min(100, hidden_dim); grid_size = 10
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15)); fig.suptitle(title, fontsize=16)
    for i in range(units_to_plot):
        row, col = i // grid_size, i % grid_size
        ax = axes[row, col]
        smoothed_activity = gaussian_filter(activity_map[i], sigma=1.5)
        if smoothed_activity.max() > 0:
            smoothed_activity = (smoothed_activity - smoothed_activity.min()) / (smoothed_activity.max() - smoothed_activity.min())
        ax.imshow(smoothed_activity, origin='lower', cmap='jet', interpolation='gaussian', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for i in range(units_to_plot, grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        axes[row, col].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def extract_hidden_activity_frozen_weights(model, obs_seq, act_seq):
    model.eval()
    with torch.no_grad():
        pred, hiddens = model(obs_seq.to(device), act_seq.to(device), return_hidden=True)
    return pred.cpu(), hiddens.cpu().numpy().squeeze(axis=1)

def compute_shaped_reward(current_pos,
                          goal_pos,
                          previous_distance=None,
                          step_count=0,
                          max_steps=500,
                          visited_positions=None,
                          action_history=None,
                          prev_coords=None,
                          current_coords=None,
                          goal_coords=None,
                          action_idx=None,
                          arena_m=32):
    distance_to_goal = np.linalg.norm(current_pos - goal_pos)

    if (current_coords is not None) and (goal_coords is not None):
        if (abs(current_coords[0]-goal_coords[0]) <= 1) and (abs(current_coords[1]-goal_coords[1]) <= 1):
            speed_bonus = max(0, (max_steps - step_count) / max_steps * 200.0)
            return 1000.0 + speed_bonus, distance_to_goal
    if distance_to_goal < (1.5 / arena_m):
        speed_bonus = max(0, (max_steps - step_count) / max_steps * 200.0)
        return 1000.0 + speed_bonus, distance_to_goal

    distance_reward = 120.0 * np.exp(-3.0 * distance_to_goal)

    shaping_k, alpha = 1.5, 3.0
    phi_curr = shaping_k * np.exp(-alpha * distance_to_goal)
    phi_prev = shaping_k * np.exp(-alpha * previous_distance) if previous_distance is not None else 0.0
    gamma_shaping = 0.99
    shaping_reward = gamma_shaping * phi_curr - phi_prev

    progress_reward = 0.0
    stagnation_penalty = 0.0
    if previous_distance is not None:
        progress = previous_distance - distance_to_goal
        if progress > 0:
            progress_reward = 240.0 * progress
        else:
            progress_reward = -30.0 * abs(progress)
        if progress < 1e-4:
            stagnation_penalty = -2.0

    heading_bonus = 0.0
    if (action_idx is not None) and (prev_coords is not None) and (goal_coords is not None):
        coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)], dtype=float)
        a_vec = coord_updates[action_idx]
        g_vec = (goal_coords - prev_coords).astype(float)
        if np.linalg.norm(g_vec) > 1e-6:
            a_unit = a_vec / (np.linalg.norm(a_vec) + 1e-8)
            g_unit = g_vec / (np.linalg.norm(g_vec) + 1e-8)
            cos_al = np.clip(np.dot(a_unit, g_unit), -1.0, 1.0)
            heading_bonus = 8.0 * max(0.0, cos_al)

    proximity_bonus = 320.0 * np.exp(-8.0 * distance_to_goal) if distance_to_goal < 0.3 else 0.0

    exploration_bonus = 0.0
    if visited_positions is not None:
        grid_pos = (int(current_pos[0] * 32), int(current_pos[1] * 32))
        if grid_pos not in visited_positions:
            exploration_bonus = 10.0
            visited_positions.add(grid_pos)

    diversity_bonus = 0.0
    if action_history is not None and len(action_history) >= 3:
        recent_actions = action_history[-3:]
        if len(set(recent_actions)) >= 2:
            diversity_bonus = 5.0

    time_penalty = -0.4 * (step_count / max_steps)

    total_reward = (distance_reward + shaping_reward + progress_reward + heading_bonus +
                    proximity_bonus + exploration_bonus + diversity_bonus +
                    time_penalty + stagnation_penalty)
    return total_reward, distance_to_goal

def get_intelligent_exploration_action(coords, hd_idx, goal_coords=None,
                                     exploration_rate=0.3, m=32, step_count=0,
                                     action_history=None, visited_positions=None):
    def _safe_normalize(p):
        p = np.asarray(p, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p[p < 0] = 0.0
        s = p.sum()
        if s <= 0.0:
            p = np.ones_like(p, dtype=np.float64) / len(p)
        else:
            p /= s
        residual = 1.0 - p.sum()
        if abs(residual) > 1e-12:
            idx = int(np.argmax(p)); p[idx] = max(0.0, p[idx] + residual)
            p = np.clip(p, 0.0, None); p /= p.sum()
        return p

    wall_buffer = 1
    border_flags = [
        coords[0] >= m-wall_buffer, coords[1] >= m-wall_buffer,
        coords[0] <= wall_buffer, coords[1] <= wall_buffer
    ]
    action_probs = np.ones(8, dtype=np.float64) * 0.125
    if goal_coords is not None:
        direction_to_goal = goal_coords - coords
        goal_distance = np.linalg.norm(direction_to_goal)
        if goal_distance > 0:
            goal_angle = np.arctan2(direction_to_goal[1], direction_to_goal[0])
            action_angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4,
                                      np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4], dtype=np.float64)
            angle_diffs = np.abs(np.mod(action_angles - goal_angle + np.pi, 2*np.pi) - np.pi)
            alignment_scores = np.exp(-2.0 * angle_diffs)
            goal_bias = 0.7 if goal_distance < 8 else (0.55 if goal_distance < 16 else 0.4)
            if np.random.random() > exploration_rate:
                action_probs = action_probs * 0.1 + alignment_scores * goal_bias
    if action_history is not None and len(action_history) >= 2:
        recent_action = action_history[-1]
        very_recent = action_history[-2]
        action_probs[recent_action] *= 0.6
        opposite_actions = {0:4, 1:5, 2:6, 3:7, 4:0, 5:1, 6:2, 7:3}
        if very_recent in opposite_actions:
            action_probs[opposite_actions[recent_action]] *= 0.25
    if visited_positions is not None and step_count > 20:
        coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])
        for i, update in enumerate(coord_updates):
            next_pos = coords + update
            next_grid = (int(next_pos[0] * 0.5), int(next_pos[1] * 0.5))
            if next_grid not in visited_positions:
                action_probs[i] *= 1.5
    action_options = np.array(['E','NE','N','NW','W','SW','S','SE'])
    coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])
    boundary_penalties = np.ones(8, dtype=np.float64)
    for i, update in enumerate(coord_updates):
        next_coords = coords + update
        if (next_coords[0] < wall_buffer or next_coords[0] >= m-wall_buffer or
            next_coords[1] < wall_buffer or next_coords[1] >= m-wall_buffer):
            boundary_penalties[i] = 0.0
        else:
            dist_to_edge_x = min(next_coords[0] - wall_buffer, m-wall_buffer-1-next_coords[0])
            dist_to_edge_y = min(next_coords[1] - wall_buffer, m-wall_buffer-1-next_coords[1])
            min_dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)
            if min_dist_to_edge < 5:
                boundary_penalties[i] = max(0.0, min_dist_to_edge / 5.0)
    action_probs *= boundary_penalties
    if action_probs.sum() <= 0.0:
        action_probs = np.ones(8, dtype=np.float64)
        if border_flags[0]: action_probs[np.isin(action_options, ['E','NE','SE'])] = 0.0
        if border_flags[1]: action_probs[np.isin(action_options, ['N','NE','NW'])] = 0.0
        if border_flags[2]: action_probs[np.isin(action_options, ['W','NW','SW'])] = 0.0
        if border_flags[3]: action_probs[np.isin(action_options, ['S','SE','SW'])] = 0.0
    action_probs = _safe_normalize(action_probs)
    next_action_idx = np.random.choice(8, p=action_probs)
    next_coords = coords + coord_updates[next_action_idx]
    return next_coords, next_action_idx

def border_action_mask(coords, m=32):
    mask = np.ones(8, dtype=np.float32)
    x, y = coords
    if x >= m-3: mask[[0,1,7]] = 0
    if y >= m-3: mask[[1,2,3]] = 0
    if x <= 2:   mask[[3,4,5]] = 0
    if y <= 2:   mask[[5,6,7]] = 0
    return mask

def apply_mask_to_logits(logits, mask):
    mask_t = torch.tensor(mask, dtype=logits.dtype, device=logits.device)
    neg_inf = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
    return torch.where(mask_t > 0, logits, neg_inf)

def train_fixed_spatial_rl(predictive_model, actor, critic,
                          total_timesteps=300000,
                          curriculum_stages=4,
                          gamma=0.99,
                          lambda_pred=0.2,
                          predictive_lr=1e-5,
                          actor_lr=5e-4,
                          critic_lr=5e-4,
                          entropy_start=0.02,
                          entropy_end=0.005,
                          reward_scale=0.05,
                          reward_clip=20.0,
                          value_coef=0.25):
    print(f"\n🚀 Starting FIXED Spatial RL Training (32×32, No Hint)")
    predictive_model.to(device).train()
    actor.to(device).train()
    critic.to(device).train()

    predictive_optimizer = optim.AdamW(predictive_model.parameters(), lr=predictive_lr, weight_decay=1e-5)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=actor_lr, weight_decay=1e-4)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=1e-4)

    actor_scheduler = optim.lr_scheduler.OneCycleLR(actor_optimizer, max_lr=actor_lr*2, total_steps=max(1, total_timesteps//10), pct_start=0.1, anneal_strategy='cos')
    critic_scheduler = optim.lr_scheduler.OneCycleLR(critic_optimizer, max_lr=critic_lr*2, total_steps=max(1, total_timesteps//10), pct_start=0.1, anneal_strategy='cos')

    prediction_loss_fn = nn.MSELoss()

    m_grid = 32
    coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])
    fov_cells = fov_cells_all

    stage_duration = total_timesteps // curriculum_stages
    current_stage = 0
    goal_timeouts = [600, 500, 400, 320]
    min_goal_distances = [5, 7, 9, 11]
    exploration_rates = [0.65, 0.5, 0.35, 0.25]
    goal_reward_multipliers = [0.6, 0.8, 0.95, 1.0]

    current_coords = np.random.randint(low=6, high=m_grid-6, size=2)
    current_hd_idx = np.random.randint(8)
    Ag.pos = grid_centers[current_coords[0], current_coords[1]]
    Ag.head_direction = hds[current_hd_idx]
    h = torch.zeros(1, predictive_model.hidden_dim, device=device)
    b = torch.zeros(1, predictive_model.hidden_dim, device=device)

    phase = "exploration"
    steps_until_goal = np.random.randint(120, 241)
    goal_coords = None
    goal_pos = None
    previous_distance = None
    steps_since_goal_introduced = 0
    repositioning_steps_left = 0
    step_count = 0

    phase_history, reward_history, position_history = [], [], []
    hidden_states_history, prediction_losses, rl_losses = [], [], []
    goals_reached = 0; goals_timed_out = 0
    episode_rewards = []; current_episode_reward = 0
    visited_positions = set(); action_history = []; success_history = []
    experience_buffer = []; buffer_size = 12000
    stage_goals_reached = 0; stage_goals_attempted = 0
    recent_success_rates = []
    adv_norm_buffer = []

    print(f"Starting Stage 1/{curriculum_stages}: Timeout={goal_timeouts[0]}, MinDist={min_goal_distances[0]}, Exploration={exploration_rates[0]}")

    for timestep in tqdm(range(total_timesteps), desc="Fixed Spatial RL Training"):
        frac = timestep / max(1, total_timesteps-1)
        entropy_coef = entropy_start + (entropy_end - entropy_start) * frac

        new_stage = min(timestep // stage_duration, curriculum_stages - 1)
        if new_stage != current_stage:
            current_stage = new_stage
            print(f"\n📈 Advancing to Stage {current_stage + 1}/{curriculum_stages}")
            stage_goals_reached = 0; stage_goals_attempted = 0

        phase_history.append(phase); position_history.append(Ag.pos.copy())
        obs = get_current_observation(Ag, fov_cells).unsqueeze(0)

        if phase == "exploration":
            current_coords, action_idx = get_intelligent_exploration_action(
                current_coords, current_hd_idx, goal_coords,
                exploration_rates[current_stage], m_grid, step_count,
                action_history, visited_positions
            )
            current_hd_idx = action_idx
            action_history.append(action_idx); action_history = action_history[-10:]

            action_encoding = torch.zeros(1, predictive_model.act_dim, device=device); action_encoding[0, action_idx] = 1.0

            current_coords = np.clip(current_coords, 3, m_grid-4)
            Ag.pos = grid_centers[current_coords[0], current_coords[1]]
            Ag.head_direction = hds[current_hd_idx]
            h, b = predictive_model.forward_step(obs, action_encoding, h, b)
            hidden_states_history.append(h.squeeze().cpu().detach().numpy())

            next_obs = get_current_observation(Ag, fov_cells).unsqueeze(0)
            predicted_next_obs = torch.sigmoid(predictive_model.W_out(h))
            pred_loss = prediction_loss_fn(predicted_next_obs, next_obs)

            predictive_optimizer.zero_grad(); pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictive_model.parameters(), max_norm=1.0)
            predictive_optimizer.step()

            prediction_losses.append(pred_loss.item()); rl_losses.append(0.0); reward_history.append(0.0)

            steps_until_goal -= 1
            if steps_until_goal <= 0:
                phase = "goal_directed"
                min_distance = min_goal_distances[current_stage]; max_attempts = 200; attempts = 0
                while attempts < max_attempts:
                    goal_coords = np.random.randint(low=6, high=m_grid-6, size=2)
                    distance = np.linalg.norm(goal_coords - current_coords)
                    if min_distance <= distance <= 24: break
                    attempts += 1
                if attempts >= max_attempts:
                    angle = np.random.uniform(0, 2*np.pi); distance = min_distance + np.random.uniform(0, 6)
                    goal_coords = current_coords + distance * np.array([np.cos(angle), np.sin(angle)])
                    goal_coords = np.clip(goal_coords, 6, m_grid-6).astype(int)
                goal_pos = grid_centers[goal_coords[0], goal_coords[1]]
                previous_distance = np.linalg.norm(Ag.pos - goal_pos)
                steps_since_goal_introduced = 0; step_count = 0; current_episode_reward = 0
                stage_goals_attempted += 1
                visited_positions.clear(); action_history = []
                print(f"\n🎯 Step {timestep}: Stage {current_stage+1} Goal at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}), Distance: {previous_distance:.2f}")

        elif phase == "goal_directed":
            h_detached = h.detach()
            b_detached = b.detach()
            base_logits = actor(h_detached)
            mask = border_action_mask(current_coords, m_grid)
            temperature = max(0.15, 1.0 - 0.9 * (timestep / total_timesteps))
            base_logits = apply_mask_to_logits(base_logits / temperature, mask)
            dist = Categorical(logits=base_logits)
            eps_floor = 0.12
            epsilon = max(eps_floor, exploration_rates[current_stage] * (1 - 0.8 * timestep / total_timesteps))

            if np.random.random() < epsilon:
                prev_for_step = current_coords.copy()
                current_coords, action_idx_np = get_intelligent_exploration_action(
                    current_coords, current_hd_idx, goal_coords, 0.2, m_grid,
                    step_count, action_history, visited_positions
                )
                action_idx = torch.tensor(action_idx_np, device=device)
                action_log_prob = dist.log_prob(action_idx)
            else:
                prev_for_step = current_coords.copy()
                action_idx = dist.sample()
                action_log_prob = dist.log_prob(action_idx)

            entropy = dist.entropy()
            action_encoding = torch.zeros(1, predictive_model.act_dim, device=device); action_encoding[0, action_idx.item()] = 1.0
            new_coords = prev_for_step + np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])[action_idx.item()]
            if (new_coords[0] < 3 or new_coords[0] >= m_grid-3 or new_coords[1] < 3 or new_coords[1] >= m_grid-3):
                new_coords = prev_for_step
            current_coords = new_coords
            current_hd_idx = action_idx.item()
            Ag.pos = grid_centers[current_coords[0], current_coords[1]]
            Ag.head_direction = hds[current_hd_idx]
            action_history.append(action_idx.item()); action_history = action_history[-10:]
            h, b = predictive_model.forward_step(obs, action_encoding, h, b)
            hidden_states_history.append(h.squeeze().cpu().detach().numpy())

            reward, distance_to_goal = compute_shaped_reward(
                Ag.pos, goal_pos, previous_distance, step_count,
                goal_timeouts[current_stage], visited_positions, action_history,
                prev_coords=prev_for_step, current_coords=current_coords,
                goal_coords=goal_coords, action_idx=action_idx.item(), arena_m=m_grid
            )
            reward *= goal_reward_multipliers[current_stage]
            reward = np.clip(reward * reward_scale, -reward_clip, reward_clip)

            previous_distance = distance_to_goal
            goal_reached = (abs(current_coords[0]-goal_coords[0]) <= 1) and (abs(current_coords[1]-goal_coords[1]) <= 1)
            steps_since_goal_introduced += 1; step_count += 1
            reward_history.append(reward); current_episode_reward += reward

            next_obs = get_current_observation(Ag, fov_cells).unsqueeze(0)
            predicted_next_obs = torch.sigmoid(predictive_model.W_out(h))
            pred_loss = nn.MSELoss()(predicted_next_obs, next_obs)

            with torch.no_grad():
                next_state_value = critic(h.detach()) if not goal_reached else torch.tensor([[0.0]], device=device)

            current_state_value = critic(h_detached)
            td_target = torch.tensor([[reward]], dtype=current_state_value.dtype, device=device) + gamma * next_state_value * (1 - int(goal_reached))
            advantage = td_target - current_state_value

            adv_norm_buffer.append(float(advantage.detach().cpu().numpy()))
            if len(adv_norm_buffer) > 256: adv_norm_buffer.pop(0)
            adv_mean = np.mean(adv_norm_buffer); adv_std = np.std(adv_norm_buffer) + 1e-8
            advantage_norm = (advantage - adv_mean) / adv_std

            actor_loss = -action_log_prob * advantage_norm.detach() - entropy_coef * entropy
            critic_loss = F.mse_loss(current_state_value, td_target.detach())
            experience_buffer.append({'state': h_detached.clone(), 'b_state': b_detached.clone(), 'action': action_idx.item(),
                                      'reward': reward, 'next_state': h.detach().clone(), 'done': goal_reached})
            if len(experience_buffer) > buffer_size: experience_buffer = experience_buffer[-buffer_size:]

            total_loss = lambda_pred * pred_loss + actor_loss + value_coef * critic_loss

            predictive_optimizer.zero_grad(); actor_optimizer.zero_grad(); critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictive_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            predictive_optimizer.step(); actor_optimizer.step(); critic_optimizer.step()

            if timestep % 10 == 0:
                actor_scheduler.step(); critic_scheduler.step()

            prediction_losses.append(pred_loss.item())
            rl_losses.append((actor_loss + value_coef * critic_loss).item())

            if len(experience_buffer) > 150 and timestep % 50 == 0:
                sample_size = min(32, len(experience_buffer))
                sampled_exp = random.sample(experience_buffer, sample_size)
                replay_actor_loss = 0; replay_critic_loss = 0
                for exp in sampled_exp:
                    exp_logits = actor(exp['state'])
                    exp_dist = Categorical(logits=exp_logits)
                    exp_state_value = critic(exp['state'])
                    exp_next_value = critic(exp['next_state']) if not exp['done'] else torch.tensor([[0.0]], device=device)
                    exp_td_target = torch.tensor([[exp['reward']]], device=device, dtype=exp_state_value.dtype) + gamma * exp_next_value
                    exp_advantage = exp_td_target - exp_state_value
                    exp_action_log_prob = exp_dist.log_prob(torch.tensor(exp['action'], device=device))
                    replay_actor_loss += -exp_action_log_prob * exp_advantage.detach()
                    replay_critic_loss += F.mse_loss(exp_state_value, exp_td_target.detach())
                replay_actor_loss /= sample_size; replay_critic_loss /= sample_size
                actor_optimizer.zero_grad(); critic_optimizer.zero_grad()
                (replay_actor_loss + value_coef * replay_critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                actor_optimizer.step(); critic_optimizer.step()

            timeout = steps_since_goal_introduced >= goal_timeouts[current_stage]
            if goal_reached or timeout:
                if goal_reached:
                    goals_reached += 1; stage_goals_reached += 1; success_history.append(1)
                    print(f"✅ Step {timestep}: GOAL REACHED in {step_count} steps! Total reward: {current_episode_reward:.1f}")
                else:
                    goals_timed_out += 1; success_history.append(0)
                    print(f"⏰ Step {timestep}: Goal timed out after {step_count} steps. Total reward: {current_episode_reward:.1f}")

                episode_rewards.append(current_episode_reward)
                recent_success_rates.append(1 if goal_reached else 0)
                if len(recent_success_rates) > 50: recent_success_rates = recent_success_rates[-50:]
                phase = "repositioning_pause"; repositioning_steps_left = 40
                goal_coords = None; goal_pos = None; previous_distance = None
                current_episode_reward = 0

        elif phase == "repositioning_pause":
            current_coords, action_idx = get_intelligent_exploration_action(
                current_coords, current_hd_idx, None, 0.9, m_grid, step_count, action_history, visited_positions
            )
            current_hd_idx = action_idx
            action_encoding = torch.zeros(1, predictive_model.act_dim, device=device); action_encoding[0, action_idx] = 1.0
            current_coords = np.clip(current_coords, 3, m_grid-4)
            Ag.pos = grid_centers[current_coords[0], current_coords[1]]
            Ag.head_direction = hds[current_hd_idx]
            with torch.no_grad():
                # FIX: Pass and update b_trace
                h, b = predictive_model.forward_step(obs, action_encoding, h, b)
            hidden_states_history.append(h.squeeze().cpu().detach().numpy())
            prediction_losses.append(0.0); rl_losses.append(0.0); reward_history.append(0.0)
            repositioning_steps_left -= 1
            if repositioning_steps_left <= 0:
                phase = "exploration"; steps_until_goal = np.random.randint(90, 181)

        h = h.detach()
        b = b.detach()

        if timestep % 20000 == 0 and timestep > 0:
            recent_success = np.mean(recent_success_rates) if recent_success_rates else 0
            stage_success = stage_goals_reached / max(1, stage_goals_attempted)
            total_success = goals_reached / max(1, goals_reached + goals_timed_out)
            print(f"\n📊 Progress Report - Step {timestep}")
            print(f"   Recent Success Rate: {recent_success:.3f}")

    total_success_rate = goals_reached / max(1, goals_reached + goals_timed_out)
    print(f"\n🎉 === FIXED SPATIAL RL TRAINING COMPLETE ===")
    print(f"Total goals reached: {goals_reached}")
    print(f"Total goals timed out: {goals_timed_out}")
    print(f"Final success rate: {total_success_rate:.3f}")

    return {
        'phase_history': phase_history,
        'position_history': np.array(position_history),
        'hidden_states_history': np.array(hidden_states_history),
        'prediction_losses': prediction_losses,
        'rl_losses': rl_losses,
        'reward_history': reward_history,
        'episode_rewards': episode_rewards,
        'goals_reached': goals_reached,
        'goals_timed_out': goals_timed_out,
        'success_rate': total_success_rate,
        'success_history': success_history
    }

def test_fixed_policy(predictive_model, actor, goal_pos=(0.50, 0.50),
                     num_trials=30, max_steps=300, visualize=True):
    print(f"\n🧪 Testing FIXED Policy (32×32, No Hint)")
    predictive_model.eval(); actor.eval()
    m_grid = 32
    coord_updates = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])
    fov_cells = fov_cells_all
    goal_grid_coords = np.array([int(goal_pos[0] * m_grid), int(goal_pos[1] * m_grid)])
    actual_goal_pos = grid_centers[goal_grid_coords[0], goal_grid_coords[1]]

    successful_trials, total_steps = 0, 0
    all_trajectories, step_counts = [], []

    def _safe_norm(p):
        p = np.asarray(p, dtype=np.float64); p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p[p<0]=0.0; s=p.sum()
        if s<=0: p=np.ones_like(p)/len(p)
        else: p/=s
        p[np.argmax(p)] += (1.0 - p.sum())
        return np.clip(p,0,None)

    for trial in range(num_trials):
        attempts = 0
        while attempts < 100:
            start_coords = np.random.randint(low=6, high=m_grid-6, size=2)
            start_distance = np.linalg.norm(start_coords - goal_grid_coords)
            if 10 <= start_distance <= 22: break
            attempts += 1
        if attempts >= 100: start_coords = np.array([10, 10])

        current_coords = start_coords.copy()
        current_hd_idx = np.random.randint(8)
        Ag.pos = grid_centers[current_coords[0], current_coords[1]]
        Ag.head_direction = hds[current_hd_idx]
        # FIX: Initialize b_trace for the new RNN
        h = torch.zeros(1, predictive_model.hidden_dim, device=device)
        b = torch.zeros(1, predictive_model.hidden_dim, device=device)

        for _ in range(3):
            obs = get_current_observation(Ag, fov_cells).unsqueeze(0)
            a0 = torch.zeros(1, predictive_model.act_dim, device=device)
            with torch.no_grad():
                # FIX: Pass and update b_trace
                h, b = predictive_model.forward_step(obs, a0, h, b)

        trajectory = [Ag.pos.copy()]
        success = False

        for step in range(max_steps):
            obs = get_current_observation(Ag, fov_cells).unsqueeze(0)

            with torch.no_grad():
                base_logits = actor(h)
                mask = border_action_mask(current_coords, m_grid)
                base_logits = apply_mask_to_logits(base_logits, mask)
                probs_actor = torch.softmax(base_logits, dim=-1).squeeze(0).cpu().numpy()

            probs = _safe_norm(probs_actor)
            action_idx = np.random.choice(8, p=probs)
            current_hd_idx = action_idx

            a_onehot = torch.zeros(1, predictive_model.act_dim, device=device); a_onehot[0, action_idx]=1.0
            with torch.no_grad():
                # FIX: Pass and update b_trace
                h, b = predictive_model.forward_step(obs, a_onehot, h, b)

            new_coords = current_coords + coord_updates[action_idx]
            if (new_coords[0] < 3 or new_coords[0] >= m_grid-3 or new_coords[1] < 3 or new_coords[1] >= m_grid-3):
                new_coords = current_coords
            current_coords = new_coords
            Ag.pos = grid_centers[current_coords[0], current_coords[1]]
            Ag.head_direction = hds[action_idx]

            trajectory.append(Ag.pos.copy())

            if (abs(current_coords[0]-goal_grid_coords[0]) <= 1) and (abs(current_coords[1]-goal_grid_coords[1]) <= 1):
                successful_trials += 1
                total_steps += step + 1
                step_counts.append(step + 1)
                print(f"✅ Trial {trial+1}: SUCCESS in {step+1} steps")
                success = True
                break

        if not success:
            total_steps += max_steps
            step_counts.append(max_steps)
            print(f"❌ Trial {trial+1}: Failed (approx Chebyshev dist to patch center: {max(abs(current_coords[0]-goal_grid_coords[0]), abs(current_coords[1]-goal_grid_coords[1]))-1:.1f})")

        all_trajectories.append(np.array(trajectory))

    success_rate = successful_trials / num_trials
    avg_steps = total_steps / num_trials
    avg_success_steps = np.mean([s for s in step_counts if s < max_steps]) if successful_trials > 0 else max_steps

    print(f"\n📊 Test Results:")
    print(f"   Success Rate: {successful_trials}/{num_trials} ({100*success_rate:.1f}%)")
    print(f"   Average Steps (All): {avg_steps:.1f}")
    print(f"   Average Steps (Successful): {avg_success_steps:.1f}")

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect('equal'); ax1.set_facecolor('lightgray')
        ax1.set_title(f'All Test Trajectories\nSuccess Rate: {successful_trials}/{num_trials} ({100*success_rate:.1f}%)', fontsize=14)
        patch_r = (1.5 / m_grid)
        goal_circle = Circle(actual_goal_pos, patch_r, color='gold', zorder=10, edgecolor='black', linewidth=3)
        ax1.add_patch(goal_circle); ax1.text(actual_goal_pos[0], actual_goal_pos[1]-0.08, 'GOAL (3x3)', ha='center', va='top', fontweight='bold', fontsize=14)
        for i, trajectory in enumerate(all_trajectories):
            success = (abs(int(trajectory[-1,0]*m_grid)-goal_grid_coords[0]) <= 1) and (abs(int(trajectory[-1,1]*m_grid)-goal_grid_coords[1]) <= 1)
            color = 'green' if success else 'red'; alpha = 0.8 if success else 0.4; linewidth = 2 if success else 1
            ax1.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=linewidth, alpha=alpha)
            ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', marker='o', s=60, edgecolor='black', linewidth=1, zorder=5)
        ax1.set_xlabel('X Position'); ax1.set_ylabel('Y Position'); ax1.grid(True, alpha=0.3)

        successful_trajectories = [traj for traj in all_trajectories if (abs(int(traj[-1,0]*m_grid)-goal_grid_coords[0]) <= 1) and (abs(int(traj[-1,1]*m_grid)-goal_grid_coords[1]) <= 1)]
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect('equal'); ax2.set_facecolor('lightgray')
        ax2.set_title(f'Successful Trajectories Only\n{len(successful_trajectories)} successes, Avg steps: {avg_success_steps:.1f}', fontsize=14)
        goal_circle2 = Circle(actual_goal_pos, patch_r, color='gold', zorder=10, edgecolor='black', linewidth=3)
        ax2.add_patch(goal_circle2); ax2.text(actual_goal_pos[0], actual_goal_pos[1]-0.08, 'GOAL (3x3)', ha='center', va='top', fontweight='bold', fontsize=14)
        colors = plt.cm.Set3(np.linspace(0, 1, len(successful_trajectories)))
        for i, trajectory in enumerate(successful_trajectories):
            ax2.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], linewidth=2, alpha=0.8)
            ax2.scatter(trajectory[0, 0], trajectory[0, 1], color=colors[i], marker='o', s=60, edgecolor='black', linewidth=1, zorder=5)
        ax2.set_xlabel('X Position'); ax2.set_ylabel('Y Position'); ax2.grid(True, alpha=0.3)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='green', lw=2, label='Successful'),
                          Line2D([0], [0], color='red', lw=1, alpha=0.4, label='Failed'),
                          Line2D([0], [0], color='blue', marker='o', markersize=8, label='Start Position', linestyle='None')]
        ax1.legend(handles=legend_elements, loc='upper right'); plt.tight_layout(); plt.show()

    return success_rate, avg_steps, avg_success_steps

if __name__ == "__main__":
    obs_array = np.nan_to_num(obs_array, nan=0.0)
    obs_seq = torch.tensor(obs_array[:,:-1], dtype=torch.float32)
    act_seq = torch.tensor(act_array[:,1:], dtype=torch.float32)
    next_obs_seq = torch.tensor(obs_array[:,1:], dtype=torch.float32)
    full_dataset = TensorDataset(obs_seq, act_seq, next_obs_seq)
    val_fraction = 0.2
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    torch.manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=0, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0, pin_memory=False)
    obs_dim, act_dim, hidden_dim = obs_seq.shape[-1], act_seq.shape[-1], 512

    print("🔥 Stage 1: Enhanced Predictive Training")
    model = NextStepRNN(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    train_loss, val_loss = train_next_step_rnn(model, train_loader, val_loader, num_epochs=200, lr=0.001)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title("Predictive Model Training (Pre-training Phase)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3);
    plt.show()

    plot_latent_space_spatial_activity(model, obs_seq_test, act_seq_test, positions,
                                     title="After Pre-training (Before RL)")

    print("\n🚀 Stage 2: FIXED SPATIAL RL TRAINING (No Hint)")
    model_fixed = NextStepRNN(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    model_fixed.load_state_dict(model.state_dict())
    actor_net_fixed = SpatiallyAwareActor(hidden_dim=hidden_dim, act_dim=act_dim, use_attention=True)
    critic_net_fixed = SpatiallyAwareCritic(hidden_dim=hidden_dim)
    print(f"Actor network parameters: {sum(p.numel() for p in actor_net_fixed.parameters()):,}")
    print(f"Critic network parameters: {sum(p.numel() for p in critic_net_fixed.parameters()):,}")

    training_results_fixed = train_fixed_spatial_rl(
        model_fixed, actor_net_fixed, critic_net_fixed,
        total_timesteps=300000,
        curriculum_stages=4,
        gamma=0.99,
        lambda_pred=0.2,
        predictive_lr=1e-5,
        actor_lr=5e-4,
        critic_lr=5e-4,
        entropy_start=0.02,
        entropy_end=0.005,
        reward_scale=0.05,
        reward_clip=20.0,
        value_coef=0.25
    )

    print("\n📊 Stage 3: Results Analysis and Testing")
    episode_rewards = training_results_fixed['episode_rewards']
    if len(episode_rewards) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        episodes = np.arange(len(episode_rewards))
        ax1.plot(episodes, episode_rewards, 'b-', alpha=0.6, linewidth=1)
        if len(episode_rewards) >= 10:
            window_size = max(5, min(20, len(episode_rewards) // 10))
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            avg_episodes = np.arange(window_size-1, len(episode_rewards))
            ax1.plot(avg_episodes, moving_avg, 'r-', linewidth=3, label=f'{window_size}-episode average')
            ax1.legend()
        ax1.set_title('Episode Rewards During Training', fontsize=12); ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward'); ax1.grid(True, alpha=0.3)

        success_history = training_results_fixed.get('success_history', [])
        if len(success_history) >= 5:
            window_size = max(5, min(20, len(success_history) // 5))
            success_rate_plot = np.convolve(success_history, np.ones(window_size)/window_size, mode='valid')
            success_episodes = np.arange(window_size-1, len(success_history))
            ax2.plot(success_episodes, success_rate_plot, 'g-', linewidth=2); ax2.set_ylim(0, 1)
        ax2.set_title('Success Rate Over Episodes', fontsize=12); ax2.set_xlabel('Episode'); ax2.set_ylabel('Success Rate'); ax2.grid(True, alpha=0.3)

        pred_losses = training_results_fixed['prediction_losses']; rl_losses = training_results_fixed['rl_losses']
        if len(pred_losses) > 100:
            sample_indices = np.arange(0, len(pred_losses), 100)
            ax3.plot(sample_indices, [pred_losses[i] for i in sample_indices], 'b-', label='Prediction Loss', alpha=0.7)
            ax3.plot(sample_indices, [rl_losses[i] for i in sample_indices], 'r-', label='RL Loss', alpha=0.7)
        else:
            ax3.plot(pred_losses, 'b-', label='Prediction Loss', alpha=0.7)
            ax3.plot(rl_losses, 'r-', label='RL Loss', alpha=0.7)
        ax3.set_title('Training Losses', fontsize=12); ax3.set_xlabel('Training Steps'); ax3.set_ylabel('Loss'); ax3.legend(); ax3.grid(True, alpha=0.3)

        ax4.hist(episode_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_title('Episode Reward Distribution', fontsize=12); ax4.set_xlabel('Total Reward'); ax4.set_ylabel('Frequency'); ax4.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    print("\n🧪 Testing the FIXED policy...")
    success_rate_center, avg_steps_center, avg_success_steps_center = test_fixed_policy(
        model_fixed, actor_net_fixed, goal_pos=(0.50, 0.50), num_trials=30, max_steps=300, visualize=True
    )

    test_goals = [(0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)]
    all_success_rates = [success_rate_center]; all_step_counts = [avg_success_steps_center]
    print(f"\n🎯 Testing additional goal positions...")
    for i, goal_pos in enumerate(test_goals):
        print(f"\nTesting goal position {i+1}/4: {goal_pos}")
        sr, avg_steps, avg_success_steps = test_fixed_policy(
            model_fixed, actor_net_fixed, goal_pos=goal_pos, num_trials=20, max_steps=300, visualize=False
        )
        all_success_rates.append(sr); all_step_counts.append(avg_success_steps)

    overall_success_rate = np.mean(all_success_rates)
    overall_avg_steps = np.mean(all_step_counts)

    plot_latent_representations_after_rl(
        model_fixed,
        hidden_activity_data=training_results_fixed['hidden_states_history'],
        positions_data=training_results_fixed['position_history'],
        title="FIXED RL Training: Final Spatial Representations (32×32, No Hint)"
    )

    print("\n" + "="*80)
    print("🎯 FIXED SPATIAL RL TRAINING - FINAL RESULTS")
    print("="*80)
    print(f"   Training success rate: {training_results_fixed['success_rate']:.1%}")
    print(f"   Center goal success rate: {success_rate_center:.1%}")
    print(f"   Overall success rate (5 positions): {overall_success_rate:.1%}")
    print(f"   Average steps to success (across successes): {overall_avg_steps:.1f}")
    print("\n🔧 KEY CHANGES IN THIS VERSION:")
    print("• Policy is learned entirely from RL signal (no hint).")
    print("• Actor uses self-attention with 2 heads.")
    print("• Training loop is now more robust to improve generalization.")