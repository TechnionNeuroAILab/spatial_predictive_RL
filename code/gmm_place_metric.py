import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


# 1) Rate map: occupancy-normalized mean activity per bin
def build_rate_map(
    df_neuron: pd.DataFrame, m: int = 32, smooth_sigma_bins: float = 1.0
):
    """
    df_neuron: columns include ['x','y','activity'] for a single neuron (after merging by timestep).
    m: number of bins per axis (grid m x m).
    Returns: rate_map, (xedges,yedges), sum_a, occ
    """
    x = df_neuron["x"].to_numpy(float)
    y = df_neuron["y"].to_numpy(float)
    a = df_neuron["activity"].to_numpy(float)

    xedges = np.linspace(np.nanmin(x), np.nanmax(x), m)
    yedges = np.linspace(np.nanmin(y), np.nanmax(y), m)

    sum_a, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=a)
    occ, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])

    rate = sum_a / occ

    return rate, (xedges, yedges), sum_a, occ


# 2) Convert rate map into (coords, weights)
def rate_map_to_coords(
    rate_map: np.ndarray,
    edges: Tuple[np.ndarray, np.ndarray],
    sum_a: np.ndarray,
    min_rate_frac: float = 0.30,
):
    """
    Keep bins with >= min_rate_frac * max(rate).
    Use spike mass (sum_a per bin) as the weight for robust likelihood/ICL.
    Returns: coords (N,2), weights (N,)
    """
    xedges, yedges = edges
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(xc, yc, indexing="ij")

    max_rate = np.nanmax(rate_map)
    mask = rate_map >= min_rate_frac * max_rate

    coords = np.column_stack((XX[mask], YY[mask]))
    weights = sum_a[mask]
    return coords, weights


# 3) GMM fit 
def fit_gmm(
    X: np.ndarray,
    k: int,
    covariance_type="full",
    reg_covar=1e-3,
    random_state=0,
    n_init=5,
) -> GaussianMixture:

    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
        n_init=n_init,
    )
    gmm.fit(X)
    return gmm


def weighted_ll(gmm: GaussianMixture, X: np.ndarray, w: np.ndarray) -> float:
    if X.size == 0:
        return float("-inf")
    return float(np.sum(w * gmm.score_samples(X)))

# 4) ICL (smaller is better): ICL ≈ BIC + 2 * entropy(resp)
def icl_score(
    gmm: GaussianMixture, X: np.ndarray, w: np.ndarray, ll: float | None = None
) -> float:
    # parameter count for 2D full-cov GMM: means(2*k) + cov(3*k) + weights(k-1)
    k = gmm.n_components
    p = k * (2 + 3) + (k - 1)
    n_eff = max(np.sum(w), 1.0)

    bic = -2.0 * ll + p * np.log(n_eff)

    # Weighted entropy of responsibilities
    resp = gmm.predict_proba(X)
    resp = np.clip(resp, 1e-12, 1.0)
    entropy = -np.sum((w[:, None]) * (resp * np.log(resp)))

    return float(bic + 2.0 * entropy)


def select_k_by_icl(
    coords: np.ndarray,
    weights: np.ndarray,
    k_max: int = 6,
    random_state: int = 42,
    reg_covar: float = 1e-3,
    n_init: int = 5,
) -> Dict[str, Any]:
    scores = []
    for k in range(1, k_max + 1):
        gmm = fit_gmm(
            coords,
            k,
            covariance_type="full",
            reg_covar=reg_covar,
            random_state=random_state,
            n_init=n_init,
        )
        ll = weighted_ll(gmm, coords, weights)
        icl = icl_score(gmm, coords, weights, ll=ll)
        scores.append((k, icl, ll, gmm))
    best_k, best_icl, best_ll, best_gmm = min(scores, key=lambda t: t[1])
    return {
        "k": int(best_k),
        "icl": float(best_icl),
        "ll": float(best_ll),
        "gmm": best_gmm,
    }


# -----------------------------
# 5) Per-neuron processing and batch runner
# -----------------------------
def process_one_neuron(
    df_behavior_after: pd.DataFrame,
    df_activity_after: pd.DataFrame,
    neuron_id: int,
    m: int = 32,
    smooth_sigma_bins: float = 1.0,
    min_rate_frac: float = 0.30,
    k_max: int = 6,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Merge behavior+activity on timestep for a given neuron, compute ICL-best GMM,
    and return {'neuron_id', 'k', 'centers'}.
    """
    df_a = df_activity_after[df_activity_after["neuron_id"] == neuron_id]

    df = df_behavior_after.merge(df_a, on="timestep", how="inner")

    # Build rate map
    rate_map, edges, sum_a, _ = build_rate_map(
        df, m=m, smooth_sigma_bins=smooth_sigma_bins
    )

    # Convert to coords+weights (use spike mass)
    coords, weights = rate_map_to_coords(
        rate_map, edges, sum_a=sum_a, min_rate_frac=min_rate_frac
    )

    # Select K by ICL
    icl_res = select_k_by_icl(coords, weights, k_max=k_max, random_state=random_state)
    best_gmm = icl_res["gmm"]
    k = icl_res["k"]
    icl_score = icl_res["icl"]
    centers = best_gmm.means_.astype(float).tolist()  # list of [x,y]
    return {
        "neuron_id": neuron_id,
        "k": k,
        "centers": [(round(c[0], 3), round(c[1], 3)) for c in centers],
        "icl_score": icl_score
    }


def run_all_neurons(
    behavior_pickle_path: str = "data/df_behavior_after.pkl",
    activity_pickle_path: str = "data/df_activations_after.pkl",
    out_pickle_path: str = "data/df_centers_metric_after.pkl",
    m: int = 32,
    smooth_sigma_bins: float = 1.0,
    min_rate_frac: float = 0.30,
    k_max: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads behavior+activity pickles, runs ICL selection per neuron, and saves df_centers.pkl.
    Output columns: neuron_id (int), k (int), centers (list[tuple[float,float]])
    """
    df_behavior_after = pd.read_pickle(behavior_pickle_path)
    df_activity_after = pd.read_pickle(activity_pickle_path)

    neuron_ids = sorted(df_activity_after["neuron_id"].unique().tolist())

    rows = []
    for nid in tqdm(neuron_ids, total=len(neuron_ids), desc="ICL per neuron"):
        res = process_one_neuron(
            df_behavior_after,
            df_activity_after,
            nid,
            m=m,
            smooth_sigma_bins=smooth_sigma_bins,
            min_rate_frac=min_rate_frac,
            k_max=k_max,
            random_state=random_state,
        )
        rows.append(res)

    out_df = pd.DataFrame(rows, columns=["neuron_id", "k", "centers", "icl_score"])
    out_df = (
        out_df
        .explode("centers", ignore_index=True)   # split list into separate rows
        .rename(columns={"centers": "center"})   # centers -> center (single tuple)
    )
    out_df.to_pickle(out_pickle_path)
    return out_df


if __name__ == "__main__":
    df_centers = run_all_neurons(
        behavior_pickle_path="data/df_behavior_before.pkl",
        activity_pickle_path="data/df_activations_before.pkl",
        out_pickle_path="data/df_centers_before.pkl",
        m=32,  # grid matches 32x32 env
        smooth_sigma_bins=1.0,  # light smoothing in bins
        min_rate_frac=0.3,  # keep top bins (by rate) for stability
        k_max=6,
        random_state=42,
    )
    print(df_centers.head())
