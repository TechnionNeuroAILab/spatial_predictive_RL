from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_1samp, ttest_ind, wilcoxon

from common import (
    bootstrap_mean_ci,
    cohen_d_independent,
    fdr_bh,
    save_json,
)


def load_vector(path: str | Path, column: str | None = None) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")

    suffix = p.suffix.lower()

    if suffix == ".npy":
        vec = np.load(p)
        return np.asarray(vec, dtype=float).reshape(-1)

    if suffix == ".npz":
        npz = np.load(p)
        if len(npz.files) == 0:
            raise ValueError(f"No arrays in {p}")
        vec = np.asarray(npz[npz.files[0]], dtype=float)
        return vec.reshape(-1)

    if suffix == ".csv":
        df = pd.read_csv(p)
        if column is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) != 1:
                raise ValueError(
                    f"CSV {p} has {len(num_cols)} numeric columns. "
                    "Provide --shift-column/--conc-column explicitly."
                )
            column = num_cols[0]
        return df[column].to_numpy(dtype=float)

    if suffix == ".pkl":
        obj = pd.read_pickle(p)
        if isinstance(obj, pd.Series):
            return obj.to_numpy(dtype=float)
        if isinstance(obj, pd.DataFrame):
            if column is None:
                num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
                if len(num_cols) != 1:
                    raise ValueError(
                        f"Pickle {p} has {len(num_cols)} numeric columns. "
                        "Provide --shift-column/--conc-column explicitly."
                    )
                column = num_cols[0]
            return obj[column].to_numpy(dtype=float)

        arr = np.asarray(obj, dtype=float)
        return arr.reshape(-1)

    raise ValueError(f"Unsupported file format: {p.suffix}")


def sanitize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(-1)
    return v[np.isfinite(v)]


def one_sample_cohen_d(vec: np.ndarray, mu0: float = 0.0) -> float:
    v = sanitize(vec)
    if v.size < 2:
        return np.nan
    sd = np.std(v, ddof=1)
    if np.isclose(sd, 0.0):
        return np.nan
    return float((np.mean(v) - mu0) / sd)


def bootstrap_delta_mean(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = sanitize(x)
    y = sanitize(y)

    if x.size == 0 or y.size == 0:
        return np.nan, np.nan, np.nan

    deltas = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        sx = rng.choice(x, size=x.size, replace=True)
        sy = rng.choice(y, size=y.size, replace=True)
        deltas[i] = np.mean(sy) - np.mean(sx)

    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def analyze_shift(vec: np.ndarray, alpha: float, n_boot: int, seed: int) -> dict[str, Any]:
    v = sanitize(vec)
    out: dict[str, Any] = {
        "n": int(v.size),
        "mean": float(np.mean(v)) if v.size else np.nan,
        "std": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
    }

    if v.size >= 2:
        t = ttest_1samp(v, popmean=0.0, alternative="greater")
        out["t_stat"] = float(t.statistic)
        out["t_p_one_tailed"] = float(t.pvalue)
        try:
            w = wilcoxon(v, alternative="greater", zero_method="wilcox")
            out["wilcoxon_stat"] = float(w.statistic)
            out["wilcoxon_p_one_tailed"] = float(w.pvalue)
        except ValueError:
            out["wilcoxon_stat"] = np.nan
            out["wilcoxon_p_one_tailed"] = np.nan

        out["cohen_d_one_sample"] = one_sample_cohen_d(v, mu0=0.0)
        mean, low, high = bootstrap_mean_ci(v, n_boot=n_boot, ci=95.0, seed=seed)
        out["bootstrap_mean"] = mean
        out["bootstrap_ci_low"] = low
        out["bootstrap_ci_high"] = high

    return out


def analyze_concentration(before: np.ndarray, after: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    b = sanitize(before)
    a = sanitize(after)
    out: dict[str, Any] = {
        "n_before": int(b.size),
        "n_after": int(a.size),
        "mean_before": float(np.mean(b)) if b.size else np.nan,
        "mean_after": float(np.mean(a)) if a.size else np.nan,
        "cohen_d_independent": cohen_d_independent(a, b),
    }
    if b.size >= 2 and a.size >= 2:
        t = ttest_ind(b, a, equal_var=False, alternative="less")
        out["welch_t"] = float(t.statistic)
        out["welch_p_one_tailed"] = float(t.pvalue)
        ks = ks_2samp(a, b, alternative="greater", method="auto")
        out["ks_stat"] = float(ks.statistic)
        out["ks_p_one_tailed"] = float(ks.pvalue)
        d_mean, d_low, d_high = bootstrap_delta_mean(b, a, n_boot=n_boot, seed=seed)
        out["bootstrap_delta_mean_after_minus_before"] = d_mean
        out["bootstrap_delta_ci_low"] = d_low
        out["bootstrap_delta_ci_high"] = d_high

    return out


def add_multiple_corrections(results: dict[str, Any]) -> dict[str, Any]:
    entries: list[tuple[str, float]] = []
    for block_name in ["shift", "concentration"]:
        block = results.get(block_name, {})
        for k, v in block.items():
            if not isinstance(v, (int, float, np.number)):
                continue
            if "_p" in k and np.isfinite(v):
                entries.append((f"{block_name}.{k}", float(v)))

    if not entries:
        results["multiple_correction"] = {"n_tests": 0, "entries": []}
        return results
    names = [n for n, _ in entries]
    pvals = np.array([p for _, p in entries], dtype=float)

    bonf = np.clip(pvals * len(pvals), 0.0, 1.0)
    fdr = fdr_bh(pvals)

    corrected = []
    for i, name in enumerate(names):
        corrected.append(
            {
                "test": name,
                "p_raw": float(pvals[i]),
                "p_bonferroni": float(bonf[i]),
                "p_fdr_bh": float(fdr[i]),
            }
        )

    results["multiple_correction"] = {
        "n_tests": int(len(corrected)),
        "entries": corrected,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supplementary statistical tests for manuscript reporting.")
    parser.add_argument("--output-json", type=str, default="../New_Results/statistical_supplement/statistical_summary.json")
    parser.add_argument("--shift-file", type=str, default="", help="Optional file containing shift values.")
    parser.add_argument("--shift-column", type=str, default="", help="Column name for shift values if input is csv/pkl DataFrame.")
    parser.add_argument("--conc-before-file", type=str, default="", help="Optional concentration vector for BEFORE.")
    parser.add_argument("--conc-after-file", type=str, default="", help="Optional concentration vector for AFTER.")
    parser.add_argument("--conc-column", type=str, default="", help="Column name for concentration vectors if csv/pkl DataFrame.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results: dict[str, Any] = {
        "alpha": float(args.alpha),
        "n_boot": int(args.n_boot),
    }

    if args.shift_file:
        shift_col = args.shift_column if args.shift_column else None
        shift_vec = load_vector(args.shift_file, column=shift_col)
        results["shift"] = analyze_shift(shift_vec, alpha=args.alpha, n_boot=args.n_boot, seed=args.seed)
    else:
        results["shift"] = {"note": "No shift file provided; shift tests skipped."}

    if args.conc_before_file and args.conc_after_file:
        conc_col = args.conc_column if args.conc_column else None
        conc_before = load_vector(args.conc_before_file, column=conc_col)
        conc_after = load_vector(args.conc_after_file, column=conc_col)
        results["concentration"] = analyze_concentration(conc_before, conc_after, n_boot=args.n_boot, seed=args.seed)
    else:
        results["concentration"] = {"note": "Concentration files missing; concentration tests skipped."}

    results = add_multiple_corrections(results)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, results)
    print(f"Saved statistical supplement to: {out}")

if __name__ == "__main__":
    main()
