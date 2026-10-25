"""Inputs a figure actually plots, saved so a later edit only redraws.

A full generator run writes ``figure_data.npz`` beside its panels. ``--plot-only``
loads that file and skips the run scan, the nulls, and the rate-map rebuild.
Layout-only restacking of panel PDFs stays ``--assemble_only``: that path cannot
recolor a map, because the map is already baked into the PDF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def add_plot_only(parser):
    """Flag shared by every generator ``regenerate_all.py --plot-only`` calls."""
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw panels from figure_data.npz in the output directory; "
             "do not rescan runs or recompute statistics",
    )
    return parser


def save_figure_data(path, payload) -> Path:
    """Write ``payload`` (nested dicts, lists, tuples, scalars, ndarrays)."""
    path = _npz_path(path)
    arrays: dict[str, np.ndarray] = {}
    meta = _encode(payload, arrays)
    raw = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, __meta__=raw, **arrays)
    print(f"  Saved figure data: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_figure_data(path, quiet: bool = False):
    """Inverse of :func:`save_figure_data`. Ndarrays come back as ndarrays."""
    path = _npz_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved figure data at {path}. "
            "Run this generator once without --plot-only."
        )
    with np.load(path, allow_pickle=False) as blob:
        meta = json.loads(bytes(blob["__meta__"]).decode("utf-8"))
        arrays = {
            key: np.array(blob[key], copy=True)
            for key in blob.files if key != "__meta__"
        }
    if not quiet:
        print(f"  Loaded figure data: {path}")
    return _decode(meta, arrays)


def update_figure_data(path, payload) -> Path:
    """Merge ``payload`` into an existing file so a one-panel run keeps the rest."""
    path = _npz_path(path)
    if path.exists():
        current = load_figure_data(path, quiet=True)
        if not isinstance(current, dict) or not isinstance(payload, dict):
            current = payload
        else:
            current.update(payload)
        payload = current
    return save_figure_data(path, payload)


def require_keys(payload, *keys, path=None):
    """Raise if a plot-only load is missing something the requested panels need."""
    missing = [key for key in keys if key not in payload]
    if missing:
        where = f" in {path}" if path else ""
        raise KeyError(
            f"Saved figure data{where} has no {missing}. "
            "Run this generator once without --plot-only."
        )
    return payload


def _npz_path(path) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    return path


def _encode(obj, arrays: dict[str, np.ndarray]):
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError("figure data cannot store an object array")
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(obj)
        return {"__ndarray__": key}
    if isinstance(obj, np.generic):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return _encode(np.asarray(obj), arrays)
    if isinstance(obj, dict):
        return {"__map__": [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return [_encode(v, arrays) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"cannot save a {type(obj).__name__} in figure data")


def _decode(obj, arrays: dict[str, np.ndarray]):
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    if isinstance(obj, dict):
        if set(obj) == {"__ndarray__"}:
            return arrays[obj["__ndarray__"]]
        if set(obj) == {"__tuple__"}:
            return tuple(_decode(v, arrays) for v in obj["__tuple__"])
        if set(obj) == {"__map__"}:
            out = {}
            for key, value in obj["__map__"]:
                decoded = _decode(key, arrays)
                if isinstance(decoded, list):
                    decoded = tuple(decoded)
                out[decoded] = _decode(value, arrays)
            return out
    return obj
