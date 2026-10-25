"""Convert run pickles to the Parquet layout dataframe_io writes.

DataFrames become columnar Parquet via save_frame. Dicts of arrays, and the
fitted-estimator pickles under the paper archive, become save_payload files.
Each pickle is deleted only after the Parquet file round-trips.
"""
from __future__ import annotations

import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_LIB = Path(__file__).resolve().parents[1] / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from dataframe_io import load_payload, save_frame, save_payload  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {".git", "__pycache__"}


def _jsonable_attr(value):
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if value is None or isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating, str, tuple)):
        return value
    return None


def _as_payload(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    payload = {"__class__": f"{type(obj).__module__}.{type(obj).__name__}"}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        stored = _jsonable_attr(value)
        if stored is None and value is not None:
            continue
        payload[name] = stored
    arrays = [k for k, v in payload.items() if isinstance(v, np.ndarray)]
    if not arrays:
        raise TypeError(f"No arrays to store from {type(obj)}")
    return payload


def _frames_match(src: pd.DataFrame, dst: pd.DataFrame) -> None:
    if list(dst.columns) != list(src.columns):
        raise AssertionError(f"columns {list(dst.columns)} != {list(src.columns)}")
    if len(dst) != len(src):
        raise AssertionError(f"length {len(dst)} != {len(src)}")
    for col in src.columns:
        if dst[col].dtype != src[col].dtype:
            raise AssertionError(f"{col} dtype {dst[col].dtype} != {src[col].dtype}")
        n = len(src)
        if n == 0:
            continue
        idx = np.linspace(0, n - 1, num=min(2048, n), dtype=np.int64)
        a = src[col].to_numpy()[idx]
        b = dst[col].to_numpy()[idx]
        if not np.array_equal(a, b):
            raise AssertionError(f"values differ in column {col}")


def _payloads_match(src: dict, dst: dict) -> None:
    if set(dst) != set(src):
        raise AssertionError(f"keys {sorted(dst)} != {sorted(src)}")
    for key, value in src.items():
        got = dst[key]
        if isinstance(value, np.ndarray):
            if not np.array_equal(got, value):
                raise AssertionError(f"array {key} differs")
        elif got != value:
            raise AssertionError(f"value {key} differs: {got!r} != {value!r}")


def convert_one(path_str: str) -> tuple[str, int, int, str]:
    """Return (status, bytes_in, bytes_out, detail). status is ok or error."""
    src = Path(path_str)
    dest = src.with_suffix(".parquet")
    try:
        with src.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, pd.DataFrame):
            save_frame(dest, obj)
            _frames_match(obj, pd.read_parquet(dest))
        else:
            payload = _as_payload(obj)
            save_payload(dest, payload)
            _payloads_match(payload, load_payload(dest))
        out_bytes = dest.stat().st_size
        in_bytes = src.stat().st_size
        src.unlink()
        return "ok", in_bytes, out_bytes, ""
    except Exception as exc:
        dest.with_name(dest.name + ".partial").unlink(missing_ok=True)
        if src.exists():
            dest.unlink(missing_ok=True)
        return "error", src.stat().st_size if src.exists() else 0, 0, f"{src.name}: {type(exc).__name__}: {exc}"


def iter_pickles(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
        for name in filenames:
            if name.endswith(".pkl"):
                yield str(Path(dirpath) / name)


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    paths = list(iter_pickles(ROOT))
    workers = int(os.environ.get("PARQUET_WORKERS", "8"))
    print(f"Converting {len(paths)} pickles with {workers} workers", flush=True)
    ok = err = 0
    bytes_in = bytes_out = 0
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(convert_one, p) for p in paths]
        for i, fut in enumerate(as_completed(futures), start=1):
            status, nin, nout, detail = fut.result()
            if status == "ok":
                ok += 1
                bytes_in += nin
                bytes_out += nout
            else:
                err += 1
                errors.append(detail)
            if i % 50 == 0 or i == len(paths):
                print(
                    f"{i}/{len(paths)} ok={ok} err={err} "
                    f"in={bytes_in/1024**3:.1f} GiB out={bytes_out/1024**3:.1f} GiB",
                    flush=True,
                )
    if errors:
        print("Failures:", flush=True)
        for line in errors[:30]:
            print(" ", line, flush=True)
    print(f"Done. ok={ok} err={err}", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
