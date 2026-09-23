#!/usr/bin/env python3
"""Compare regenerated figures against a saved baseline, by rendered pixels.

PDFs embed a creation timestamp and a document ID, so two byte-different PDFs
can be visually identical. This rasterises both with pdftoppm and reports the
fraction of differing pixels -- the only comparison that answers the question
that matters after moving/restyling the pipeline: *did the content change?*

    python code/compare_figures.py --baseline <dir-of-pdfs>

The baseline dir is a flat directory of `*_raw.pdf` files (as produced by
`cp` from a previous state of code/figures).
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PC = Path(__file__).resolve().parent
FIG = PC / "figures"


def render(pdf: Path, dpi: int, out_dir: Path) -> list[Path]:
    stem = out_dir / pdf.stem
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(stem)],
                   check=True, capture_output=True)
    return sorted(out_dir.glob(f"{pdf.stem}*.png"))


def load(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("L"), dtype=np.int16)


def compare(a: Path, b: Path, dpi: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        da, db = Path(td) / "a", Path(td) / "b"
        da.mkdir(); db.mkdir()
        pa, pb = render(a, dpi, da), render(b, dpi, db)
        if len(pa) != len(pb):
            return {"status": "PAGE COUNT", "detail": f"{len(pa)} vs {len(pb)}"}
        fracs, shapes = [], []
        for x, y in zip(pa, pb):
            ia, ib = load(x), load(y)
            if ia.shape != ib.shape:
                shapes.append(f"{ia.shape} vs {ib.shape}")
                continue
            diff = np.abs(ia - ib)
            fracs.append(float((diff > 8).mean()))
        if shapes:
            return {"status": "SIZE", "detail": "; ".join(shapes)}
        frac = max(fracs) if fracs else 0.0
        if frac == 0.0:
            return {"status": "identical", "frac": 0.0}
        if frac < 0.001:
            return {"status": "~identical", "frac": frac}
        return {"status": "CHANGED", "frac": frac}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    base = Path(args.baseline)
    current = {p.name: p for p in FIG.rglob("*_raw.pdf")}
    rows = []
    for bp in sorted(base.glob("*_raw.pdf")):
        cp = current.get(bp.name)
        if cp is None:
            rows.append((bp.name, {"status": "MISSING NOW", "frac": None}))
            continue
        rows.append((bp.name, compare(bp, cp, args.dpi)))

    w = max(len(n) for n, _ in rows) + 2
    for name, r in rows:
        frac = r.get("frac")
        extra = f"{frac*100:6.2f}% px differ" if frac else r.get("detail", "")
        print(f"  {r['status']:12s} {name:<{w}s} {extra}")
    changed = [n for n, r in rows if r["status"] not in ("identical", "~identical")]
    print(f"\n{len(rows)-len(changed)}/{len(rows)} visually unchanged")
    if changed:
        print("changed: " + ", ".join(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
