#!/usr/bin/env python3
"""
Copy the assembled figure PDFs into paper/paper_figures/v2/ under stable names.

Design decisions, both deliberate:

1. **Explicit dest -> src allowlist, never a glob.** THREE different files in
   this repo are named `figure3_raw.pdf`: the real Figure 3, an unrelated one in
   `code_analysis/figure_plasticity/`, and (formerly) a stray written into
   `code_analysis/figure 2/figures/` by a `sys.modules["utils"]` caching bug. A
   glob would silently pick the wrong one.

2. **A versioned subdirectory, not in-place overwrite.** `paper_figures/` already
   contains a `figure1_raw.pdf` from July, plus the whole set the current
   manuscript points at. Writing into `v2/` and prepending that one path to
   `\\graphicspath` means reverting is a one-line change and no existing file is
   ever destroyed.

PDFs only — LaTeX does not use the PNGs and they are ~2x the size.

Writes MANIFEST.txt recording, per file: source path, source mtime, sha256, the
main-repo git HEAD, and the sync time, so a figure in the PDF can always be
traced back to the run that produced it.

Usage:
    python New_Code/sync_paper_figures.py --dry-run
    python New_Code/sync_paper_figures.py
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

REPO = Path(__file__).resolve().parent.parent
CA = REPO / "code_analysis"
DEST = REPO / "paper" / "paper_figures" / "v2"

# dest stem -> source pdf. Every entry checked by hand.
FIGURES: dict[str, Path] = {
    # main figures
    "fig1": CA / "figure 1" / "figures" / "figure1_raw.pdf",
    "fig2": CA / "figure 2" / "figures" / "figure2_raw.pdf",
    "fig3": CA / "figure 3" / "figures" / "figure3_raw.pdf",
    "fig4": CA / "figure 4" / "figures" / "figure4_raw.pdf",
    # fig5 is the data-confrontation figure (Krishnan + Sosa cartoons and
    # model-vs-data panels, generate_figure5_data.py); the former fig5
    # (H1/H2 schematic + dissociation) moved to the supplement as
    # figS_hypotheses.
    "fig5": CA / "figure 5" / "figures" / "figure5_data_raw.pdf",
    "figS_hypotheses": CA / "figure 5" / "figures" / "figure5_raw.pdf",
    # supplements
    "figS_pipeline":     CA / "figure 1" / "figures_supp" / "figure1_supp_raw.pdf",
    "figS_frozenreplay": CA / "figure 2" / "figures_supp" / "figure2_supp_raw.pdf",
    "figS_head":         CA / "figure 3" / "figures_supp" / "figure3_supp_raw.pdf",
    "figS_relocation":   CA / "figure 4" / "figures_supp" / "figure4_supp_raw.pdf",
    "figS_dissociation": CA / "figure 5" / "figures_supp" / "figure5_supp_raw.pdf",
    "figS_rewardvector": CA / "figure_reward_vector" / "figures_supp" / "figure_reward_vector_supp_raw.pdf",
    "figS_sr":           CA / "figure_sr" / "figures_supp" / "figure_sr_supp_raw.pdf",
    # Requires Repro/multi_seed_rigid/ to exist; until then a clearly-marked
    # placeholder sits at the destination and this entry is skipped.
    "figS_rigid":        CA / "figure_sr" / "figures_supp" / "figure_rigid_supp_raw.pdf",
    # D1-D4 confrontation/discriminator supplements (2026 revision)
    "figS_extinction":   CA / "figure_extinction" / "figures_supp" / "figure_extinction_supp_raw.pdf",
    "figS_discriminator": CA / "figure_discriminator" / "figures_supp" / "figure_discriminator_supp_raw.pdf",
    # Sosa reanalysis lives under external_data/, not code_analysis/
    "figS_sosa":         REPO / "external_data" / "sosa2025" / "figures" / "figure_sosa_supp_raw.pdf",
}

OPTIONAL = {"figS_rigid"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    missing = {k: v for k, v in FIGURES.items() if not v.exists() and k not in OPTIONAL}
    if missing:
        print("REFUSING TO RUN — missing sources:")
        for k, v in missing.items():
            print(f"  {k}: {v}")
        raise SystemExit(1)

    if not args.dry_run:
        DEST.mkdir(parents=True, exist_ok=True)

    lines = [
        "# paper_figures/v2 manifest",
        f"# synced: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# main-repo git HEAD: {git_head()}",
        "",
    ]
    for stem, src in sorted(FIGURES.items()):
        if not src.exists():
            print(f"  skip (optional, absent): {stem}")
            continue
        dst = DEST / f"{stem}.pdf"
        digest = sha256(src)
        mtime = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        rel = src.relative_to(REPO)
        if args.dry_run:
            print(f"  would copy {rel}  ->  paper_figures/v2/{stem}.pdf")
        else:
            copy2(src, dst)
            print(f"  {stem}.pdf  <-  {rel}")
        lines += [f"{stem}.pdf", f"    source: {rel}", f"    mtime:  {mtime}",
                  f"    sha256: {digest}", ""]

    if not args.dry_run:
        (DEST / "MANIFEST.txt").write_text("\n".join(lines))
        print(f"\nWrote {DEST / 'MANIFEST.txt'}")
    print(f"\n{len([k for k,v in FIGURES.items() if v.exists()])} figures synced to {DEST}")


if __name__ == "__main__":
    main()
