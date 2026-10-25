#!/usr/bin/env python3
"""
Copy the assembled figure PDFs into paper_repo/paper_figures/v2/ under stable names.

Design decisions, both deliberate:

1. **Explicit dest -> src allowlist, never a glob.** Several files in this repo
   have shared the name `figure3_raw.pdf`: the real Figure 3, one from a
   since-deleted plasticity-figure directory, and a stray written into
   `fig2/figures/` by a `sys.modules["utils"]`
   caching bug. A glob would silently pick the wrong one.

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
    python code/sync_paper_figures.py --dry-run
    python code/sync_paper_figures.py
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "code" / "figures"
DEST = REPO / "paper_repo" / "paper_figures" / "v2"

# dest stem -> source pdf. Every entry checked by hand.
# Dest stems are the names paper_repo/main.tex passes to \includegraphics.
FIGURES: dict[str, Path] = {
    # main figures
    # The architecture schematic was panel a of fig1, where a 150 pt row cap
    # scaled it to 0.548 and its labels printed at 4.7 pt. It is its own
    # figure now, at the full text width and its native aspect ratio.
    "fig_model": FIG / "fig1" / "figures" / "figure_model_raw.pdf",
    "fig1": FIG / "fig1" / "figures" / "figure1_raw.pdf",
    "fig2": FIG / "fig2" / "figures" / "figure2_raw.pdf",
    "fig3": FIG / "fig3" / "figures" / "figure3_raw.pdf",
    "fig4": FIG / "fig4" / "figures" / "figure4_raw.pdf",
    # The data-confrontation figure (generate_figure5_data.py) was one
    # 183 x 262 mm plate -- taller than the text block, so LaTeX capped its
    # height and shrank it to 61%, printing 96% of its text below 5 pt. It
    # splits along its own subject matter into omission (Krishnan) and
    # relocation (Sosa). The former fig5 (H1/H2 schematic + dissociation)
    # was retired from the manuscript.
    "fig_omission": FIG / "fig5" / "figures" / "figure_omission_raw.pdf",
    "fig_relocdata": FIG / "fig5" / "figures" / "figure_relocdata_raw.pdf",
    # supplements
    "figS_pipeline":     FIG / "fig1" / "figures_supp" / "figure1_supp_raw.pdf",
    "figS_relocation":   FIG / "fig4" / "figures_supp" / "figure4_supp_raw.pdf",
    "figS_dissociation": FIG / "fig5" / "figures_supp" / "figure5_supp_raw.pdf",
    "figS_rewardvector": FIG / "figS_reward_vector" / "figures_supp" / "figure_reward_vector_supp_raw.pdf",
    "figS_sr":           FIG / "figS_sr" / "figures_supp" / "figure_sr_supp_raw.pdf",
    # D1-D4 confrontation/discriminator supplements (2026 revision)
    "figS_extinction":   FIG / "figS_extinction" / "figures_supp" / "figure_extinction_supp_raw.pdf",
    "figS_discriminator": FIG / "figS_discriminator" / "figures_supp" / "figure_discriminator_supp_raw.pdf",
    "figS_sosa":         FIG / "figS_sosa" / "figures_supp" / "figure_sosa_supp_raw.pdf",
}

# No optional sync entries: unreproducible stems (figS_manifold, figS_fourcond,
# figS_fate, figS_decoder, figS_head) were removed from the manuscript and are
# not synced.
OPTIONAL: set[str] = set()

def check_manuscript_coverage(main_tex: Path) -> tuple[list[str], list[str]]:
    """Return manuscript stems missing from the allowlist and unused entries.

    The allowlist is deliberately hand-maintained, which means it can fall
    behind a figure that gets split or renamed -- exactly what happened when
    fig5 became fig_omission and fig_relocdata.  Enforcing equality keeps the
    publication pipeline one-to-one with the manuscript.
    """
    import re
    if not main_tex.exists():
        return [], []
    used = {Path(m).stem for m in
            re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", main_tex.read_text())}
    # the two hand-drawn architecture plates live in paper_repo, not here
    generated = used - {"v10_architecture", "v11_architecture"}
    return sorted(generated - set(FIGURES)), sorted(set(FIGURES) - generated)


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

    missing_stems, unused_stems = check_manuscript_coverage(
        REPO / "paper_repo" / "main.tex"
    )
    if missing_stems or unused_stems:
        print("REFUSING TO RUN — figure allowlist does not match main.tex:")
        if missing_stems:
            print("  missing from allowlist: " + ", ".join(missing_stems))
        if unused_stems:
            print("  not used by manuscript: " + ", ".join(unused_stems))
        raise SystemExit(1)

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
