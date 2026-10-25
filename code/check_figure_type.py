#!/usr/bin/env python3
"""Typography gate for the manuscript figure set.

Every figure in this paper is assembled twice: once when a per-figure `pypdf`
assembler merges the panel PDFs onto a page, and once when LaTeX applies the
`\\includegraphics` width/height options. Both steps scale, and the scales
compound, so a label authored at 7 pt can print at 3 pt without anything in the
Python source looking wrong. That is invisible to every other check in this
repo -- `smoke_test.py` never renders, and `compare_figures.py` compares
rasters at a fixed dpi, so a uniformly-too-small figure passes it cleanly.

This script closes that gap. It measures what a reader actually sees:

    effective_pt = span_size_in_pdf * latex_include_scale

`mutool draw -F stext` is the tool for the first factor because it reports each
span's size *in final page coordinates with the CTM applied* -- so a panel that
was merged at 0.6x reports its true on-page size, not its authored size.
Verified against figS_rigid, whose 6.0 pt ticks report 3.597 pt = 6.0 x 0.5994,
exactly its measured merge scale. The alternatives do not work: `pdffonts`
reports names only, and `pypdf.extract_text()` does not expose per-span sizes
without reimplementing the PDF text-state machine.

The second factor is parsed straight out of `main.tex`, including the
`keepaspectratio` case where a height cap can win over the width option (which
is what reduces Figure 5 to 61 %).

Gates, in the order they are worth fixing:

  scale   |abs(scale - 1)| <= 0.005. This is the real invariant; every other
          failure is downstream of it. A figure authored at the text width and
          included at \\linewidth needs no rescale at all.
  floor   no glyph below 5.0 pt (Nature's hard minimum).
  height  artwork height on the page must fit inside what is left of the text
          block after the caption.
  fonts   one typeface across the set, so the figures read as one document.

Usage:
    python code/check_figure_type.py                 # gate every figure
    python code/check_figure_type.py --only fig1 fig4
    python code/check_figure_type.py --json > type.json
    python code/check_figure_type.py --baseline type.json    # diff vs a record
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN_TEX = REPO / "paper_repo" / "main.tex"
FIG_DIR = REPO / "paper_repo" / "paper_figures" / "v2"

# Page geometry is shared with the generators (code/lib/pagegeom.py) so the
# gate and the thing it gates cannot drift apart. pagegeom deliberately has no
# matplotlib dependency, so this stays a plain measuring script.
sys.path.insert(0, str(REPO / "code" / "lib"))
from pagegeom import (  # noqa: E402
    TEXT_W_PT, TEXT_H_PT, FLOOR_PT, TARGET_PT, COL_PT,
    CAPTION_CHARS_PER_LINE, height_budget_pt,
)

SCALE_TOL = 0.005

ALLOWED_FONTS = {
    "TeXGyreHeros-Regular", "TeXGyreHeros-Bold",
    "TeXGyreHeros-Italic", "TeXGyreHeros-BoldItalic",
}

# fig1's architecture plate is an externally authored Illustrator/Inkscape
# asset. Its typography cannot be reached from Python, so it is exempted --
# loudly, by name, so the exemption gets revisited rather than forgotten.
FONT_EXEMPT: dict[str, set[str]] = {
    "fig1": {"Arial-BoldMT", "ArialMT", "Arial-BoldItalicMT", "Tahoma-Bold"},
}
# Only the imported architecture plate has a fixed aspect ratio it cannot
# negotiate. Every other page, fig1 included, is assembled by us and must come
# out at the text width exactly.
SCALE_EXEMPT: set[str] = set()

_INCLUDE_RE = re.compile(
    r"\\includegraphics\s*(?:\[(?P<opts>[^\]]*)\])?\s*\{(?P<stem>[^}]+)\}")
_CAPTION_RE = re.compile(r"\\caption\{")


def _frac(opts: str, key: str, base: str) -> float | None:
    """Pull `key=<number><base>` out of an includegraphics option list.

    Handles the three spellings in use: `.9\\linewidth`, `\\linewidth` (bare,
    meaning 1.0) and `0.75\\textwidth`.
    """
    m = re.search(rf"{key}\s*=\s*([0-9.]*)\\{base}", opts)
    if not m:
        return None
    return float(m.group(1)) if m.group(1) not in ("", ".") else 1.0


def latex_includes(main_tex: Path) -> dict[str, dict]:
    """stem -> {w_frac, h_frac, keepaspect, line, caption_lines}.

    Widths are resolved against the enclosing float: \\linewidth is the text
    width inside a `figure*` and the column width inside a single-column
    `figure` (Fig. 1, the model schematic).
    """
    text = main_tex.read_text()
    lines = text.splitlines()
    out: dict[str, dict] = {}
    for i, line in enumerate(lines, start=1):
        m = _INCLUDE_RE.search(line)
        if not m:
            continue
        opts = m.group("opts") or ""
        stem = Path(m.group("stem")).stem
        w = _frac(opts, "width", "linewidth") or _frac(opts, "width", "textwidth")
        h = _frac(opts, "height", "textheight")
        float_env = next((m2.group(1) for k in range(i - 1, max(i - 15, 0) - 1, -1)
                          if (m2 := re.search(r"\\begin\{(figure\*?)\}", lines[k]))), "figure*")
        out[stem] = {
            "base_w": TEXT_W_PT if float_env == "figure*" else COL_PT,
            "w_frac": w if w is not None else 1.0,
            "h_frac": h,
            "keepaspect": "keepaspectratio" in opts,
            "line": i,
            "caption_lines": _caption_lines(lines, i),
        }
    return out


def _caption_lines(lines: list[str], include_line: int) -> int:
    """Estimate how many typeset lines this float's caption occupies.

    Scans forward from the \\includegraphics to the \\caption in the same
    float and divides its length by the characters that fit on a 505.69 pt
    line at 7 pt. Only used to derive a height budget, so an estimate is
    enough -- but it must not be wildly optimistic, hence the +1 rounding.
    """
    for j in range(include_line, min(include_line + 12, len(lines))):
        if _CAPTION_RE.search(lines[j]):
            body = lines[j]
            # Strip LaTeX markup that does not consume horizontal space.
            body = re.sub(r"\\(protect|subref|textbf|Cref|emph|text|label)\b", "", body)
            body = re.sub(r"[{}\\$]", "", body)
            return max(2, len(body) // CAPTION_CHARS_PER_LINE + 1)
    return 6


def page_size_pt(pdf: Path) -> tuple[float, float]:
    root = _stext(pdf)
    page = root.find("page")
    return float(page.get("width")), float(page.get("height"))


def _stext(pdf: Path) -> ET.Element:
    """Structured text for `pdf`, with sizes already in page coordinates."""
    proc = subprocess.run(["mutool", "draw", "-F", "stext", "-o", "-", str(pdf)],
                          capture_output=True, check=True)
    return ET.fromstring(proc.stdout)


def spans(pdf: Path) -> list[tuple[str, float, str]]:
    """[(font_name, size_pt, text)] for every text span on page 1."""
    out = []
    for font in _stext(pdf).iter("font"):
        chars = [c.get("c", "") for c in font.iter("char")]
        text = "".join(chars).strip()
        if text:
            out.append((font.get("name", "?"), float(font.get("size", 0.0)), text))
    return out


def include_scale(pdf_w: float, pdf_h: float, inc: dict) -> float:
    """The factor LaTeX applies. With keepaspectratio a height cap can win."""
    s = inc["w_frac"] * inc.get("base_w", TEXT_W_PT) / pdf_w
    if inc["h_frac"] is not None:
        s_h = inc["h_frac"] * TEXT_H_PT / pdf_h
        s = min(s, s_h) if inc["keepaspect"] else s
    return s


def check(stem: str, pdf: Path, inc: dict) -> dict:
    pdf_w, pdf_h = page_size_pt(pdf)
    scale = include_scale(pdf_w, pdf_h, inc)
    sp = spans(pdf)
    sizes = sorted(s * scale for _, s, _ in sp)
    fonts = {f.split("+")[-1] for f, _, _ in sp}
    foreign = sorted(fonts - ALLOWED_FONTS - FONT_EXEMPT.get(stem, set()))
    below = [(f.split("+")[-1], round(s * scale, 2), t)
             for f, s, t in sp if s * scale < FLOOR_PT]
    budget = height_budget_pt(inc["caption_lines"])

    res = {
        "stem": stem, "pdf_w": round(pdf_w, 2), "pdf_h": round(pdf_h, 2),
        "w_frac": inc["w_frac"], "h_frac": inc["h_frac"],
        "scale": round(scale, 4),
        "on_page_w": round(pdf_w * scale, 1), "on_page_h": round(pdf_h * scale, 1),
        "budget_h": round(budget, 1),
        "n_spans": len(sp),
        "min_pt": round(sizes[0], 2) if sizes else None,
        "p05_pt": round(sizes[len(sizes) // 20], 2) if sizes else None,
        "median_pt": round(sizes[len(sizes) // 2], 2) if sizes else None,
        "max_pt": round(sizes[-1], 2) if sizes else None,
        "frac_below_floor": round(sum(s < FLOOR_PT for s in sizes) / len(sizes), 3) if sizes else 0.0,
        "foreign_fonts": foreign,
        "below_floor_examples": below[:6],
        "failures": [],
    }

    if stem not in SCALE_EXEMPT and abs(scale - 1.0) > SCALE_TOL:
        res["failures"].append(
            f"scale {scale:.4f} != 1 (artwork {pdf_w:.1f} pt wide, included at "
            f"{inc['w_frac']:g}x textwidth): author the page at "
            f"{inc['w_frac'] * inc.get('base_w', TEXT_W_PT):.2f} pt and include at width=\\linewidth")
    if sizes and sizes[0] < FLOOR_PT:
        worst = below[0]
        res["failures"].append(
            f"{res['frac_below_floor']:.0%} of text below {FLOOR_PT} pt; "
            f"smallest {sizes[0]:.2f} pt -- {worst[0]} {worst[2][:40]!r}")
    if pdf_h * scale > budget:
        res["failures"].append(
            f"artwork {pdf_h * scale:.0f} pt tall exceeds the "
            f"{budget:.0f} pt left by a {inc['caption_lines']}-line caption "
            f"(over by {pdf_h * scale - budget:.0f} pt)")
    if foreign:
        res["failures"].append(f"foreign fonts: {', '.join(foreign)}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="STEM")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--baseline", type=Path,
                    help="a previous --json record; report the delta")
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--generated", action="store_true",
                    help="measure the generators' own output instead of the "
                         "synced copies, so a figure can be gated before it "
                         "reaches the manuscript")
    args = ap.parse_args()

    includes = latex_includes(MAIN_TEX)
    if args.generated:
        sys.path.insert(0, str(REPO / "code"))
        from sync_paper_figures import FIGURES          # the same allowlist
        sources = FIGURES
    else:
        sources = {stem: args.fig_dir / f"{stem}.pdf" for stem in includes}

    results = []
    for stem, pdf in sorted(sources.items()):
        if args.only and stem not in args.only:
            continue
        if not Path(pdf).exists():    # hand-drawn PNG plates; not-yet-built stems
            continue
        # A stem with no \includegraphics yet (a figure being split out) is
        # measured as if it were included at the full text width, which is
        # what it will be.
        inc = includes.get(stem, {"w_frac": 1.0, "h_frac": None,
                                  "keepaspect": False, "line": 0,
                                  "caption_lines": 8})
        results.append(check(stem, Path(pdf), inc))

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    base = {}
    if args.baseline and args.baseline.exists():
        base = {r["stem"]: r for r in json.loads(args.baseline.read_text())}

    print(f"{'figure':<20}{'scale':>7}{'min pt':>8}{'p05':>7}{'med':>7}"
          f"{'<5pt':>7}{'height':>14}  status")
    print("-" * 92)
    n_fail = 0
    for r in results:
        delta = ""
        if r["stem"] in base and base[r["stem"]]["min_pt"] is not None:
            d = r["min_pt"] - base[r["stem"]]["min_pt"]
            delta = f"  ({d:+.2f})"
        status = "ok" if not r["failures"] else f"FAIL x{len(r['failures'])}"
        n_fail += bool(r["failures"])
        print(f"{r['stem']:<20}{r['scale']:>7.3f}{r['min_pt']:>8.2f}{r['p05_pt']:>7.2f}"
              f"{r['median_pt']:>7.2f}{r['frac_below_floor']:>6.0%}"
              f"{r['on_page_h']:>8.0f}/{r['budget_h']:<5.0f}  {status}{delta}")
    print("-" * 92)
    for r in results:
        if r["failures"]:
            print(f"\n{r['stem']}:")
            for f in r["failures"]:
                print(f"  - {f}")
    print(f"\n{len(results) - n_fail}/{len(results)} figures pass "
          f"(floor {FLOOR_PT} pt, target {TARGET_PT} pt).")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
