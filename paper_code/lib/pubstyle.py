"""Single source of truth for publication figure style.

Before this module, each of the ten figure directories carried its own copy of
`apply_nature_style()` and its own ad-hoc colours, and the rate maps were drawn
in `jet` (37 call sites). That made the figure set look assembled rather than
designed. Everything visual now comes from here; each figure's `utils.py`
re-exports these names so existing call sites keep working.

Design decisions and why:

TYPEFACE -- TeX Gyre Heros, a high-quality Helvetica clone. Arial and Helvetica
are not installed on this machine, so the previous font stack silently fell back
to DejaVu Sans (matplotlib's default, and instantly recognisable as such).
Nature/Science figures are set in Helvetica or Arial; Heros matches those metrics.

COLOUR -- the Okabe-Ito colourblind-safe qualitative palette, which is the
accessibility reference set: distinguishable under deuteranopia, protanopia and
tritanopia, and separable in greyscale. Colours are exposed by ROLE
(`C["pre"]`, `C["head"]`, ...) not by hue, so a condition keeps one colour in
every panel of every figure.

RATE MAPS -- `RATEMAP`, a custom white -> amber -> vermillion -> maroon ramp,
chosen by rendering real sparse place fields through eight candidates as an
actual 2x8 tile grid (the way they appear in figs 1-4) and comparing:
  jet/turbo      perceptually non-uniform; invents structure in flat background
  viridis/cividis dark ground buries small fields
  magma/inferno  fields pop, but sixteen black tiles dominate the page
  magma_r/YlOrRd light, but a saturated cream ground reads as muddy
RATEMAP is monotonic in lightness (so it survives greyscale printing and
photocopying), reads intuitively as hot = active, and keeps the page white.
The floor is a barely-tinted white rather than pure white so the arena is still
distinguishable from the page, with a light grey frame for the same reason.

SIZES -- Nature: 89 mm single column, 183 mm double, and no text below 5 pt at
final size. Type scale here is 7 pt body / 6 pt ticks / 8 pt bold panel letters.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------- metrics
MM_TO_PT = 72.0 / 25.4
COL_SINGLE = 89.0 * MM_TO_PT      # Nature single column
COL_WIDE = 120.0 * MM_TO_PT       # 1.5 column
COL_DOUBLE = 183.0 * MM_TO_PT     # full page width
PAGE_W = COL_DOUBLE               # historical name used across the figure dirs

FS_BASE = 7.0
FS_TICK = 6.0
FS_SMALL = 5.5
FS_PANEL = 8.0                    # bold panel letters
LABEL_SIZE = FS_PANEL             # historical name

FONT_STACK = ["TeX Gyre Heros", "Nimbus Sans", "Helvetica", "Arial",
              "Inter", "DejaVu Sans"]

# ---------------------------------------------------------------------- colour
# Okabe-Ito, the colourblind-safe reference palette.
OI = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "verm":    "#D55E00",
    "purple":  "#CC79A7",
}

# Semantic roles. A condition keeps its colour in every panel of every figure.
C = {
    # learning phases / probes
    "pre":       OI["blue"],       # Phase A, prediction, pre-RL
    "policy":    OI["verm"],       # Phase B, RL policy / goal-directed
    "frozen_pre":  OI["green"],    # Phase C, frozen replay, pre-RL weights
    "frozen_post": OI["purple"],   # Phase C, frozen replay, post-RL weights
    # model variants
    "nohead":    OI["skyblue"],
    "head":      OI["verm"],
    "rigid":     "#7A7A7A",
    # reward-relocation phases
    "reward_a":  OI["blue"],
    "reward_b":  OI["orange"],
    # confrontation with experiment
    "model":     OI["blue"],
    "data":      OI["black"],
    # extinction timeline
    "pred":      OI["blue"],
    "postlearn": OI["verm"],
    "postext":   OI["green"],
    # accents / furniture
    "reward":    "#00A0B0",        # reward marker: visible on the light ramp
    "muted":     "#6E6E6E",
    "grid":      "#D9D9D9",
    "zero":      "#9A9A9A",
}

# Backwards-compatible aliases used by existing panel code.
COLOR_NOHEAD = C["nohead"]
COLOR_HEAD = C["head"]
COLOR_MODEL = C["model"]
COLOR_DATA = C["data"]
COLOR_PRED = C["pred"]
COLOR_POSTLEARN = C["postlearn"]
COLOR_POSTEXT = C["postext"]
PHASE_COLORS = {"A": C["pre"], "B": C["policy"],
                "C_pre": C["frozen_pre"], "C_post": C["frozen_post"]}
COLOR_A = C["reward_a"]           # reward A (pre-relocation)
COLOR_B = C["reward_b"]           # reward B (post-relocation)
COLOR_RIGID = C["rigid"]
COLOR_DISTANCE = OI["verm"]       # Gauthier distance cells
COLOR_GOALDIR = OI["blue"]        # Sarel goal-direction cells
COLOR_HD = C["muted"]             # head-direction control
COLOR_REWARD = C["reward"]
COLOR_MUTED = C["muted"]

# ------------------------------------------------------------------- colormaps
# Stops are POSITIONED, not evenly spaced: the low third stays almost white so a
# sparse place field still reads as sparse. With evenly spaced stops the ramp
# reaches full amber by 0.33 and small fields visually inflate into fat blobs --
# a real perceptual change relative to the jet original, not just a recolour.
RATEMAP = LinearSegmentedColormap.from_list("pub_ratemap", [
    (0.00, "#FFFDFA"),   # near-white floor: arena reads as empty, not as colour
    (0.20, "#FFF7E2"),
    (0.40, "#FFE49E"),
    (0.56, "#FBC148"),
    (0.70, "#F0821A"),
    (0.83, "#D6410A"),
    (0.93, "#97200F"),
    (1.00, "#4A0C08"),
])
RATEMAP.set_bad("#F2F2F2")

# Signed quantities (shift toward/away from reward): blue-white-red, balanced
# lightness either side of the neutral point so neither pole dominates.
DIVERGING = LinearSegmentedColormap.from_list("pub_div", [
    "#2166AC", "#7BA7CF", "#D3E3F0", "#FFFFFF", "#FBDDD0", "#DE8A66", "#B2182B"])
DIVERGING.set_bad("#F2F2F2")

# Occupancy / density / correlation matrices.
SEQUENTIAL = "magma"

RATEMAP_CMAP = RATEMAP            # historical name in the figure dirs

for _cm in (RATEMAP, DIVERGING):
    try:
        mpl.colormaps.register(_cm)
    except (ValueError, AttributeError):
        pass


# ----------------------------------------------------------------------- style
def apply_nature_style() -> None:
    """Install the publication rcParams. Idempotent; call once per process."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "mathtext.fontset": "dejavusans",
        "font.size": FS_BASE,
        "axes.labelsize": FS_BASE,
        "axes.titlesize": FS_BASE,
        "figure.titlesize": FS_BASE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,

        # Lighter furniture: thin spines, short outward ticks, no box.
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1A1A1A",
        "axes.titlelocation": "left",
        "axes.titlepad": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelcolor": "#1A1A1A",
        "ytick.labelcolor": "#1A1A1A",

        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.4,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.2,
        "legend.columnspacing": 1.0,

        "grid.color": C["grid"],
        "grid.linewidth": 0.4,
        "axes.grid": False,

        "image.cmap": "pub_ratemap",
        "image.interpolation": "nearest",

        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        # TrueType, so text stays editable in Illustrator/Inkscape.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


# --------------------------------------------------------------------- helpers
def panel_label(ax, letter: str, dx: float = -0.026, dy: float = 0.012,
                size: float = FS_PANEL, **kw):
    """Bold lowercase panel letter in figure-relative position above-left."""
    return ax.figure.text(
        ax.get_position().x0 + dx, ax.get_position().y1 + dy, letter,
        fontsize=size, fontweight="bold", va="bottom", ha="left",
        color="#000000", **kw)


def title(ax, text: str, **kw):
    """Short, left-aligned panel title. Titles name the panel; the caption
    carries the sentence. Long strings are a smell, so they are flagged."""
    if len(text) > 48:
        import warnings
        warnings.warn(f"panel title is caption-length ({len(text)} chars): {text[:60]!r}",
                      stacklevel=2)
    return ax.set_title(text, loc="left", fontsize=FS_BASE, pad=3, **kw)


def fmt_p(p: float | None, italic: bool = True) -> str:
    """APA-ish p formatting: no '0.05000000001', no 'p = n/a'."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    lead = "$P$" if italic else "P"
    if p < 1e-4:
        exp = int(np.floor(np.log10(p)))
        return f"{lead} < 10$^{{{exp + 1}}}$"
    if p < 0.001:
        return f"{lead} < 0.001"
    return f"{lead} = {p:.3f}".rstrip("0").rstrip(".") if p < 0.1 else f"{lead} = {p:.2f}"


def stat_annot(ax, text: str, loc: str = "upper right", size: float = FS_SMALL,
               color: str | None = None, pad: float = 0.02, **kw):
    """Put statistics INSIDE the axes in small grey type.

    Statistics stacked above a panel as a multi-line title (the previous
    pattern) reads as a caption fragment and pushes panels apart.
    """
    color = color or C["muted"]
    va, ha = ("top", "right")
    x, y = 1 - pad, 1 - pad
    if "lower" in loc:
        va, y = "bottom", pad
    if "left" in loc:
        ha, x = "left", pad
    if "center" in loc:
        ha, x = "center", 0.5
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=size,
                   color=color, va=va, ha=ha, linespacing=1.35, **kw)


def despine(ax, left: bool = True, bottom: bool = True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(left=False, labelleft=False)
    if not bottom:
        ax.tick_params(bottom=False, labelbottom=False)


def arena_axes(ax, frame: bool = True):
    """Style an axes that shows a rate map / arena: square, no ticks, hairline
    frame so the arena is bounded even where the map is near-white."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_linewidth(0.4)
        s.set_color("#C8C8C8")
    return ax


def reward_marker(ax, x: float, y: float, m: int | None = None, size: float = 6.0,
                  zorder: int = 6, halo: bool = True):
    """The reward location, drawn identically everywhere.

    A teal star inside a white keyline, plus an optional outer dark hairline, so
    it stays visible on the near-white floor of RATEMAP *and* on top of a
    saturated field peak. Small unhaloed markers disappeared into dark peaks.
    """
    if m is not None:            # data in bin units rather than [0, 1]
        x, y = x * m, y * m
    out = []
    if halo:
        out += ax.plot([x], [y], marker="*", ms=size + 1.6, mfc="none",
                       mec="white", mew=1.3, ls="none", zorder=zorder,
                       clip_on=False)
    out += ax.plot([x], [y], marker="*", ms=size, mfc=C["reward"],
                   mec="#0B3C44", mew=0.4, ls="none", zorder=zorder + 0.1,
                   clip_on=False)
    return out


def thin_colorbar(fig, mappable, ax, label: str = "", ticks=(0, 1),
                  ticklabels=("0", "max"), width: float = 0.012,
                  shrink: float = 0.55, pad: float = 0.008):
    """A short, thin colorbar with two ticks and the label on top.

    Rate maps previously had no colorbar at all; where they did, the bar was as
    tall as the panel with eleven redundant ticks."""
    pos = ax.get_position()
    h = pos.height * shrink
    cax = fig.add_axes([pos.x1 + pad, pos.y0 + (pos.height - h) / 2, width, h])
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks(list(ticks))
    cb.set_ticklabels(list(ticklabels))
    cax.tick_params(size=1.2, width=0.4, pad=1.5, labelsize=FS_SMALL)
    if label:
        cax.set_title(label, fontsize=FS_SMALL, pad=2.5, loc="center")
    return cb


def zero_line(ax, orient: str = "h", **kw):
    kw = {"color": C["zero"], "lw": 0.5, "ls": (0, (2.5, 2)), "zorder": 0.5, **kw}
    return ax.axhline(0, **kw) if orient == "h" else ax.axvline(0, **kw)


def style_violin(parts, face: str, alpha: float = 0.75, edge: float = 0.5):
    """Flat fill + thin matching edge; kills matplotlib's default blue-with-
    heavy-outline violins."""
    for b in parts.get("bodies", []):
        b.set_facecolor(face)
        b.set_edgecolor(face)
        b.set_alpha(alpha)
        b.set_linewidth(edge)
    for key in ("cmins", "cmaxes", "cbars", "cmedians", "cmeans"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(0.6)
    return parts


def hist_outline(ax, values, bins, color: str, label: str | None = None,
                 fill_alpha: float = 0.18, lw: float = 0.9, **kw):
    """Overlapping distributions as a light fill plus a crisp step outline.

    Two alpha-blended solid histograms (the previous pattern) turn muddy grey
    where they overlap and neither distribution stays readable."""
    ax.hist(values, bins=bins, color=color, alpha=fill_alpha,
            histtype="stepfilled", lw=0, **kw)
    return ax.hist(values, bins=bins, color=color, histtype="step",
                   lw=lw, label=label, **kw)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, stem, dpi: int = 300):
    """Write PNG (review) + PDF (vector, for the manuscript)."""
    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    png_path, pdf_path = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    for path, fmt in ((png_path, "png"), (pdf_path, "pdf")):
        fig.savefig(path, format=fmt, dpi=dpi, facecolor="white",
                    edgecolor="none", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path
