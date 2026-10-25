"""Single source of truth for publication figure style.

Before this module, each of the ten figure directories carried its own copy of
`apply_nature_style()` and its own ad-hoc colours, and the rate maps were drawn
in `jet` (37 call sites). That made the figure set look assembled rather than
designed. Everything visual now comes from here; each figure's `utils.py`
re-exports these names so existing call sites keep working.

Design decisions and why:

typeface -- TeX Gyre Heros, a high-quality Helvetica clone. Arial and Helvetica
are not installed on this machine, so the previous font stack silently fell back
to DejaVu Sans (matplotlib's default, and instantly recognisable as such).
Nature/Science figures are set in Helvetica or Arial; Heros matches those metrics.

colour -- the Okabe-Ito colourblind-safe qualitative palette, which is the
accessibility reference set: distinguishable under deuteranopia, protanopia and
tritanopia, and separable in greyscale. Colours are exposed by role
(`C["pre"]`, `C["head"]`, ...) not by hue, so a condition keeps one colour in
every panel of every figure.

rate maps -- `RATEMAP`, matplotlib's YlGnBu. A sequential yellow-green-blue
ramp: low rates stay pale so a sparse place field still reads as sparse, and
the peak is dark blue. It is registered under the historical name `pub_ratemap`
and re-exported by each figure directory's `utils.py`, so every place-field
panel picks it up without a per-call colormap. Scatter, violin, bar and
Okabe-Ito condition colours are not this ramp.

sizes -- Nature: 89 mm single column, 183 mm double, and no text below 5 pt at
final size. Type scale here is 7 pt body / 6 pt ticks / 8 pt bold panel letters.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------- metrics
# Page geometry lives in pagegeom.py, which code/check_figure_type.py also
# imports, so the size the figures are drawn at and the size the gate measures
# against cannot drift apart.
from pagegeom import (  # noqa: E402
    MM_TO_PT, TEXT_W_PT, TEXT_H_PT, COL_PT, FLOOR_PT, TARGET_PT,
    NATURE_SINGLE_PT, NATURE_DOUBLE_PT,
)

# The figure set used to be authored at Nature's 183 mm and then scaled by
# \includegraphics to fit a 177.7 mm text block -- a different factor per
# figure (0.61 to 1.13). Authoring at the text width makes declared size equal
# printed size. To target a 183 mm journal later, set PAGE_W to
# NATURE_DOUBLE_PT: the whole set then rescales uniformly, which is exactly the
# operation the old per-figure scales made impossible.
PAGE_W = TEXT_W_PT                # what a figure* spans
COL_SINGLE = COL_PT               # what a single-column figure spans
COL_WIDE = (TEXT_W_PT + COL_PT) / 2.0
COL_DOUBLE = PAGE_W               # historical name used across the figure dirs

FS_BASE = 7.0                     # axis labels, panel titles, body text
FS_TICK = 6.0                     # tick labels, legends
FS_SMALL = 5.5                    # in-axes statistics, colourbar labels
FS_PANEL = 8.0                    # bold panel letters
LABEL_SIZE = FS_PANEL             # historical name
FS_MIN = FLOOR_PT                 # nothing may print below this

# The whole type scale. `assert_type_scale` refuses to save a panel that uses
# anything else, which is what retires the ~240 one-off `fontsize=` literals
# (4.5, 4.8, 5.2, 5.4, 5.7, 6.3, 9.0 ...) scattered through the panel code.
TYPE_SCALE = (FS_SMALL, FS_TICK, FS_BASE, FS_PANEL)

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
# Place fields and rate maps. A copy so set_bad does not mutate the global
# YlGnBu, registered under the name call sites and image.cmap already use.
RATEMAP = mpl.colormaps["YlGnBu"].copy()
RATEMAP.name = "pub_ratemap"
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
        # Was "dejavusans", which is why pdffonts reported DejaVu *and* TeX
        # Gyre Heros in the same figure: every $\Delta$, $\times$ and
        # superscript silently switched typeface mid-sentence.
        "mathtext.fontset": "custom",
        "mathtext.rm": "TeX Gyre Heros",
        "mathtext.it": "TeX Gyre Heros:italic",
        "mathtext.bf": "TeX Gyre Heros:bold",
        "mathtext.sf": "TeX Gyre Heros",
        "mathtext.cal": "TeX Gyre Heros:italic",
        "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.default": "regular",
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
def panel_label(ax, letter: str, dx_pt: float = -11.0, dy_pt: float = 2.0,
                size: float = FS_PANEL, **kw):
    """Bold lowercase panel letter above and left of `ax`.

    Offsets are in POINTS. The previous signature took figure *fractions*,
    so the identical call produced an 13 pt offset on a full page and 4 pt on
    a small panel -- which is why it had no call sites and four incompatible
    hand-rolled mechanisms existed instead.

    For a composite assembled from panel PDFs, use
    `figlayout.compose(..., letters=True)`; it places letters in a strip
    reserved by the layout, where they cannot collide with panel content.
    """
    fig = ax.figure
    w_pt, h_pt = (d * 72.0 for d in fig.get_size_inches())
    pos = ax.get_position()
    return fig.text(pos.x0 + dx_pt / w_pt, pos.y1 + dy_pt / h_pt, letter,
                    fontsize=size, fontweight="bold", va="bottom", ha="left",
                    color="#000000", **kw)


def assert_type_scale(fig, *, strict: bool = True, tol: float = 0.05) -> list:
    """Report every Text artist whose size is not on the type scale.

    Uniform type across sixteen figures is not achievable by reviewing
    diffs -- there were ~240 hand-set sizes between 4.5 and 9.0 pt. It is
    achievable by refusing to save a panel that introduces a new one.

    Returns the offenders as (size, text) pairs; raises when `strict`.
    """
    bad = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        size = t.get_fontsize()
        if not any(abs(size - s) <= tol for s in TYPE_SCALE):
            bad.append((round(size, 2), str(t.get_text())[:40]))
    if bad and strict:
        listed = "\n".join(f"      {s:5.2f} pt  {txt!r}" for s, txt in sorted(set(bad))[:12])
        raise ValueError(
            f"off-scale type in this panel ({len(bad)} artists). "
            f"The scale is {', '.join(f'{s:g}' for s in TYPE_SCALE)} pt "
            f"(FS_SMALL, FS_TICK, FS_BASE, FS_PANEL):\n{listed}\n"
            f"    Round each to the nearest scale step rather than inventing "
            f"one: sizes below {FS_MIN:g} pt are illegible at final size, and "
            f"sizes between the steps are what made the figure set look "
            f"assembled rather than designed.")
    return bad


def title(ax, text: str, **kw):
    """Short, left-aligned panel title. Titles name the panel; the caption
    carries the sentence. Long strings are a smell, so they are flagged."""
    if len(text) > 48:
        import warnings
        warnings.warn(f"panel title is caption-length ({len(text)} chars): {text[:60]!r}",
                      stacklevel=2)
    return ax.set_title(text, loc="left", fontsize=FS_BASE, pad=3, **kw)


def fmt_p(p: float | None, italic: bool = True) -> str:
    """apa-ish p formatting: no '0.05000000001', no 'p = n/a'.

    Deliberately no scientific notation. mathtext sets a superscript at 0.7x
    the surrounding size, so `P < 10^{-6}` at FS_SMALL printed its exponent at
    3.85 pt -- the smallest type left anywhere in the figure set, and below the
    5 pt floor. TeX Gyre Heros has no Unicode superscript glyphs to fall back
    on either (U+2070, U+2074-2079 and U+207B are all absent), so a raised
    exponent here can only be illegible, foreign-fonted, or both.

    Panels therefore report the conventional floor and the exact exponent stays
    in the caption and Results text, where LaTeX sets it at 7 pt.
    """
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    lead = "$P$" if italic else "P"
    if p < 1e-4:
        return f"{lead} < 0.0001"
    if p < 0.001:
        return f"{lead} < 0.001"
    return f"{lead} = {p:.3f}".rstrip("0").rstrip(".") if p < 0.1 else f"{lead} = {p:.2f}"


def stat_annot(ax, text: str, loc: str = "upper right", size: float = FS_SMALL,
               color: str | None = None, pad: float = 0.02, **kw):
    """Put statistics inside the axes in small grey type.

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


def plain_log_ticks(ax, which: str = "both"):
    """Label a log axis 1, 10, 100 ... instead of 10^0, 10^1, 10^2.

    matplotlib's log formatter sets the exponent as a mathtext superscript,
    which renders at 0.7x the tick size -- 4.2 pt at FS_TICK, below the 5 pt
    floor -- and TeX Gyre Heros has no Unicode superscript glyphs to fall back
    on (U+2070, U+2074-2079 and U+207B are all absent), so the raised form can
    only be illegible, foreign-fonted, or both.
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def _fmt(v, _pos):
        if v <= 0:
            return ""
        return f"{v:g}" if 1e-3 <= abs(v) < 1e5 else f"{v:.0e}".replace("e+0", "e")

    axes = ((ax.xaxis,) if which == "x" else
            (ax.yaxis,) if which == "y" else (ax.xaxis, ax.yaxis))
    for a in axes:
        if a.get_scale() == "log":
            a.set_major_formatter(FuncFormatter(_fmt))
            a.set_minor_formatter(NullFormatter())
    return ax


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


# Field-centre metrics. Every shift panel shows both: peak solid, centre of
# mass (COM) as the lighter tint of the same hue, so the condition colour still
# carries the condition and the shade carries the metric.
METRICS = ("peak", "com")
METRIC_LABEL = {"peak": "Peak", "com": "Centre of mass"}
METRIC_ALPHA = {"peak": 1.0, "com": 0.45}


def sem(values) -> float:
    """Standard error of the mean, s.d.(ddof=1)/sqrt(n), ignoring NaN."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def mean_sem_marker(ax, x: float, values, color: str = "#000000", ms: float = 3.6,
                    zorder: int = 6, **kw):
    """Black diamond at the mean with an s.e.m. error bar."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return ax.errorbar([x], [v.mean()], yerr=[sem(v)], fmt="D", color=color,
                       ms=ms, mec="white", mew=0.4, elinewidth=0.9, capsize=2.0,
                       capthick=0.9, zorder=zorder, **kw)


def network_strip(ax, values_by_group, positions, colors, *, pair: bool = True,
                  jitter: float = 0.08, size: float = 5.0, alpha: float = 0.55,
                  seed: int = 0):
    """One point per network for each group, optional grey lines pairing a
    network across groups, and mean +/- s.e.m. diamonds."""
    rng = np.random.default_rng(seed)
    vals = [np.asarray(v, dtype=float) for v in values_by_group]
    xs = [p + rng.uniform(-jitter, jitter, len(v)) for p, v in zip(positions, vals)]
    if pair and len({len(v) for v in vals}) == 1:
        for i in range(len(vals[0])):
            ax.plot([x[i] for x in xs], [v[i] for v in vals], color="#BDBDBD",
                    lw=0.3, alpha=0.6, zorder=1)
    for x, v, c in zip(xs, vals, colors):
        ax.scatter(x, v, s=size, color=c, alpha=alpha, lw=0, zorder=2)
    for p, v in zip(positions, vals):
        mean_sem_marker(ax, p, v)
    return xs


def load_headline_stats() -> dict:
    """results/aggregate/headline_stats.json (code/aggregate/headline_stats.py)."""
    import json
    path = Path(__file__).resolve().parents[2] / "results" / "aggregate" / "headline_stats.json"
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _warn_on_overflow(fig, stem, tol_pt: float = 0.5) -> None:
    """Warn when content sits outside the canvas and will therefore be clipped.

    A fixed bbox trades an invisible failure for a visible one: the old tight
    crop silently produced an oversized panel that the assembler scaled down,
    taking the type with it. Clipping is only the better failure if someone is
    told about it, and the useful moment is here -- at the panel that can fix
    it -- rather than when a label turns up missing from the composite.
    """
    w_in, h_in = fig.get_size_inches()
    try:
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
    except Exception:                    # no renderer available yet
        return
    over = {"left": -bb.x0, "bottom": -bb.y0, "right": bb.x1 - w_in, "top": bb.y1 - h_in}
    bad = {k: v * 72.0 for k, v in over.items() if v * 72.0 > tol_pt}
    if bad:
        detail = ", ".join(f"{k} {v:.1f} pt" for k, v in sorted(bad.items()))
        warnings.warn(
            f"{Path(stem).name}: content overruns its {w_in:.2f}x{h_in:.2f} in "
            f"canvas ({detail}) and will be clipped. Usually a colourbar or "
            f"legend placed outside the axes -- give tight_layout a `rect` that "
            f"reserves room for it, or shorten the label.",
            stacklevel=3)


def save_figure(fig, stem, dpi: int = 300, *, exact_size: bool = False):
    """Write PNG (review) + PDF (vector, for the manuscript).

    With `exact_size`, the PDF measures exactly `fig.get_size_inches()` rather
    than being cropped to its content. That matters because the assemblers
    used to contain-fit whatever came out of a tight crop, so a panel that
    saved 4 % taller than its figsize was silently scaled down -- taking its
    type with it. A fixed bbox makes the panel's size a declaration instead of
    an outcome; `figlayout.save_panel` adds the overflow check that makes the
    clipping visible.

    The bbox is an explicit Bbox, never None: matplotlib resolves None to
    rcParams["savefig.bbox"], which `apply_nature_style` sets to "tight".
    """
    from matplotlib.transforms import Bbox

    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    png_path, pdf_path = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    if exact_size:
        _warn_on_overflow(fig, stem)
        bbox, pad = Bbox.from_bounds(0, 0, *fig.get_size_inches()), 0.0
    else:
        bbox, pad = "tight", 0.02
    for path, fmt in ((png_path, "png"), (pdf_path, "pdf")):
        fig.savefig(path, format=fmt, dpi=dpi, facecolor="white",
                    edgecolor="none", bbox_inches=bbox, pad_inches=pad)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path
