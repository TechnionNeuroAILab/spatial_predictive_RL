"""Declarative page layout for multi-panel figures.

Replaces nine near-identical `assemble*()` functions, one per figure directory,
each with its own margins (1/5/8/10 mm), gaps (2/3/3.5/5 mm), and its own idea
of whether to centre, left-pack or scale a row.

**The inversion this module exists for.** The old assemblers measured each
panel PDF and *contain-fit* it into a slot:

    scale = min(slot_w / panel_w, slot_h / panel_h)

so the panel's declared font sizes were multiplied by whatever that came to --
0.55 in fig1's panel a, 1.30 in figS_hypotheses' panels b/c. Nothing in the
panel source hinted at it. Combined with the `\\includegraphics` scale, printed
type across the paper ranged from 2.07 pt to 10.13 pt against a 5 pt floor.

Here the layout owns *both* dimensions of every slot, a panel is authored at
exactly `slot_size_in(key)`, and `compose()` only translates. A scale other
than 1 is no longer a silent adjustment -- it is `LayoutScaleError`, and the
message hands back the figsize the panel should have used.

That is why `row_h_in` is required rather than derived. Deriving row heights
from the measured panels (what `fig1/generate_figure1.py:_panel_slot_geometry`
did) means a panel can only be authored with a width, and the fit can never be
1:1 in both axes.

Typical use:

    LAYOUT = Layout([("a", 0, 0, 4), ("b", 1, 0, 2), ("c", 1, 2, 2)],
                    ncols=4, row_h_in=[1.55, 1.30], name="fig3",
                    caption_lines=8)

    fig = panel_figure(LAYOUT, "b")      # figsize is not yours to choose
    ...
    save_panel(fig, LAYOUT, "b", panels_dir / "panel_b")
    compose(LAYOUT, pdfs, out_dir / "figure3_raw")
"""
from __future__ import annotations

import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Collection, Mapping, Sequence

import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation

from pagegeom import MM_TO_PT, PAGE_W, height_budget_pt
from pubstyle import FS_PANEL, FS_TICK, save_figure, ensure_dir

# Defaults chosen once, for the whole figure set.
GUTTER_PT = 3.0 * MM_TO_PT      # between columns
ROW_GAP_PT = 3.0 * MM_TO_PT     # between rows, on top of the letter strip
MARGIN_PT = 0.0                 # the float is already inset by \textwidth
LETTER_STRIP_PT = 10.0          # reserved above each row for the panel letter
SCALE_TOL = 0.01


class LayoutScaleError(RuntimeError):
    """A panel does not fill its slot at scale 1."""


class LayoutHeightError(RuntimeError):
    """The page is taller than the float can be once the caption is set."""


@dataclass(frozen=True)
class Slot:
    """A panel's rectangle on the page, in PDF points from the bottom-left."""
    key: str
    x_pt: float
    y_pt: float
    w_pt: float
    h_pt: float
    letter_y_pt: float          # top of the reserved strip above the slot

    @property
    def size_in(self) -> tuple[float, float]:
        return self.w_pt / 72.0, self.h_pt / 72.0


def _normalise(spec, ncols: int | None) -> tuple[list[tuple[str, int, int, int]], int]:
    """Accept the three spec dialects already in use in this repo.

    1. explicit  [("a", row, col, colspan), ...]  with ncols
    2. row lists [("a", "b"), ("c", "d", "e")]    -- equal columns per row
    3. mosaic    "aaaa\\nbbcc\\nddef"              -- "." is empty
    """
    if isinstance(spec, str):
        rows = [r.strip() for r in spec.strip().splitlines()]
        n = len(rows[0])
        if any(len(r) != n for r in rows):
            raise ValueError("mosaic rows must all be the same length")
        placed: list[tuple[str, int, int, int]] = []
        seen: set[str] = set()
        for r, row in enumerate(rows):
            c = 0
            while c < n:
                ch = row[c]
                span = 1
                while c + span < n and row[c + span] == ch:
                    span += 1
                if ch != "." and ch not in seen:
                    placed.append((ch, r, c, span))
                    seen.add(ch)
                c += span
        return placed, n

    spec = list(spec)
    if spec and isinstance(spec[0], (tuple, list)) and len(spec[0]) == 4 \
            and isinstance(spec[0][1], int):
        if ncols is None:
            raise ValueError("explicit (key,row,col,colspan) spec needs ncols=")
        return [tuple(s) for s in spec], ncols

    # row-lists: give every row an equal share of a common column count
    n = _lcm_of_lengths([len(r) for r in spec])
    placed = []
    for r, row in enumerate(spec):
        span = n // len(row)
        for i, key in enumerate(row):
            placed.append((key, r, i * span, span))
    return placed, n


def _lcm_of_lengths(lengths: Sequence[int]) -> int:
    from math import gcd
    n = 1
    for length in lengths:
        n = n * length // gcd(n, length)
    return n


class Layout:
    """A grid of slots on a fixed-width page.

    `row_h_in` gives the drawable height of each row in inches, excluding the
    letter strip. `caption_lines` turns on the height budget check: pass the
    number of typeset lines the caption occupies in `main.tex` and a layout
    that cannot fit on the page with it raises at construction time -- before
    any data is loaded, which matters when the figure takes 19 minutes.
    """

    def __init__(self, spec, row_h_in: Sequence[float], *, ncols: int | None = None,
                 page_w_pt: float = PAGE_W, gutter_pt: float = GUTTER_PT,
                 row_gap_pt: float = ROW_GAP_PT, margin_pt: float = MARGIN_PT,
                 letter_strip_pt: float = LETTER_STRIP_PT,
                 caption_lines: int | None = None,
                 max_page_h_pt: float | None = None,
                 name: str = "figure") -> None:
        placed, ncols = _normalise(spec, ncols)
        n_rows = 1 + max(r for _, r, _, _ in placed)
        if len(row_h_in) != n_rows:
            raise ValueError(
                f"{name}: layout has {n_rows} rows but row_h_in has "
                f"{len(row_h_in)} entries")

        self.name = name
        self.ncols = ncols
        self.page_w_pt = page_w_pt
        self.letter_strip_pt = letter_strip_pt
        self._order = [k for k, _, _, _ in placed]

        col_w = (page_w_pt - 2 * margin_pt - (ncols - 1) * gutter_pt) / ncols
        row_h_pt = [h * 72.0 for h in row_h_in]
        strip = letter_strip_pt
        self.page_h_pt = (2 * margin_pt + sum(row_h_pt) + n_rows * strip
                          + (n_rows - 1) * row_gap_pt)

        budget = max_page_h_pt
        if budget is None and caption_lines is not None:
            budget = height_budget_pt(caption_lines)
        if budget is not None and self.page_h_pt > budget:
            tallest = max(range(n_rows), key=lambda r: row_h_pt[r])
            raise LayoutHeightError(
                f"{name}: page is {self.page_h_pt:.0f} pt tall but only "
                f"{budget:.0f} pt is left once the caption is set "
                f"(over by {self.page_h_pt - budget:.0f} pt). "
                f"Row {tallest} is the tallest at {row_h_pt[tallest]:.0f} pt "
                f"({row_h_in[tallest]:.2f} in); shrink it, or move panels into "
                f"a supplement.")

        self._slots: dict[str, Slot] = {}
        for key, row, col, span in placed:
            # Rows run top-down; PDF coordinates run bottom-up.
            above = sum(row_h_pt[r] + strip + row_gap_pt for r in range(row))
            row_top = self.page_h_pt - margin_pt - above
            self._slots[key] = Slot(
                key=key,
                x_pt=margin_pt + col * (col_w + gutter_pt),
                y_pt=row_top - strip - row_h_pt[row],
                w_pt=span * col_w + (span - 1) * gutter_pt,
                h_pt=row_h_pt[row],
                letter_y_pt=row_top,
            )

    # -- queries ------------------------------------------------------------
    def slot(self, key: str) -> Slot:
        try:
            return self._slots[key]
        except KeyError:
            raise KeyError(
                f"{self.name}: no slot {key!r}; layout has "
                f"{', '.join(self._order)}") from None

    def slot_size_in(self, key: str) -> tuple[float, float]:
        """The figsize this panel must be authored at."""
        return self.slot(key).size_in

    def slot_width_in(self, key: str) -> float:
        """Width only. Kept because fig1's panel factories take `width_in=`."""
        return self.slot(key).w_pt / 72.0

    def page_size_pt(self) -> tuple[float, float]:
        return self.page_w_pt, self.page_h_pt

    def keys(self) -> list[str]:
        return list(self._order)

    def __repr__(self) -> str:
        return (f"<Layout {self.name} {self.page_w_pt:.0f}x{self.page_h_pt:.0f} pt, "
                f"{len(self._slots)} slots>")


# --------------------------------------------------------------- authoring
def panel_figure(layout: Layout, key: str, **kw) -> plt.Figure:
    """A figure sized to its slot. The figsize is not the caller's to choose."""
    return plt.figure(figsize=layout.slot_size_in(key), **kw)


def panel_subplots(layout: Layout, key: str, nrows: int = 1, ncols: int = 1, **kw):
    """`plt.subplots` at the slot size."""
    return plt.subplots(nrows, ncols, figsize=layout.slot_size_in(key), **kw)


def save_panel(fig, layout: Layout, key: str, stem, *, dpi: int = 300):
    """Save at exactly the slot size, and complain if the content overflows.

    `save_figure(exact_size=True)` clips instead of shrinking, and warns when
    it has to -- see `pubstyle._warn_on_overflow`. That trades the old silent
    failure (oversized panel, scaled down at assembly, 7 pt type printing at
    3 pt) for a loud one at the panel that can fix it.
    """
    return save_figure(fig, stem, dpi=dpi, exact_size=True)


# ------------------------------------------------------------- composition
def compose(layout: Layout, panel_pdfs: Mapping[str, Path], out_stem,
            *, letters: bool | Mapping[str, str] = True,
            titles: Mapping[str, str] | None = None,
            fit: Collection[str] = (), tol: float = SCALE_TOL,
            dpi: int = 300) -> tuple[Path, Path]:
    """Translate each panel PDF into its slot; write `<stem>.pdf` and `.png`.

    Returns `(png_path, pdf_path)` -- the contract every current `assemble*()`
    already has, so `regenerate_all.py` and `sync_paper_figures.py` need no
    changes.

    `fit` names panels allowed to be scaled, for externally authored artwork
    with an aspect ratio we do not control. Keep it empty wherever possible;
    each entry is a panel whose type size is not governed by the type scale.
    """
    out_stem = Path(out_stem).with_suffix("")
    ensure_dir(out_stem.parent)
    pdf_path, png_path = out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")

    page_w, page_h = layout.page_size_pt()
    writer = PdfWriter()
    page = writer.add_blank_page(width=page_w, height=page_h)

    drawn = []
    for key, src_path in panel_pdfs.items():
        slot = layout.slot(key)
        src = PdfReader(str(src_path)).pages[0]
        box = src.mediabox
        iw, ih = float(box.width), float(box.height)
        sx, sy = slot.w_pt / iw, slot.h_pt / ih

        if key in fit:
            scale = min(sx, sy)
            dx = slot.x_pt + (slot.w_pt - iw * scale) / 2.0
            dy = slot.y_pt + (slot.h_pt - ih * scale) / 2.0
        elif abs(sx - 1.0) > tol or abs(sy - 1.0) > tol:
            raise LayoutScaleError(_scale_message(layout, key, slot, iw, ih, sx, sy, tol))
        else:
            scale, dx, dy = 1.0, slot.x_pt, slot.y_pt

        page.merge_transformed_page(
            src, Transformation().scale(scale).translate(dx, dy), over=True)
        drawn.append(key)

    if letters:
        mapping = letters if isinstance(letters, Mapping) else {k: k for k in drawn}
        with tempfile.TemporaryDirectory() as tmp:
            overlay = letters_overlay(layout, mapping, Path(tmp) / "letters.pdf",
                                      titles=titles)
            page.merge_page(PdfReader(str(overlay)).pages[0], over=True)

    with open(pdf_path, "wb") as fh:
        writer.write(fh)
    _pdf_to_png(pdf_path, png_path, dpi=dpi)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    return png_path, pdf_path


def _scale_message(layout, key, slot, iw, ih, sx, sy, tol) -> str:
    w_in, h_in = slot.size_in
    # A contain-fit would apply the smaller of the two, and that is what the
    # panel's type would be multiplied by.
    applied = min(sx, sy)
    return (
        f"\n{layout.name}: panel {key!r} does not fill its slot at scale 1.\n"
        f"    slot        {slot.w_pt:7.2f} x {slot.h_pt:7.2f} pt   "
        f"({w_in:.3f} x {h_in:.3f} in)\n"
        f"    panel PDF   {iw:7.2f} x {ih:7.2f} pt   "
        f"({iw / 72:.3f} x {ih / 72:.3f} in)\n"
        f"    scale       {sx:7.3f} x {sy:7.3f}       (tolerance {tol:.1%})\n"
        f"    Fitting it would apply {applied:.3f}, so a 7.0 pt label would "
        f"print at {7.0 * applied:.2f} pt and a 5.5 pt one\n"
        f"    at {5.5 * applied:.2f} pt -- the type scale would not survive "
        f"assembly, which is the defect\n"
        f"    this layout exists to prevent.\n"
        f"    Fix: author the panel at its slot size --\n"
        f"        fig = figlayout.panel_figure(LAYOUT, {key!r})   "
        f"# figsize=({w_in:.3f}, {h_in:.3f})\n"
        f"        figlayout.save_panel(fig, LAYOUT, {key!r}, stem)\n"
        f"    or, for artwork whose aspect ratio is fixed elsewhere, pass "
        f"fit={{{key!r}}} to compose().")


def letters_overlay(layout: Layout, letters: Mapping[str, str], out_path: Path,
                    *, titles: Mapping[str, str] | None = None,
                    size: float = FS_PANEL) -> Path:
    """A full-page transparent PDF carrying the panel letters.

    One convention for the whole paper: lowercase, bold, `size` pt, black,
    anchored at the top-left of the slot's reserved strip -- so a letter can
    never land on panel content, which is what the four previous mechanisms
    (two of which used a different size) each risked in their own way.

    Saved with an explicit Bbox rather than `bbox_inches=None`. Matplotlib
    resolves `None` to `rcParams["savefig.bbox"]`, which `apply_nature_style`
    sets to "tight"; the existing overlay writers only escape that because
    `add_axes([0, 0, 1, 1])` happens to make the tight bbox equal the page.
    """
    from matplotlib.transforms import Bbox

    ensure_dir(out_path.parent)
    page_w, page_h = layout.page_size_pt()
    fig = plt.figure(figsize=(page_w / 72.0, page_h / 72.0))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    ax.patch.set_alpha(0.0)

    for key, letter in letters.items():
        slot = layout.slot(key)
        ax.text(slot.x_pt, slot.letter_y_pt, letter, fontsize=size,
                fontweight="bold", va="top", ha="left", color="#000000")
        if titles and key in titles:
            ax.text(slot.x_pt + size * 1.15, slot.letter_y_pt, titles[key],
                    fontsize=FS_TICK, va="top", ha="left", color="#1A1A1A",
                    linespacing=1.25)

    fig.savefig(out_path, format="pdf", transparent=True,
                bbox_inches=Bbox.from_bounds(0, 0, page_w / 72.0, page_h / 72.0),
                pad_inches=0.0)
    plt.close(fig)
    return out_path


def existing_panels(panels_dir: Path, keys, prefix: str = "panel_") -> dict[str, Path]:
    """Collect already-rendered panel PDFs, for `--assemble_only`.

    Layout is cheap to get wrong and expensive to re-derive: figS_dissociation
    spends 1147 s re-reading pickles before it draws anything. The panel PDFs
    are already persisted, so iterating on the page geometry should cost
    seconds, not a coffee break.
    """
    pdfs = {}
    for key in keys:
        pdf = Path(panels_dir) / f"{prefix}{key}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(
                f"--assemble_only needs {pdf}, which does not exist. "
                f"Run the generator once without the flag to render the panels.")
        pdfs[key] = pdf
    return pdfs


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    """Rasterise a one-page PDF with Poppler's pdftoppm (already a hard dep)."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                        str(pdf_path), str(prefix)], check=True)
        copy2(prefix.with_suffix(".png"), png_path)
