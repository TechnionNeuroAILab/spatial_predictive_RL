"""Geometry of the page the figures are actually printed on.

Separate from `pubstyle` on purpose. `pubstyle` answers "what should this look
like"; this module answers "how big is the hole it has to fit in", which is a
property of `paper_repo/biorxiv.cls` and nothing to do with taste. Keeping it
free of matplotlib also lets `code/check_figure_type.py` import it without
pulling in the plotting stack.

Every number below is read off `paper_repo/main.log`'s geometry block for
`\\documentclass[times,twoside,watermark]{biorxiv}`, not guessed from the class
source -- the class sets `\\textheight` twice and the second one wins.

The figure set was previously authored at Nature's 183 mm double-column
specification while the document's text block is 177.7 mm, and the
`\\includegraphics` calls then applied a further 0.61-1.13 scale each. Authoring
at TEXT_W_PT instead makes declared size equal printed size, which is the only
version of "uniform fonts" that can be checked mechanically.
"""
from __future__ import annotations

MM_TO_PT = 72.0 / 25.4

# --- the document ----------------------------------------------------------
TEXT_W_PT = 505.69374        # \textwidth   (177.7 mm) -- a figure* spans this
TEXT_H_PT = 691.40196        # \textheight  (243 mm)
COL_PT = 242.88841           # \columnwidth (85.4 mm) -- (TEXT_W - COLSEP)/2
COLSEP_PT = 19.91692         # \columnsep   (7 mm)

# Nature's own specification, kept for a future journal submission: changing
# PAGE_W to this rescales the whole set uniformly, which is the operation the
# old system could never perform.
NATURE_SINGLE_PT = 89.0 * MM_TO_PT
NATURE_DOUBLE_PT = 183.0 * MM_TO_PT

PAGE_W = TEXT_W_PT           # the width every figure* is authored at
PAGE_W_IN = PAGE_W / 72.0

# --- legibility ------------------------------------------------------------
FLOOR_PT = 5.0               # Nature's hard minimum at final size
TARGET_PT = 7.0              # what panel body text should measure

# --- how much height a float can spend -------------------------------------
# A figure* takes the artwork plus its caption. The caption is set 7/9 sans by
# biorxiv.cls (\DeclareCaptionFormat{smallformat}); 11 pt per line leaves slack
# for the \dblfloatsep above and the descenders below.
CAPTION_LEADING_PT = 11.0
DBL_FLOAT_SEP_PT = 20.0
CAPTION_CHARS_PER_LINE = 168     # 7 pt sans across TEXT_W_PT


def height_budget_pt(caption_lines: int) -> float:
    """Artwork height that still leaves the caption room on the same page."""
    return TEXT_H_PT - DBL_FLOAT_SEP_PT - caption_lines * CAPTION_LEADING_PT - 4.0
