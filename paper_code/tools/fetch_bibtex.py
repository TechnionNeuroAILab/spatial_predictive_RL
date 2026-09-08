#!/usr/bin/env python3
"""
Resolve the manuscript's missing citation keys to BibTeX via Crossref.

Two-step by design: resolve a curated search string -> DOI (printing the
returned title/year/journal so a human can confirm the match), then
content-negotiate BibTeX from that DOI. No DOI is ever hand-typed.

Output is normalised to paper/bibliography.bib's house style:
    title / author / journal / volume / number / pages / year / publisher
    "--" page ranges, no doi/url/issn/month, key = our key not Crossref's.

Entries that Crossref does not index well (arXiv preprints, conference papers)
are hand-specified in HANDWRITTEN below and pass through unchanged.

Usage:
    python paper_code/tools/fetch_bibtex.py --resolve-only     # review matches first
    python paper_code/tools/fetch_bibtex.py > /tmp/new_entries.bib
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request

MAILTO = "stefano.recanatesi@alleninstitute.org"  # Crossref polite pool

# Keys whose Crossref top hit is NOT the paper we want. Verified by hand from
# --resolve-only output: Crossref ranks "Faculty Opinions recommendation of X"
# stubs and errata above the real article for these queries.
PINNED_DOI: dict[str, str] = {
    "markus1995interactions": "10.1523/jneurosci.15-11-07079.1995",  # top hit is the 1996 erratum
    "lee2006gradual":         "10.1016/j.neuron.2006.06.033",        # top 2 are Faculty Opinions
    "banino2018vector":       "10.1038/s41586-018-0102-6",           # top hit is Faculty Opinions
    "george2022ratinabox":    "10.7554/elife.85274",                 # published eLife 2024, not the preprint
}

# key -> bibliographic search string. Curated; each should have exactly one
# canonical match. See --resolve-only to verify before trusting the output.
QUERIES: dict[str, str] = {
    "o1971hippocampus":        "The hippocampus as a spatial map preliminary evidence from unit activity in the freely-moving rat O'Keefe Dostrovsky",
    # NB: companion paper in the same issue (Muller, Kubie & Ranck, 7:1935-1950)
    # is a different study. Match on the title, not the year/journal.
    "muller1987effects":       "The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells Muller Kubie",
    "markus1995interactions":  "Interactions between location and task affect the spatial and directional firing of hippocampal neurons Markus",
    "leutgeb2005independent":  "Independent codes for spatial and episodic memory in hippocampal neuronal ensembles Leutgeb",
    "moser2008place":          "Place cells grid cells and the brain's spatial representation system Moser Kropff Moser",
    "hollup2001accumulation":  "Accumulation of hippocampal place fields at the goal location in an annular watermaze task Hollup",
    "dupret2010reorganization": "The reorganization and reactivation of hippocampal maps predict spatial memory performance Dupret",
    "zaremba2017impaired":     "Impaired hippocampal place cell dynamics in a mouse model of the 22q11.2 deletion Zaremba",
    "lee2006gradual":          "Gradual translocation of spatial correlates of neuronal firing in the hippocampus toward prospective reward locations Lee",
    # NB: bittner2017behavioral is already defined in bibliography.bib — not fetched.
    "foster2000model":       "A model of hippocampally dependent navigation using the temporal difference learning rule Foster Morris Dayan",
    "gustafson2011grid":       "Grid cells place cells and geodesic generalization for spatial reinforcement learning Gustafson Daw",
    "banino2018vector":        "Vector-based navigation using grid-like representations in artificial agents Banino",
    "foster2006reverse":       "Reverse replay of behavioural sequences in hippocampal place cells during the awake state Foster Wilson",
    "pfeiffer2013hippocampal": "Hippocampal place-cell sequences depict future paths to remembered goals Pfeiffer Foster",
    "olafsdottir2018role":     "The role of hippocampal replay in memory and planning Olafsdottir Bush Barry",
    "schapiro2013neural":      "Neural representations of events arise from temporal community structure Schapiro",
    "bethus2010dopamine":      "Dopamine and memory modulation of the persistence of memory for novel hippocampal NMDA receptor-dependent paired associates Bethus Tse Morris",
    "mcnamara2014dopaminergic": "Dopaminergic neurons promote hippocampal reactivation and spatial memory persistence McNamara",
    "ito2015prefrontal":       "A prefrontal-thalamo-hippocampal circuit for goal-directed spatial navigation Ito",
    "spellman2015hippocampal": "Hippocampal-prefrontal input supports spatial encoding in working memory Spellman",
    # Renamed from ormond2015place: the sentence cites the RODENT counterpart to
    # Sarel 2017's bat goal-vector code, which is Ormond & O'Keefe 2022 Nature.
    # (Ormond & McNaughton 2015 PNAS is about MEC inactivation and does not fit.)
    "ormond2022goal":          "Hippocampal place cells have goal-oriented vector fields during navigation Ormond O'Keefe",
    # Published version of the RatInABox preprint (key year kept for stability).
    "george2022ratinabox":     "RatInABox a toolkit for modelling locomotion and neuronal activity in continuous environments George Barry",
}

# Not reliably indexed by Crossref (arXiv / conference). House style applied by hand.
HANDWRITTEN: dict[str, str] = {
    "schulman2017proximal": """@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}""",
    "cueva2018emergence": """@inproceedings{cueva2018emergence,
  title={Emergence of grid-like representations by training recurrent neural networks to perform spatial localization},
  author={Cueva, Christopher J and Wei, Xue-Xin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}""",
}


def _get(url: str, accept: str | None = None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": f"bibfetch (mailto:{MAILTO})"})
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", errors="replace")


def resolve(query: str, rows: int = 3) -> list[dict]:
    url = ("https://api.crossref.org/works?query.bibliographic="
           + urllib.parse.quote(query) + f"&rows={rows}&mailto={MAILTO}")
    items = json.loads(_get(url))["message"]["items"]
    out = []
    for w in items:
        out.append({
            "doi": w.get("DOI", ""),
            "title": (w.get("title") or ["?"])[0],
            "year": (w.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "journal": (w.get("container-title") or ["?"])[0],
            "type": w.get("type", "?"),
        })
    return out


def bibtex_for(doi: str) -> str:
    return _get("https://doi.org/" + doi, accept="application/x-bibtex").strip()


KEEP = ["title", "author", "journal", "volume", "number", "pages", "year", "publisher"]


def _parse_fields(raw: str) -> dict[str, str]:
    """Extract field={value} pairs with brace matching.

    Crossref returns the whole entry on ONE line, so a line-oriented parser
    sees nothing. Values may themselves contain braces, hence explicit depth
    tracking rather than a regex.
    """
    fields: dict[str, str] = {}
    for m in re.finditer(r"([a-zA-Z]+)\s*=\s*", raw):
        name = m.group(1).lower()
        i = m.end()
        if i >= len(raw):
            break
        if raw[i] == "{":
            depth, j = 0, i
            while j < len(raw):
                if raw[j] == "{":
                    depth += 1
                elif raw[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            fields[name] = raw[i + 1:j]
        else:  # bare value, e.g. month=July
            j = i
            while j < len(raw) and raw[j] not in ",}":
                j += 1
            fields[name] = raw[i:j].strip()
    return fields


def normalise(raw: str, key: str) -> str:
    """Crossref BibTeX -> this file's house style."""
    entry_type = "article"
    m = re.match(r"\s*@(\w+)\s*\{", raw)
    if m:
        t = m.group(1).lower()
        entry_type = {"article": "article", "inproceedings": "inproceedings",
                      "book": "book", "inbook": "incollection"}.get(t, "article")

    parsed = _parse_fields(raw)
    fields = {k: v for k, v in parsed.items() if k in KEEP}

    if "pages" in fields:
        p = fields["pages"].replace("–", "--").replace("—", "--")
        p = p.replace("---", "--")
        if "--" not in p and "-" in p:
            p = p.replace("-", "--")
        fields["pages"] = p
    # normalise typographic apostrophes/quotes that break under pdflatex
    for k, v in list(fields.items()):
        v = v.replace("’", "'").replace("‘", "'")
        v = v.replace("“", "``").replace("”", "''")
        fields[k] = re.sub(r"\s+", " ", v).strip()

    lines = [f"@{entry_type}{{{key},"]
    kept = [n for n in KEEP if fields.get(n)]
    for i, name in enumerate(kept):
        comma = "," if i < len(kept) - 1 else ""
        lines.append(f"  {name}={{{fields[name]}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve-only", action="store_true",
                    help="print candidate matches for human review; emit no BibTeX")
    ap.add_argument("--only", nargs="*", help="restrict to these keys")
    args = ap.parse_args()

    keys = args.only if args.only else list(QUERIES)
    for key in keys:
        if key not in QUERIES:
            continue
        pinned = PINNED_DOI.get(key)
        try:
            cands = resolve(QUERIES[key])
        except Exception as e:
            print(f"% RESOLVE FAILED {key}: {e}", file=sys.stderr)
            continue
        if args.resolve_only:
            print(f"\n### {key}" + (f"   [PINNED {pinned}]" if pinned else ""))
            for i, c in enumerate(cands):
                mark = "->" if (pinned or "") == c["doi"] or (not pinned and i == 0) else "  "
                print(f" {mark} {c['doi']}  {c['year']}  {c['journal'][:38]:38s}  {c['title'][:64]}")
            continue
        doi = pinned or cands[0]["doi"]
        try:
            print(normalise(bibtex_for(doi), key))
            print()
        except Exception as e:
            print(f"% BIBTEX FAILED {key} ({doi}): {e}", file=sys.stderr)
        time.sleep(0.4)

    if not args.resolve_only:
        for key, entry in HANDWRITTEN.items():
            if args.only and key not in args.only:
                continue
            print(entry)
            print()


if __name__ == "__main__":
    main()
