"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only]

writes kaggle/dataset/ (the grid results and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package
source and reruns the grid).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/public-lb-selection-bias-grid"
NOTEBOOK_ID = "locdunhat/how-much-does-the-best-public-score-overstate"
TITLE = "How Much Does the Best Public Score Overstate"


def build_dataset(public: bool) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["grid.csv"]:
        (out / name).write_bytes((ROOT / "results" / name).read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Public LB Selection Bias Grid",
        "id": DATASET_ID,
        "subtitle": "How far the best public score of N correlated submissions sits above its true score",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["benchmark", "simulations"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in sorted((ROOT / "selbias").glob("*.py"))}


SETUP = """

import os, sys
ROOT = "/kaggle/working/lb-selection-bias" if os.path.isdir("/kaggle/working") else os.path.abspath("lb-selection-bias")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
"""

RUN = """import time
import pandas as pd
from selbias import run as bench
pd.set_option("display.width", 250); pd.set_option("display.precision", 4)
t = time.time()
g = bench.grid()
print(len(g), "cells in", round(time.time() - t, 1), "s")
null = g[g.spread == 0]
"""

NULL = """null[null.n_public == 1200].pivot_table(index="n_subs", columns="q", values="optimism_vs_true").round(4)
"""

RULE = """null[null.n_public == 1200].pivot_table(index="n_subs", columns="q", values="rough_rule").round(4)
"""

ROWS = """null[null.q == 0.025].pivot_table(index="n_subs", columns="n_public", values="optimism_vs_true").round(4)
"""

SPREAD = """s = g[(g.n_public == 1200) & (g.q == 0.025)]
s.pivot_table(index="n_subs", columns="spread", values=["optimism_vs_true", "picked_true_best", "true_regret"]).round(4)
"""

PRIVATE = """null[null.n_public == 1200].pivot_table(index="n_subs", columns="q", values=["optimism_vs_private", "sd_optimism_vs_private"]).round(4)
"""

COMPARE = """import glob
hits = glob.glob("/kaggle/input/**/grid.csv", recursive=True)
if hits:
    saved = pd.read_csv(hits[0])
    keys = ["n_public", "n_subs", "q", "spread"]
    m = g.merge(saved, on=keys, suffixes=("", "_saved"))
    print("rows matched:", len(m), "of", len(g), "| largest difference in optimism_vs_true:",
          float((m.optimism_vs_true - m.optimism_vs_true_saved).abs().max()))
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("markdown", "## 1. The grid\n\n600 / 1,200 / 3,000 public rows (private rows: 4 times as many), 1-50 submissions, "
                     "flip probability q = 0.01-0.1, true spread 0 / 0.002 / 0.005; 4,000 trials per cell."),
        ("code", RUN),
        ("markdown", "## 2. Optimism of the best public score, all submissions equally good (1,200 public rows)"),
        ("code", NULL),
        ("markdown", "## 3. The rough rule: sqrt(q(1 - q) / n_public) x E[max of N normals]"),
        ("code", RULE),
        ("markdown", "## 4. More public rows (q = 0.025, pairwise flip 5%)"),
        ("code", ROWS),
        ("markdown", "## 5. When some submissions really are better (1,200 rows, q = 0.025)"),
        ("code", SPREAD),
        ("markdown", "## 6. What the private score shows"),
        ("code", PRIVATE),
        ("markdown", "## 7. Against the saved grid"),
        ("code", COMPARE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "lb-selection-bias.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "lb-selection-bias.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public))
    print("wrote", build_notebook(public))
