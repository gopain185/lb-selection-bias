"""The grid: selection bias of the best public score over public size, number of submissions, pairwise flip rate and
true spread between submissions.

    python -m selbias.run            # writes results/grid.csv
    python -m selbias.run --quick    # a small grid, for a smoke check
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from selbias.core import Cell, rough_optimism, simulate

ROOT = Path(__file__).resolve().parent.parent
N_PUBLIC = [600, 1200, 3000]
N_SUBS = [1, 2, 5, 10, 20, 50]
Q = [0.01, 0.025, 0.05, 0.1]
SPREAD = [0.0, 0.002, 0.005]
ACC = 0.62
N_PRIVATE_PER_PUBLIC = 4
TRIALS = 4000


def grid(quick: bool = False, trials: int = TRIALS) -> pd.DataFrame:
    n_public, n_subs, qs, spreads = (([1200], [1, 10], [0.05], [0.0, 0.005]) if quick else (N_PUBLIC, N_SUBS, Q, SPREAD))
    rows = []
    for i, (n, s, q, sp) in enumerate(itertools.product(n_public, n_subs, qs, spreads)):
        cell = Cell(n_public=n, n_private=N_PRIVATE_PER_PUBLIC * n, n_subs=s, acc=ACC, q=q, spread=sp)
        out = simulate(cell, trials=trials, seed=1000 + i)
        rows.append({"n_public": n, "n_private": cell.n_private, "n_subs": s, "acc": ACC, "q": q, "spread": sp, **out,
                     "rough_rule": rough_optimism(n, s, q)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    df = grid(a.quick)
    out = ROOT / "results" / ("grid_quick.csv" if a.quick else "grid.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")
    print("wrote", out, len(df), "cells")
