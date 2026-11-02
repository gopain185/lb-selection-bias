"""Selection bias of the best public-leaderboard score among many correlated submissions.

Model. A test set has a public part of `n_public` rows and a private part of `n_private` rows. There is one reference
correctness vector: each row is right with probability `acc`. Submission i agrees with the reference except on rows it
flips, each row flipped independently with probability `q`; so two submissions disagree on about 2q(1 - q) of the rows
(the "pairwise flip rate"). Flipping is symmetric, which leaves every submission's expected accuracy at

    acc (1 - q) + (1 - acc) q

so under the null all submissions are equally good. Optionally each submission also gets a true offset drawn from
N(0, spread), added to both its public and its private score: then some submissions really are better.

Given the reference vector with k correct rows out of n, the number of right rows a submission turns wrong is
Binomial(k, q) and the number of wrong rows it turns right is Binomial(n - k, q), independently across submissions.
That gives exact accuracies without drawing row-level masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def expected_accuracy(acc: float, q: float) -> float:
    return acc * (1 - q) + (1 - acc) * q


def pairwise_flip(q: float) -> float:
    """Expected share of rows on which two submissions disagree about correctness."""
    return 2 * q * (1 - q)


def scores(rng: np.random.Generator, n_rows: int, n_subs: int, acc: float, q: float, trials: int) -> np.ndarray:
    """(trials, n_subs) accuracies of n_subs submissions on n_rows rows, all with the same expected accuracy."""
    k = rng.binomial(n_rows, acc, size=(trials, 1))
    lost = rng.binomial(np.broadcast_to(k, (trials, n_subs)), q)
    gained = rng.binomial(np.broadcast_to(n_rows - k, (trials, n_subs)), q)
    return (k - lost + gained) / n_rows


@dataclass(frozen=True)
class Cell:
    n_public: int
    n_private: int
    n_subs: int
    acc: float
    q: float
    spread: float


def simulate(cell: Cell, trials: int, seed: int) -> dict:
    """Pick the submission with the best public score and compare its public score with its private and true scores."""
    rng = np.random.default_rng(seed)
    base = expected_accuracy(cell.acc, cell.q)
    offset = rng.normal(0.0, cell.spread, size=(trials, cell.n_subs)) if cell.spread > 0 else np.zeros((trials, cell.n_subs))
    public = scores(rng, cell.n_public, cell.n_subs, cell.acc, cell.q, trials) + offset
    private = scores(rng, cell.n_private, cell.n_subs, cell.acc, cell.q, trials) + offset
    true = base + offset
    rows = np.arange(trials)
    pick = public.argmax(1)
    best_true = true.argmax(1)
    return {
        "optimism_vs_true": float((public[rows, pick] - true[rows, pick]).mean()),
        "optimism_vs_private": float((public[rows, pick] - private[rows, pick]).mean()),
        "sd_optimism_vs_private": float((public[rows, pick] - private[rows, pick]).std()),
        "true_regret": float((true[rows, best_true] - true[rows, pick]).mean()),
        "picked_true_best": float((pick == best_true).mean()) if cell.spread > 0 else float("nan"),
        "public_range": float((public.max(1) - public.min(1)).mean()),
        "pairwise_flip": pairwise_flip(cell.q),
    }


def max_normal_mean(n: int, draws: int = 200_000, seed: int = 0) -> float:
    """E[max of n independent standard normals], by simulation (used by the tests and as a rough rule)."""
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal((draws, n)).max(1).mean())


def rough_optimism(n_public: int, n_subs: int, q: float) -> float:
    """Rough rule: each submission's public score is the shared score plus its own noise of sd sqrt(q (1 - q) / n);
    the best of n_subs exchangeable submissions is that much above its true score times E[max of n normals]."""
    return float(np.sqrt(q * (1 - q) / n_public) * max_normal_mean(n_subs))
