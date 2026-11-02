import numpy as np

from selbias.core import Cell, expected_accuracy, pairwise_flip, rough_optimism, scores, simulate
from selbias.run import grid


def test_mean_accuracy_is_the_expected_accuracy():
    rng = np.random.default_rng(0)
    s = scores(rng, 1200, 5, 0.62, 0.05, 20_000)
    assert abs(s.mean() - expected_accuracy(0.62, 0.05)) < 5e-4


def test_paired_difference_matches_the_flip_rate():
    rng = np.random.default_rng(1)
    q, n = 0.05, 1200
    s = scores(rng, n, 2, 0.62, q, 40_000)
    sd = (s[:, 0] - s[:, 1]).std()
    assert abs(sd - np.sqrt(pairwise_flip(q) / n)) / np.sqrt(pairwise_flip(q) / n) < 0.03


def test_one_submission_has_no_selection_bias():
    out = simulate(Cell(1200, 4800, 1, 0.62, 0.05, 0.0), trials=20_000, seed=2)
    assert abs(out["optimism_vs_true"]) < 2e-4
    assert abs(out["optimism_vs_private"]) < 3e-4


def test_identical_submissions_have_no_selection_bias():
    out = simulate(Cell(1200, 4800, 20, 0.62, 0.0, 0.0), trials=5000, seed=3)
    assert abs(out["optimism_vs_true"]) < 5e-4
    assert out["public_range"] == 0


def test_bias_grows_with_the_number_of_submissions_and_the_flip_rate():
    a = simulate(Cell(1200, 4800, 2, 0.62, 0.05, 0.0), trials=8000, seed=4)["optimism_vs_true"]
    b = simulate(Cell(1200, 4800, 20, 0.62, 0.05, 0.0), trials=8000, seed=4)["optimism_vs_true"]
    c = simulate(Cell(1200, 4800, 20, 0.62, 0.1, 0.0), trials=8000, seed=4)["optimism_vs_true"]
    assert 0 < a < b < c


def test_rough_rule_matches_the_null_simulation():
    for s, q in [(5, 0.05), (20, 0.025)]:
        sim = simulate(Cell(1200, 4800, s, 0.62, q, 0.0), trials=8000, seed=5)["optimism_vs_true"]
        rule = rough_optimism(1200, s, q)
        assert abs(sim - rule) / rule < 0.1


def test_a_large_true_spread_is_usually_found():
    out = simulate(Cell(1200, 4800, 10, 0.62, 0.01, 0.02), trials=4000, seed=6)
    assert out["picked_true_best"] > 0.8


def test_quick_grid_runs():
    df = grid(quick=True, trials=500)
    assert len(df) == 4
    assert df["optimism_vs_true"].notna().all()
