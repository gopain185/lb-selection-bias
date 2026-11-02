# lb-selection-bias

How far above its true score is the best public-leaderboard score among many similar submissions?

After a few dozen submissions, the best public score becomes the reference, and every new idea is judged against it.
The best score was chosen by the public rows, though, so part of it is luck on those rows. This repo simulates how
large that part is when the submissions are near-copies of one another, which is the usual case late in a
competition.

Related: my earlier `lb-shakeup-bench` measures what it costs to choose the final submission by public AUC, at
Playground sizes. This repo asks a narrower question: how far the best public score sits above its true score, as a
function of a flip rate you can measure on your own OOF predictions.

## Model

- A test set has `n_public` public rows and `4 x n_public` private rows. A reference model is right on each row with
  probability 0.62.
- Each submission agrees with the reference except on the rows it flips, each one with probability `q`. Two
  submissions therefore disagree about correctness on about `2q(1 - q)` of the rows, the **pairwise flip rate**. You
  can measure that rate on out-of-fold predictions.
- Under the null all submissions are equally good. With `spread > 0`, each one also has a true offset drawn from
  N(0, spread), added to both its public and its private score.
- The submission with the best public score is selected. **Optimism** is its public score minus its true score.

The counts are drawn exactly from binomials (see `selbias/core.py`). A grid of 216 cells with 4,000 trials each runs
in about 10 seconds on a CPU.

A rough rule matches the null simulation to within 10% (this is tested):

    optimism ~= sqrt(q (1 - q) / n_public) x E[max of N standard normals]

where E[max] is 0.56, 1.16, 1.54, 1.87 and 2.25 for N = 2, 5, 10, 20 and 50.

## Results (1,200 public rows, all submissions equally good)

| submissions | q = 0.01 (flip 2%) | q = 0.025 (flip 5%) | q = 0.05 (flip 10%) | q = 0.1 (flip 18%) |
| --- | --- | --- | --- | --- |
| 2 | 0.0018 | 0.0025 | 0.0034 | 0.0045 |
| 5 | 0.0034 | 0.0050 | 0.0071 | 0.0100 |
| 10 | 0.0038 | 0.0068 | 0.0096 | 0.0131 |
| 20 | 0.0052 | 0.0083 | 0.0115 | 0.0161 |
| 50 | 0.0064 | 0.0102 | 0.0141 | 0.0194 |

1. **The bias is the same size as the gains people chase late in a competition.** Twenty submissions that disagree on
   5% of rows put the best one 0.008 above its true score. A new idea worth +0.005 then usually scores below that
   best.
2. **Its size depends on how much the submissions differ, not on how good they are.** Twenty identical submissions
   give zero bias, because the public rows cannot tell them apart.
3. **Bigger public sets help, but slowly.** At 5% flip and 20 submissions the bias is 0.0117 / 0.0083 / 0.0054 for
   600 / 1,200 / 3,000 public rows, so it shrinks with the square root of the row count.
4. **Real differences reduce the bias, but not by much.** With a true spread of 0.002 and 5% flip, 20 submissions give
   0.0074 rather than 0.0083. The selected one is the truly best only 17% of the time, and it is 0.0022 worse than the
   truly best on average.
5. **The private score shows the same bias.** On private rows the selected submission loses about as much as against
   its true score, with an sd of 0.009 to 0.022 per run. So any single drop is noisy, but on average the drop is the
   bias.

## A worked case

I had 49 scored submissions in one competition. 31 of them scored at least 0.61, and the best was 0.626. The
submissions near the top share most of their members, and they disagree on 1% to 5% of OOF rows (q about 0.005 to
0.025). Treat about 20 of them as roughly equally good: the table then puts 0.626 about 0.004 to 0.008 above that
blend's true score. That would explain a puzzle from the same week. Eight variants of the 0.626 blend, each within 0.006 of it at every
OOF cut, all scored 0.614 to 0.624 on the public LB. That range is about where an equally good blend should score.

This is an illustration, not a measurement. My submissions are not exchangeable, the public row count is a guess,
and the metric is balanced accuracy rather than accuracy.

## Limits

- Flips are independent across rows and submissions. Real submissions share their disagreements (two blends that
  differ only in one member's weight flip the same rows). For a measured pairwise flip rate, that changes little for
  pairs, but the effective number of distinct submissions is smaller than the count.
- Plain accuracy only. Balanced accuracy weights rare-class rows more, so its noise is somewhat larger for the same
  rows.
- The reference accuracy is fixed at 0.62. Noise scales with q(1 - q), not with accuracy, so this hardly matters.

## Use

    python -m pytest tests           # 8 tests
    python -m selbias.run            # results/grid.csv
    python kaggle/build_kaggle.py    # Kaggle dataset and notebook folders

Licence: MIT for the code, CC0 for `results/grid.csv`.
