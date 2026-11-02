# How Much Does the Best Public Score Overstate?

After a few dozen submissions, your best public score becomes the score to beat. It was chosen by the public rows,
though, so part of it is luck. This notebook simulates how large that part is when the submissions are near-copies of
each other: N submissions, each one flipping each row's correctness with probability q against a shared reference, and
the best public score selected.

- Everything is synthetic, and the counts are drawn exactly from binomials. The notebook embeds the package source
  (`lb-selection-bias`, MIT) and reruns the full grid of 216 cells in a few seconds on a CPU.
- The saved grid is the attached dataset (CC0). The last code cell checks that this run matches it. A different NumPy version draws different binomials, so
  differences up to about 0.0002 are Monte Carlo noise, not a change in the method.
- The pairwise flip rate, 2q(1 - q), is the share of rows on which two submissions disagree about correctness. You can
  measure it on your own out-of-fold predictions.
