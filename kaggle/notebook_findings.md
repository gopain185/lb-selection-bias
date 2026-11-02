## Findings

1. **The bias is the same size as late-competition gains.** With 1,200 public rows, 20 equally good submissions that
   disagree on 5% of rows put the best one about 0.008 above its true score, and 50 put it about 0.010 above. A real
   +0.005 idea will usually score below that best.
2. **It depends on how different the submissions are.** Identical submissions give no bias. The rough rule
   `sqrt(q(1 - q) / n_public) x E[max of N normals]` is within 10% of the simulation.
3. **More public rows shrink it only with the square root of their number:** 0.0117 / 0.0083 / 0.0054 for 600 / 1,200 /
   3,000 rows (20 submissions, 5% flip).
4. **Real differences between submissions do not remove it.** With a true spread of 0.002 and 5% flip, the best of 20
   is still 0.0074 above its true score, and it is the truly best one only 17% of the time.
5. **The private score repays it on average.** The selected submission drops by the same amount on private rows, but
   any single drop has an sd of 0.009 to 0.022.

Practical reading: before comparing a new submission with your best public score, subtract the rough-rule optimism
for the number of similar submissions behind that best. A new score below the best by less than that amount is not
evidence that the new idea is worse.

Limits: flips are independent here, while real near-copies share their flips, so count submissions conservatively.
The metric is plain accuracy.
