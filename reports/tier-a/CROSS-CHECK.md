# US13 task 13.6, hand cross-check of one aggregated figure

One figure from `aggregate.json`, recomputed by hand from the raw run outputs, with the
arithmetic written out. Checked 19 October 2026 against the regenerated
`reports/tier-a/aggregate.json` (byte-identical to `Tier-A-Results/metrics/aggregate.json`).

## The figure

`configs.tierA_base6.per_class.C4_pothole.AP50_95_mean` and `..._std`, the six-class
baseline's pothole AP50-95, mean and standard deviation over the three seeds.

This one was chosen deliberately: its standard deviation is the denominator of the headline
six-versus-four effect size (−14.8 sd, `h2_comparison.metrics.AP50_95.per_class.C4_pothole`),
so an error in it would propagate straight into the central claim of the cost-of-generality
section.

## Raw inputs

Three files, each the final metrics of one training run, field
`per_class.C4_pothole.AP50_95`:

| File | Seed | AP50-95 |
|---|---|---|
| `Tier-A-Results/metrics/tierA_base6_s0/summary.json` | 0 | 0.2364 |
| `Tier-A-Results/metrics/tierA_base6_s1/summary.json` | 1 | 0.2377 |
| `Tier-A-Results/metrics/tierA_base6_s2/summary.json` | 2 | 0.2374 |

The control value used in the comparison is
`Tier-A-Results/metrics/tierA_ctrl4_s0/summary.json`, same field: **0.2271**.

## The arithmetic

**Mean.**

```
0.2364 + 0.2377 + 0.2374 = 0.7115
0.7115 / 3              = 0.2371666666...   ->  0.2372 at 4 d.p.
```

`aggregate.json` carries `0.23716666666666666`. Agrees.

**Standard deviation.** Deviations from the mean, and their squares:

```
0.2364 - 0.23716667 = -0.00076667    ->  5.8777778e-07
0.2377 - 0.23716667 = +0.00053333    ->  2.8444444e-07
0.2374 - 0.23716667 = +0.00023333    ->  5.4444444e-08
                           sum of squares = 9.2666667e-07
```

Divided by `n - 1 = 2`, then square-rooted:

```
9.2666667e-07 / 2 = 4.6333333e-07
sqrt(4.6333333e-07) = 0.00068069      ->  0.0007 at 4 d.p.
```

`aggregate.json` carries `0.0006806859285554045`. Agrees.

**This is the sample standard deviation (n − 1), and the choice matters.** With three seeds
the two conventions are visibly different figures:

```
n - 1 (sample, used):  sqrt(9.2666667e-07 / 2) = 0.00068069
n     (population):    sqrt(9.2666667e-07 / 3) = 0.00055578
```

The effect size for this class is the control minus the baseline, divided by that spread:

```
delta = 0.2271 - 0.23716667 = -0.01006667

-0.01006667 / 0.00068069 = -14.79   -> -14.8 sd   (sample, reported)
-0.01006667 / 0.00055578 = -18.11   -> -18.1 sd   (population)
```

So the headline figure would be quoted as −14.8 sd or −18.1 sd depending purely on the
convention. `rdd.eval.aggregate` uses `statistics.stdev`, which is the sample form, everywhere;
`STD_CONVENTION` in that module and every generated table footnote say so.

## A second, weaker check, and a discrepancy worth recording

The per-epoch histories (`results.csv`) carry pooled metrics only, not per-class, so they cannot
check the figure above. They can check the pooled ones, and they do not agree exactly:

| Run | Best epoch in `results.csv` | `results.csv` mAP50-95 | `summary.json` pooled mAP50-95 |
|---|---|---|---|
| `tierA_base6_s0` | 80 | 0.37152 | 0.3712 |
| `tierA_base6_s1` | 81 | 0.37294 | 0.3725 |
| `tierA_base6_s2` | 77 | 0.37307 | 0.3724 |

The gap is 3–5 units in the fourth decimal place, in the same direction in all three runs. That
is consistent with `summary.json` coming from a final validation pass over the best checkpoint
rather than from the training loop's own epoch-end evaluation, which is what
`Tier-A-Results/README.md` describes ("Every reported figure comes from the best checkpoint").
It is not a disagreement about which epoch was best, and it is well inside the seed-to-seed
spread.

Recorded here rather than dropped, with two consequences: `summary.json` is the only valid
source for reported figures, and `results.csv` must not be used to cross-check them beyond this
order of magnitude.

## Reproducing

```bash
python -m rdd.eval.aggregate --runs Tier-A-Results/metrics \
    --out Tier-A-Results/metrics/aggregate.json \
    --tex-dir Tier-A-Results/metrics/tex
```

Run from the project root with `PYTHONPATH=rdd/src`. The copies in this directory were produced
by the same command with `--out reports/tier-a/aggregate.json --tex-dir reports/tier-a`, and are
byte-identical to the ones in the results bundle.
