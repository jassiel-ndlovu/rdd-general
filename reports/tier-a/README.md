# Tier A, generated report tables

Everything in this directory is **generated**. Do not edit it by hand; regenerate it.

```bash
PYTHONPATH=src python -m rdd.eval.aggregate \
    --runs ../Tier-A-Results/metrics \
    --out reports/tier-a/aggregate.json \
    --tex-dir reports/tier-a
```

| File | What it is |
|---|---|
| `aggregate.json` | every combined number: pooled, per-class and per-country mean ± sd per config, plus the six-versus-four comparison at AP50 and AP50-95 with effect sizes. Carries the two split SHA-256 hashes and the class count |
| `tierA_per_class.tex` | per-class table, six-class baseline. `\input` it; it needs `booktabs` and `float` |
| `tierA_per_country.tex` | per-country table, six-class baseline |
| `tierA_six_vs_four.tex` | the four-class control against the six-class baseline on the four shared classes, both IoU thresholds, with Δ/σ |
| `CROSS-CHECK.md` | US13 task 13.6: one aggregated figure recomputed by hand, with the arithmetic |

The same four generated files are written into the results bundle at
`Tier-A-Results/metrics/` (JSON) and `Tier-A-Results/metrics/tex/` (tables); the copies here
are the version-controlled ones, because the bundle is not in git. They are byte-identical,
the aggregator is deterministic over the same run summaries.

Two conventions the tables and the JSON both state, and the narrative must not contradict:

- Spreads are **sample** standard deviations (n − 1) across seeds. With three seeds this is not
  cosmetic; see `CROSS-CHECK.md`.
- In the six-versus-four comparison, **Δ is control minus baseline**, and σ is the *baseline's*
  seed-to-seed spread, because the control has a single seed.

Table `tierA_per_class.tex` is the one that prints both split hashes in full; the other two
abbreviate and refer to it, so include it if you include them.
