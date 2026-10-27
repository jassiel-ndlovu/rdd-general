"""Aggregate the three Tier A seed revalidations into spreads.

aggregate.json carries a standard deviation for pooled mAP50 and for per-class
AP50, both from the training summaries. It carries none for
per-class-within-country, because that breakdown only exists in the
revalidation pass and only seed 0 had been through it.

US14a and US17 both quote figures at that granularity, so the spreads are
needed to say whether a difference exceeds seed noise. This reconciles all
three seeds into means and standard deviations at every granularity the
revalidation produces.

Three seeds give an unstable estimate of a standard deviation, and where the
spread is small the ratio to it explodes: India's per-country spread is
0.00095, which turns a 0.023 difference into -24 sigma. The output repeats this
caution. Read the absolute difference first.

Usage:
    python -m rdd.eval.aggregate_seeds \
        --reval Tier-A-Results/metrics/tierA_base6_s0/revalidation.json \
        --reval Tier-A-Results/metrics/tierA_base6_s1/revalidation.json \
        --reval Tier-A-Results/metrics/tierA_base6_s2/revalidation.json \
        --out Tier-A-Results/metrics/aggregate_seeds.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def mean_std(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    return {
        "mean": round(statistics.fmean(xs), 4),
        # Population standard deviation over the seeds actually run, matching
        # the convention aggregate.json already uses for this project.
        "std": round(statistics.pstdev(xs), 5) if len(xs) > 1 else 0.0,
        "n": len(xs),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
    }


def collect(reports: list[dict], path: list[str], metric: str) -> list[float]:
    out = []
    for r in reports:
        node = r
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict) and metric in node:
            out.append(node[metric])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reval", action="append", required=True, type=Path,
                    help="repeatable; one revalidation.json per seed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reval]
    if len(reports) < 2:
        print("at least two seeds are needed for a spread")
        return 1

    # Every seed must describe the same split, or the spread is meaningless.
    headers = {json.dumps(r.get("data_yaml_header"), sort_keys=True)
               for r in reports}
    if len(headers) > 1:
        print("the seeds do not share a data.yaml header, so they were not "
              "evaluated on the same split. Refusing to pool them.")
        return 2

    out: dict = {
        "seeds": len(reports),
        "sources": [str(p) for p in args.reval],
        "caution": "Three seeds give an unstable standard deviation. Where the "
                   "spread is small a ratio to it is easily over-read; use the "
                   "absolute difference as the primary quantity.",
        "pooled": {},
        "per_class": {},
        "per_country": {},
        "per_class_by_country": {},
    }

    for metric in ("mAP50", "mAP50_95"):
        v = mean_std(collect(reports, ["pooled"], metric))
        if v:
            out["pooled"][metric] = v

    classes = sorted({c for r in reports for c in r.get("per_class", {})})
    for c in classes:
        out["per_class"][c] = {
            m: mean_std(collect(reports, ["per_class", c], m))
            for m in ("AP50", "AP50_95", "precision", "recall")
        }

    countries = sorted({k for r in reports for k in r.get("per_country", {})})
    for ct in countries:
        out["per_country"][ct] = {
            m: mean_std(collect(reports, ["per_country", ct], m))
            for m in ("mAP50", "mAP50_95", "precision", "recall")
        }

    # The breakdown this module exists for.
    for ct in sorted({k for r in reports
                      for k in r.get("per_class_by_country", {})}):
        block = {}
        for c in sorted({cc for r in reports
                         for cc in r.get("per_class_by_country", {})
                                     .get(ct, {})}):
            block[c] = {
                m: mean_std(collect(reports, ["per_class_by_country", ct, c], m))
                for m in ("AP50", "AP50_95", "precision", "recall")
            }
        out["per_class_by_country"][ct] = block

    print(f"seeds aggregated: {len(reports)}")
    print()
    print("per-class AP50 across seeds")
    for c in classes:
        v = out["per_class"][c]["AP50"]
        print(f"  {c:24} {v['mean']:.4f} +/- {v['std']:.5f}  "
              f"[{v['min']:.4f}, {v['max']:.4f}]")
    print()
    print("per-class-by-country AP50 spread, the figures US14a needed")
    for ct, block in out["per_class_by_country"].items():
        print(f"  {ct}")
        for c, metrics in block.items():
            v = metrics.get("AP50") or {}
            if v:
                print(f"    {c:24} {v['mean']:.4f} +/- {v['std']:.5f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
