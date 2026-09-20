"""
Aggregate Tier A runs into the tables the report needs.

Produces three things the proposal's evaluation protocol demands:

  1. **mean +/- std across seeds** -- single-seed numbers are not evidence.
  2. **per-class** breakdown -- never a single pooled number.
  3. **per-country** breakdown -- H3 is specifically about domain variance, and
     the published challenge leaderboards show >0.25 F1 spread between countries
     inside one model.

It also computes the **H2 comparison** directly: the 6-class baseline against
the 4-class control, restricted to the four classes they share. Because both
were trained on identical images and identical splits, any difference is
attributable to broadening the taxonomy.

Usage
-----
    python -m rdd.eval.aggregate --runs /opt/rdd/runs --out summary.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

SHARED = ["C1_longitudinal_crack", "C2_transverse_crack",
          "C3_alligator_crack", "C4_pothole"]


def mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt(m: float, s: float, places: int = 3) -> str:
    if math.isnan(m):
        return "     --   "
    return f"{m:.{places}f} +/- {s:.{places}f}"


def load_runs(runs_dir: Path) -> list[dict]:
    out = []
    for p in sorted(runs_dir.glob("*/summary.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:                        # noqa: BLE001
            print(f"  [warn] could not read {p}: {e}")
    return out


def group_key(run: dict) -> str:
    """Strip the trailing _s<seed> so seeds of one config group together."""
    name = run.get("run", "")
    return name.rsplit("_s", 1)[0] if "_s" in name else name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate Tier A runs.")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    runs = load_runs(args.runs)
    if not runs:
        print(f"no summary.json found under {args.runs}")
        return 1
    print(f"found {len(runs)} run(s)\n")

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)

    report: dict = {"configs": {}, "h2_comparison": None}

    # ---- pooled ------------------------------------------------------------
    print("=" * 78)
    print("POOLED  (mean +/- std across seeds)")
    print("=" * 78)
    print(f"{'config':<22}{'seeds':>6}  {'mAP50':^18}{'mAP50-95':^18}")
    for cfg in sorted(groups):
        rs = groups[cfg]
        m50, s50 = mean_std([r["pooled"]["mAP50"] for r in rs])
        m95, s95 = mean_std([r["pooled"]["mAP50_95"] for r in rs])
        print(f"{cfg:<22}{len(rs):>6}  {fmt(m50, s50):^18}{fmt(m95, s95):^18}")
        report["configs"][cfg] = {
            "n_seeds": len(rs),
            "seeds": [r.get("seed") for r in rs],
            "pooled": {"mAP50_mean": m50, "mAP50_std": s50,
                       "mAP50_95_mean": m95, "mAP50_95_std": s95},
        }

    # ---- per class ---------------------------------------------------------
    for cfg in sorted(groups):
        rs = groups[cfg]
        names = sorted({n for r in rs for n in r.get("per_class", {})})
        if not names:
            continue
        print(f"\nPER CLASS -- {cfg}")
        print(f"  {'class':<26}{'AP50':^18}{'precision':^18}{'recall':^18}")
        per_class = {}
        for n in names:
            ap50 = [r["per_class"][n]["AP50"] for r in rs if n in r.get("per_class", {})]
            pre = [r["per_class"][n]["precision"] for r in rs if n in r.get("per_class", {})]
            rec = [r["per_class"][n]["recall"] for r in rs if n in r.get("per_class", {})]
            a, sa = mean_std(ap50); p, sp = mean_std(pre); rc, sr = mean_std(rec)
            print(f"  {n:<26}{fmt(a, sa):^18}{fmt(p, sp):^18}{fmt(rc, sr):^18}")
            per_class[n] = {"AP50_mean": a, "AP50_std": sa,
                            "precision_mean": p, "recall_mean": rc}
        report["configs"][cfg]["per_class"] = per_class

    # ---- per country -------------------------------------------------------
    for cfg in sorted(groups):
        rs = groups[cfg]
        countries = sorted({c for r in rs for c in r.get("per_country", {})})
        if not countries:
            continue
        print(f"\nPER COUNTRY -- {cfg}")
        print(f"  {'country':<20}{'mAP50':^18}{'F1':^18}{'n_val':>8}")
        per_country = {}
        spread = []
        for c in countries:
            vals = [r["per_country"][c] for r in rs
                    if c in r.get("per_country", {}) and "mAP50" in r["per_country"][c]]
            if not vals:
                continue
            m, s = mean_std([v["mAP50"] for v in vals])
            f, sf = mean_std([v["f1"] for v in vals])
            n = vals[0].get("n_images", 0)
            print(f"  {c:<20}{fmt(m, s):^18}{fmt(f, sf):^18}{n:>8}")
            per_country[c] = {"mAP50_mean": m, "mAP50_std": s,
                              "f1_mean": f, "n_val_images": n}
            spread.append(m)
        if len(spread) > 1:
            rng = max(spread) - min(spread)
            print(f"  {'SPREAD (max-min)':<20}{rng:^18.3f}"
                  f"   <- H3: domain variance")
            report["configs"][cfg]["per_country_spread_mAP50"] = rng
        report["configs"][cfg]["per_country"] = per_country

    # ---- H2: broadening cost on the shared classes -------------------------
    base = next((c for c in groups if "base6" in c), None)
    ctrl = next((c for c in groups if "ctrl4" in c), None)
    if base and ctrl:
        print("\n" + "=" * 78)
        print("H2 -- cost of broadening the taxonomy, on the FOUR SHARED classes")
        print("     (identical images, identical splits; only the label set differs)")
        print("=" * 78)
        print(f"  {'class':<26}{'6-class':>12}{'4-class':>12}{'delta':>12}")
        deltas = {}
        for n in SHARED:
            b = report["configs"][base].get("per_class", {}).get(n, {}).get("AP50_mean")
            c = report["configs"][ctrl].get("per_class", {}).get(n, {}).get("AP50_mean")
            if b is None or c is None:
                continue
            d = b - c
            deltas[n] = d
            arrow = "worse" if d < -0.005 else ("better" if d > 0.005 else "~same")
            print(f"  {n:<26}{b:>12.3f}{c:>12.3f}{d:>+12.3f}   {arrow}")
        if deltas:
            md = statistics.mean(deltas.values())
            print(f"  {'MEAN DELTA':<26}{'':>12}{'':>12}{md:>+12.3f}")
            print()
            if md < -0.02:
                print("  Interpretation: broadening measurably HURTS the shared classes.")
                print("  H2 holds in the 'bounded cost' sense -- quantify and report it.")
            elif md > 0.02:
                print("  Interpretation: broadening HELPS the shared classes, likely")
                print("  because the extra labels remove background confusion.")
            else:
                print("  Interpretation: broadening is roughly FREE on the shared")
                print("  classes -- a clean positive result for the project's premise.")
            report["h2_comparison"] = {"baseline": base, "control": ctrl,
                                       "per_class_delta_AP50": deltas,
                                       "mean_delta_AP50": md}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
