"""Compare US17 architecture variants against the Tier A baseline.

Produces the table US17 asks for: each modification evaluated independently,
against the baseline under identical settings, with parameters, FLOPs and
latency beside the accuracy so that a gain can be read against what it cost.

Every difference is also expressed in units of the baseline's seed-to-seed
standard deviation. This project's measured noise floor is 0.0004 pooled
mAP50, and several Tier A per-class spreads are under 0.002, so a raw delta of
0.005 can be either decisive or meaningless depending on the class. Quoting
the raw number alone invites the wrong reading.

Usage:
    python -m rdd.eval.compare_variants \
        --baseline Tier-A-Results/metrics/tierA_base6_s0/revalidation.json \
        --aggregate Tier-A-Results/metrics/aggregate.json \
        --variant us17a=<path>/revalidation.json \
        --variant us17b=<path>/revalidation.json \
        --out-tex reports/us17/us17_variants.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = [
    "C1_longitudinal_crack", "C2_transverse_crack", "C3_alligator_crack",
    "C4_pothole", "C5_white_line_blur", "C6_crosswalk_blur",
]
SHORT = {
    "C1_longitudinal_crack": "C1 longitudinal",
    "C2_transverse_crack": "C2 transverse",
    "C3_alligator_crack": "C3 alligator",
    "C4_pothole": "C4 pothole",
    "C5_white_line_blur": "C5 white-line blur",
    "C6_crosswalk_blur": "C6 crosswalk blur",
}
# The two classes US17 exists to improve.
TARGET = ("C1_longitudinal_crack", "C2_transverse_crack")


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt_sigma(d: float, sd: float) -> str:
    if not sd:
        return "   n/a"
    return f"{d / sd:+6.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True, type=Path,
                    help="the Tier A revalidation.json")
    ap.add_argument("--aggregate", required=True, type=Path,
                    help="Tier A aggregate.json, for the seed spreads")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable; a variant's revalidation.json")
    ap.add_argument("--summary", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable; a variant's summary.json, for params and "
                         "GFLOPs recorded at training time")
    ap.add_argument("--out-tex", type=Path, default=None)
    args = ap.parse_args(argv)

    base = load(args.baseline)
    agg = load(args.aggregate)["configs"]["tierA_base6"]
    sd_class = {c: v["AP50_std"] for c, v in agg["per_class"].items()}
    sd_country = {c: v["mAP50_std"] for c, v in agg["per_country"].items()}

    variants: dict[str, dict] = {}
    for spec in args.variant:
        name, _, path = spec.partition("=")
        variants[name] = load(Path(path))
    stats: dict[str, dict] = {}
    for spec in args.summary:
        name, _, path = spec.partition("=")
        s = load(Path(path))
        stats[name] = {**s.get("model_stats", {}),
                       **{"transfer": s.get("weight_transfer", {})}}

    if not variants:
        print("no variants given")
        return 1

    names = list(variants)
    w = max(len(n) for n in names + ["baseline"]) + 2

    print("=" * (26 + (w + 10) * len(names)))
    print("US17 -- per-class AP50 against the Tier A baseline")
    print("=" * (26 + (w + 10) * len(names)))
    hdr = f"{'class':22} {'Tier A':>8}"
    for n in names:
        hdr += f" {n:>{w}} {'d/sd':>7}"
    print(hdr)
    for c in CLASSES:
        b = base["per_class"][c]["AP50"]
        row = f"{SHORT[c]:22} {b:8.4f}"
        for n in names:
            v = variants[n]["per_class"].get(c, {}).get("AP50")
            if v is None:
                row += f" {'-':>{w}} {'-':>7}"
            else:
                row += f" {v:{w}.4f} {fmt_sigma(v - b, sd_class.get(c, 0)):>7}"
        mark = "  <- target" if c in TARGET else ""
        print(row + mark)

    b = base["pooled"]["mAP50"]
    row = f"{'POOLED mAP50':22} {b:8.4f}"
    for n in names:
        v = variants[n]["pooled"]["mAP50"]
        row += f" {v:{w}.4f} {v - b:+7.4f}"
    print(row)
    b95 = base["pooled"]["mAP50_95"]
    row = f"{'POOLED mAP50-95':22} {b95:8.4f}"
    for n in names:
        v = variants[n]["pooled"]["mAP50_95"]
        row += f" {v:{w}.4f} {v - b95:+7.4f}"
    print(row)

    print()
    print("Per country")
    hdr = f"{'country':22} {'Tier A':>8}"
    for n in names:
        hdr += f" {n:>{w}} {'d/sd':>7}"
    print(hdr)
    for ct in sorted(base["per_country"], key=lambda k: -base["per_country"][k]["mAP50"]):
        b = base["per_country"][ct]["mAP50"]
        row = f"{ct:22} {b:8.4f}"
        for n in names:
            v = variants[n]["per_country"].get(ct, {}).get("mAP50")
            row += (f" {v:{w}.4f} {fmt_sigma(v - b, sd_country.get(ct, 0)):>7}"
                    if v is not None else f" {'-':>{w}} {'-':>7}")
        print(row)

    if stats:
        print()
        print("Cost of each variant")
        print(f"{'variant':22} {'params':>12} {'GFLOPs':>8} {'transferred':>14}")
        for n in names:
            st = stats.get(n, {})
            tr = st.get("transfer") or {}
            t = f"{tr.get('tensors','?')}/{tr.get('of','?')}"
            print(f"{n:22} {st.get('parameters','?'):>12} "
                  f"{st.get('gflops','?'):>8} {t:>14}")

    # The question US17 was written to answer.
    print()
    print("Verdict on the two target classes (C1, C2)")
    for n in names:
        deltas = []
        for c in TARGET:
            v = variants[n]["per_class"].get(c, {}).get("AP50")
            if v is not None:
                d = v - base["per_class"][c]["AP50"]
                deltas.append((c, d, d / sd_class[c] if sd_class.get(c) else float("nan")))
        if deltas:
            txt = "  ".join(f"{SHORT[c].split()[0]} {d:+.4f} ({z:+.1f} sd)"
                            for c, d, z in deltas)
            better = sum(1 for _, d, _ in deltas if d > 0)
            print(f"  {n:20} {txt}   -> {better}/{len(deltas)} improved")

    if args.out_tex:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in CLASSES:
            b = base["per_class"][c]["AP50"]
            cells = [f"{b:.4f}"]
            for n in names:
                v = variants[n]["per_class"].get(c, {}).get("AP50")
                cells.append(f"{v:.4f}" if v is not None else "---")
            rows.append(f"{SHORT[c]} & " + " & ".join(cells) + r" \\")
        tex = (
            "% Generated by rdd.eval.compare_variants -- do not edit by hand.\n"
            "\\begin{table}[H]\n\\centering\\small\n"
            "\\caption{US17 architecture variants against the Tier A baseline, "
            "per-class AP\\textsubscript{50}. Identical data, split, schedule "
            "and pretrained initialisation; only the architecture differs.}\n"
            "\\label{tab:us17-variants}\n"
            "\\begin{tabular}{l" + "c" * (len(names) + 1) + "}\n\\toprule\n"
            "\\textbf{Class} & \\textbf{Tier A} & "
            + " & ".join(f"\\textbf{{{n}}}" for n in names) + r" \\" + "\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
        args.out_tex.write_text(tex, encoding="utf-8")
        print(f"\nwrote {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
