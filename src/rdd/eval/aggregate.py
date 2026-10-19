r"""
Aggregate Tier A runs into the tables the report needs.

Produces four things the proposal's evaluation protocol demands:

  1. **mean +/- std across seeds** -- single-seed numbers are not evidence.
     Every spread here is the *sample* standard deviation (n-1). With three
     seeds that choice is not cosmetic: on C4_pothole AP50-95 it is 0.000681
     against 0.000556 for the population form, which moves the six-versus-four
     effect size from -14.8 sd to -18.1 sd.
  2. **per-class** breakdown -- never a single pooled number -- at both IoU
     thresholds, with a spread on every figure.
  3. **per-country** breakdown -- H3 is specifically about domain variance, and
     the published challenge leaderboards show >0.25 F1 spread between
     countries inside one model.
  4. **provenance** -- the two split SHA-256 hashes and the class count travel
     with the numbers, into the JSON and into every table caption. Runs that
     disagree on either hash are refused rather than averaged, because
     averaging them would mean combining different dataset states.

It also computes the **H2 comparison** directly: the 6-class baseline against
the 4-class control, restricted to the four classes they share, at AP50 *and*
AP50-95, with each difference expressed in units of the baseline's seed-to-seed
standard deviation. Because both were trained on identical images and identical
splits, any difference is attributable to broadening the taxonomy.

LaTeX is written, not printed: three `booktabs` tables that the report pulls in
with \input, so the tables cannot drift from the metrics.

Usage
-----
    python -m rdd.eval.aggregate --runs Tier-A-Results/metrics \
        --out Tier-A-Results/metrics/aggregate.json \
        --tex-dir Tier-A-Results/metrics/tex
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

SHARED = ["C1_longitudinal_crack", "C2_transverse_crack",
          "C3_alligator_crack", "C4_pothole"]

#: per-class metrics carried through to the JSON and the table, in table order
CLASS_METRICS = ["AP50", "AP50_95", "precision", "recall"]

#: metrics the six-versus-four comparison runs on. Threshold-free only:
#: precision and recall differ between the two models by operating point at a
#: fixed confidence threshold, which is not a capability difference.
H2_METRICS = ["AP50", "AP50_95"]

#: an effect larger than this many baseline seed sd is outside the noise floor
NOISE_FLOOR_SD = 2.0

STD_CONVENTION = ("sample standard deviation across seeds (n-1); "
                  "0.0 where a config has a single seed")

SIGN_CONVENTION = ("delta = control (4-class) minus baseline (6-class); "
                   "a negative delta means the 4-class control scored lower")

EFFECT_DEFINITION = (
    "delta divided by the baseline's seed-to-seed sample standard deviation "
    "(n-1) for the same class and metric. The control arm has a single seed, "
    "so the denominator comes from the baseline arm alone and cannot absorb "
    "control-seed variance."
)

HASH_RE = {
    "train": re.compile(r"^#\s*train\s+sha256\s*:\s*([0-9a-f]{64})\s*$", re.M),
    "val": re.compile(r"^#\s*val\s+sha256\s*:\s*([0-9a-f]{64})\s*$", re.M),
}


class ProvenanceError(RuntimeError):
    """Runs cannot be aggregated together, or cannot be identified at all."""


def mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)    # n-1, deliberately


def fmt(m: float, s: float, places: int = 3) -> str:
    if math.isnan(m):
        return "     --   "
    return f"{m:.{places}f} +/- {s:.{places}f}"


def write_text_lf(path: Path, text: str) -> None:
    """Write LF-terminated text whatever the platform.

    The repository forces LF in every working copy, and these files are
    regenerated on Windows laptops as well as on Linux nodes; letting Python
    translate newlines would churn every generated file between platforms.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


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


# ---- provenance ------------------------------------------------------------

def split_hashes(run: dict) -> dict[str, str]:
    """Pull the two split hashes out of a run's recorded data.yaml header."""
    name = run.get("run", "<unnamed run>")
    header = run.get("data_yaml_header") or ""
    if not header.strip():
        raise ProvenanceError(
            f"{name}: summary.json carries no 'data_yaml_header'. Without it "
            f"the dataset state behind these numbers is unknown, and the "
            f"report requires the split hashes with every figure.")
    found = {}
    for kind, rx in HASH_RE.items():
        m = rx.search(header)
        if not m:
            raise ProvenanceError(
                f"{name}: no '# {kind} sha256 : <64 hex>' line in "
                f"data_yaml_header. Header was:\n{header.rstrip()}")
        found[kind] = m.group(1)
    return found


def n_classes(run: dict) -> int:
    """How many classes the run's label set had.

    The recorded header carries the split hashes but not `nc:`, so the count
    comes from the per-class block the run itself reported.
    """
    return len(run.get("per_class") or {})


def dataset_identity(runs: list[dict]) -> dict[str, str]:
    """One train/val hash pair for the whole set of runs, or refuse.

    A silent hash mismatch would mean averaging results computed on different
    dataset states -- exactly the failure the project's provenance rule exists
    to catch -- so this raises rather than warns.
    """
    seen: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}}
    for r in runs:
        h = split_hashes(r)
        for kind in ("train", "val"):
            seen[kind].setdefault(h[kind], []).append(r.get("run", "?"))
    for kind in ("train", "val"):
        if len(seen[kind]) > 1:
            lines = [f"runs disagree on the {kind} split sha256 -- refusing to "
                     f"aggregate results computed on different dataset states:"]
            for hsh, names in sorted(seen[kind].items()):
                lines.append(f"  {hsh}  <-- {', '.join(sorted(names))}")
            raise ProvenanceError("\n".join(lines))
    return {kind: next(iter(seen[kind])) for kind in ("train", "val")}


def config_class_count(cfg: str, rs: list[dict]) -> int:
    """The class count shared by every seed of one config, or refuse."""
    counts: dict[int, list[str]] = {}
    for r in rs:
        counts.setdefault(n_classes(r), []).append(r.get("run", "?"))
    if len(counts) > 1:
        lines = [f"config {cfg}: seeds disagree on the class count -- these "
                 f"are not seeds of one experiment:"]
        for c, names in sorted(counts.items()):
            lines.append(f"  {c} classes  <-- {', '.join(sorted(names))}")
        raise ProvenanceError("\n".join(lines))
    return next(iter(counts))


# ---- LaTeX -----------------------------------------------------------------

TEX_HEADER = (
    "% Generated by rdd.eval.aggregate -- do not edit by hand.\n"
    "% Tables are generated and never transcribed; regenerate with\n"
    "%   python -m rdd.eval.aggregate --runs <metrics dir> \\\n"
    "%       --out <metrics dir>/aggregate.json --tex-dir <tex dir>\n"
    "% Requires: booktabs, float (for the [H] placement).\n"
)

PRETTY_METRIC = {"AP50": r"AP\textsubscript{50}",
                 "AP50_95": r"AP\textsubscript{50:95}",
                 "precision": "Precision",
                 "recall": "Recall"}

FULL_HASH_TABLE = "tab:tierA-per-class"     # the table carrying both hashes


def tex_label(name: str) -> str:
    """C1_longitudinal_crack -> C1 longitudinal crack (safe in LaTeX text)."""
    return name.replace("_", " ")


def tex_tt(name: str) -> str:
    """A run or config name as typewriter text, underscores escaped."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def tex_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def tex_num(x: float | None, places: int = 4) -> str:
    if x is None or math.isnan(x):
        return "---"
    return f"{x:.{places}f}"


def tex_pm(m: float | None, s: float | None, places: int = 4) -> str:
    if m is None or math.isnan(m):
        return "---"
    return f"{m:.{places}f} $\\pm$ {s:.{places}f}"


def tex_signed(x: float | None, places: int = 4) -> str:
    if x is None or math.isnan(x):
        return "---"
    return f"${x:+.{places}f}$"


def tex_effect(x: float | None) -> str:
    if x is None or math.isnan(x):
        return "---"
    return f"${x:+.1f}$"


def seed_phrase(block: dict) -> str:
    seeds = [str(s) for s in block.get("seeds", []) if s is not None]
    n = block.get("n_seeds", len(seeds))
    if n == 1:
        return f"a single training seed ({seeds[0]})" if seeds else "a single seed"
    return f"{n} training seeds ({', '.join(seeds)})"


def provenance_caption(nc: int, hashes: dict[str, str]) -> str:
    """Class count and abbreviated split hashes, for a table caption."""
    return (f"{nc}-class label set; split SHA-256 "
            f"train \\texttt{{{hashes['train'][:12]}}}, "
            f"val \\texttt{{{hashes['val'][:12]}}} (first 12 hex characters).")


def provenance_note(nc: int, hashes: dict[str, str], full: bool) -> str:
    """Table footnote. The full 64-hex hashes are given once, in `full` form."""
    bits = [f"{nc}-class label set.",
            "Spreads are sample standard deviations ($n-1$) across seeds."]
    if full:
        bits.append(f"Split SHA-256 in full: train "
                    f"\\texttt{{{hashes['train']}}}; "
                    f"val \\texttt{{{hashes['val']}}}.")
    else:
        bits.append(f"Split SHA-256 train \\texttt{{{hashes['train'][:12]}}}, "
                    f"val \\texttt{{{hashes['val'][:12]}}} "
                    f"(full values in Table~\\ref{{{FULL_HASH_TABLE}}}).")
    return " ".join(bits)


def tex_table(caption: str, label: str, colspec: str, head: str,
              body: list[str], note: str) -> str:
    out = [TEX_HEADER.rstrip("\n"),
           "\\begin{table}[H]",
           "\\centering",
           "\\small",
           f"\\caption{{{caption}}}",
           f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{colspec}}}",
           "\\toprule",
           head,
           "\\midrule"]
    out.extend(body)
    out += ["\\bottomrule",
            "\\end{tabular}",
            "",
            "\\vspace{2pt}",
            f"{{\\footnotesize {note}}}",
            "\\end{table}"]
    return "\n".join(out) + "\n"


def write_per_class_tex(path: Path, cfg: str, block: dict,
                        hashes: dict[str, str]) -> None:
    nc = block["n_classes"]
    head = ("\\textbf{Class} & "
            + " & ".join(f"\\textbf{{{PRETTY_METRIC[m]}}}" for m in CLASS_METRICS)
            + " \\\\")
    body = []
    for name in sorted(block["per_class"]):
        c = block["per_class"][name]
        cells = [tex_pm(c[f"{m}_mean"], c[f"{m}_std"]) for m in CLASS_METRICS]
        body.append(f"{tex_label(name)} & " + " & ".join(cells) + " \\\\")
    pooled = block["pooled"]
    body += ["\\midrule",
             "Pooled (all classes) & "
             + tex_pm(pooled["mAP50_mean"], pooled["mAP50_std"]) + " & "
             + tex_pm(pooled["mAP50_95_mean"], pooled["mAP50_95_std"])
             + " & --- & --- \\\\"]
    caption = (f"Tier~A {nc}-class baseline ({tex_tt(cfg)}): per-class "
               f"detection quality, mean $\\pm$ standard deviation over "
               f"{seed_phrase(block)}. {provenance_caption(nc, hashes)}")
    write_text_lf(path,
                  tex_table(caption, FULL_HASH_TABLE, "l c c c c", head, body,
                            provenance_note(nc, hashes, full=True)))


def write_per_country_tex(path: Path, cfg: str, block: dict,
                          hashes: dict[str, str]) -> None:
    nc = block["n_classes"]
    head = ("\\textbf{Country} & \\textbf{Val.\\ images} & "
            "\\textbf{mAP\\textsubscript{50}} & \\textbf{F1} \\\\")
    body = []
    for country in sorted(block["per_country"]):
        c = block["per_country"][country]
        body.append(f"{tex_label(country)} & {tex_int(c['n_val_images'])} & "
                    f"{tex_pm(c['mAP50_mean'], c['mAP50_std'])} & "
                    f"{tex_pm(c['f1_mean'], c['f1_std'])} \\\\")
    spread = block.get("per_country_spread_mAP50")
    if spread is not None:
        body += ["\\midrule",
                 f"Spread (max $-$ min) & --- & {tex_num(spread)} & --- \\\\"]
    caption = (f"Tier~A {nc}-class baseline ({tex_tt(cfg)}): per-country "
               f"performance, mean $\\pm$ standard deviation over "
               f"{seed_phrase(block)}. The spread row is the range of the "
               f"country means and is the H3 figure. "
               f"{provenance_caption(nc, hashes)}")
    write_text_lf(path,
                  tex_table(caption, "tab:tierA-per-country", "l r c c", head,
                            body, provenance_note(nc, hashes, full=False)))


def write_h2_tex(path: Path, h2: dict, hashes: dict[str, str]) -> None:
    nb, ncl = h2["baseline_n_classes"], h2["control_n_classes"]
    head = (f"\\textbf{{Class}} & \\textbf{{{nb}-class}} & "
            f"\\textbf{{{ncl}-class}} & \\textbf{{$\\Delta$}} & "
            f"\\textbf{{$\\Delta/\\sigma$}} \\\\")
    body: list[str] = []
    for i, metric in enumerate(H2_METRICS):
        block = h2["metrics"].get(metric)
        if not block or not block["per_class"]:
            continue
        if i:
            body.append("\\addlinespace")
        body.append(f"\\multicolumn{{5}}{{l}}"
                    f"{{\\emph{{{PRETTY_METRIC[metric]}}}}} \\\\")
        for name in SHARED:
            r = block["per_class"].get(name)
            if not r:
                continue
            body.append(
                f"\\quad {tex_label(name)} & "
                f"{tex_pm(r['baseline_mean'], r['baseline_std'])} & "
                f"{tex_num(r['control'])} & {tex_signed(r['delta'])} & "
                f"{tex_effect(r['delta_over_baseline_sd'])} \\\\")
        body.append("\\quad \\emph{mean delta} & --- & --- & "
                    f"{tex_signed(block['mean_delta'])} & --- \\\\")
    caption = (f"Cost of broadening the taxonomy: the {ncl}-class control "
               f"({tex_tt(h2['control'])}) against the {nb}-class baseline "
               f"({tex_tt(h2['baseline'])}) on the four classes they share, at "
               f"both IoU thresholds. Identical images and identical splits; "
               f"only the label set differs. $\\Delta$ is control minus "
               f"baseline, so a negative figure means the {ncl}-class control "
               f"scored lower. $\\sigma$ is the baseline's seed-to-seed "
               f"standard deviation for that class and metric. Baseline "
               f"{nb}-class and control {ncl}-class label sets; split SHA-256 "
               f"train \\texttt{{{hashes['train'][:12]}}}, "
               f"val \\texttt{{{hashes['val'][:12]}}} (first 12 hex "
               f"characters).")
    note = (f"The control arm has one seed, so $\\sigma$ comes from the "
            f"baseline arm alone and cannot absorb control-seed variance. "
            f"Spreads are sample standard deviations ($n-1$). Split SHA-256 "
            f"train \\texttt{{{hashes['train'][:12]}}}, val "
            f"\\texttt{{{hashes['val'][:12]}}} (full values in "
            f"Table~\\ref{{{FULL_HASH_TABLE}}}). Precision and recall are "
            f"excluded deliberately: between these two models they differ by "
            f"operating point at a fixed confidence threshold, which is not a "
            f"capability difference.")
    write_text_lf(path,
                  tex_table(caption, "tab:tierA-six-vs-four", "l c c c c",
                            head, body, note))


# ---- main ------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate Tier A runs.")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tex-dir", type=Path, default=None,
                    help="where the .tex tables are written "
                         "(default: alongside --out)")
    args = ap.parse_args(argv)

    runs = load_runs(args.runs)
    if not runs:
        print(f"no summary.json found under {args.runs}")
        return 1
    print(f"found {len(runs)} run(s)\n")

    try:
        hashes = dataset_identity(runs)
    except ProvenanceError as e:
        print(f"PROVENANCE ERROR\n{e}", file=sys.stderr)
        return 2

    print("DATASET IDENTITY -- identical across every run aggregated here")
    print(f"  train sha256 : {hashes['train']}")
    print(f"  val   sha256 : {hashes['val']}\n")

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)

    report: dict = {"split_sha256": hashes,
                    "std_convention": STD_CONVENTION,
                    "configs": {},
                    "h2_comparison": None}

    # ---- pooled ------------------------------------------------------------
    print("=" * 78)
    print("POOLED  (mean +/- std across seeds)")
    print("=" * 78)
    print(f"{'config':<22}{'classes':>8}{'seeds':>6}  "
          f"{'mAP50':^18}{'mAP50-95':^18}")
    for cfg in sorted(groups):
        rs = groups[cfg]
        try:
            nc = config_class_count(cfg, rs)
        except ProvenanceError as e:
            print(f"PROVENANCE ERROR\n{e}", file=sys.stderr)
            return 2
        m50, s50 = mean_std([r["pooled"]["mAP50"] for r in rs])
        m95, s95 = mean_std([r["pooled"]["mAP50_95"] for r in rs])
        print(f"{cfg:<22}{nc:>8}{len(rs):>6}  "
              f"{fmt(m50, s50):^18}{fmt(m95, s95):^18}")
        report["configs"][cfg] = {
            "n_seeds": len(rs),
            "seeds": [r.get("seed") for r in rs],
            "n_classes": nc,
            "split_sha256": hashes,
            "pooled": {"mAP50_mean": m50, "mAP50_std": s50,
                       "mAP50_95_mean": m95, "mAP50_95_std": s95},
        }

    # ---- per class ---------------------------------------------------------
    for cfg in sorted(groups):
        rs = groups[cfg]
        names = sorted({n for r in rs for n in r.get("per_class", {})})
        if not names:
            continue
        print(f"\nPER CLASS -- {cfg}  "
              f"({report['configs'][cfg]['n_classes']} classes)")
        print(f"  {'class':<26}{'AP50':^18}{'AP50-95':^18}"
              f"{'precision':^18}{'recall':^18}")
        per_class = {}
        for n in names:
            entry: dict[str, float] = {}
            cells = []
            for metric in CLASS_METRICS:
                vals = [r["per_class"][n][metric] for r in rs
                        if n in r.get("per_class", {})]
                m, s = mean_std(vals)
                entry[f"{metric}_mean"], entry[f"{metric}_std"] = m, s
                cells.append(fmt(m, s))
            print(f"  {n:<26}" + "".join(f"{c:^18}" for c in cells))
            per_class[n] = entry
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
                    if c in r.get("per_country", {})
                    and "mAP50" in r["per_country"][c]]
            if not vals:
                continue
            m, s = mean_std([v["mAP50"] for v in vals])
            f, sf = mean_std([v["f1"] for v in vals])
            n = vals[0].get("n_images", 0)
            print(f"  {c:<20}{fmt(m, s):^18}{fmt(f, sf):^18}{n:>8}")
            per_country[c] = {"mAP50_mean": m, "mAP50_std": s,
                              "f1_mean": f, "f1_std": sf, "n_val_images": n}
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
        print(f"     sign convention: {SIGN_CONVENTION}")
        print("     effect size: delta / the baseline's seed sd (n-1) for that")
        print("                  class and metric")
        print("=" * 78)
        h2: dict = {"baseline": base,
                    "control": ctrl,
                    "baseline_n_classes": report["configs"][base]["n_classes"],
                    "control_n_classes": report["configs"][ctrl]["n_classes"],
                    "split_sha256": hashes,
                    "sign_convention": SIGN_CONVENTION,
                    "effect_size_definition": EFFECT_DEFINITION,
                    "metrics": {}}
        for metric in H2_METRICS:
            print(f"\n  {metric}")
            print(f"  {'class':<26}{'6-class':>10}{'sd':>10}{'4-class':>10}"
                  f"{'delta':>11}{'delta/sd':>11}")
            rows: dict[str, dict] = {}
            for n in SHARED:
                b = report["configs"][base].get("per_class", {}).get(n)
                c = report["configs"][ctrl].get("per_class", {}).get(n)
                if not b or not c:
                    continue
                bm, bs = b[f"{metric}_mean"], b[f"{metric}_std"]
                cm = c[f"{metric}_mean"]
                d = cm - bm
                eff = (d / bs) if (bs and not math.isnan(bs)) else None
                rows[n] = {"baseline_mean": bm, "baseline_std": bs,
                           "control": cm, "delta": d,
                           "delta_over_baseline_sd": eff}
                eff_s = "--" if eff is None else f"{eff:+.1f}"
                print(f"  {n:<26}{bm:>10.4f}{bs:>10.4f}{cm:>10.4f}"
                      f"{d:>+11.4f}{eff_s:>11}")
            if not rows:
                continue
            md = statistics.mean(r["delta"] for r in rows.values())
            effs = [abs(r["delta_over_baseline_sd"]) for r in rows.values()
                    if r["delta_over_baseline_sd"] is not None]
            worst = max(effs) if effs else None
            print(f"  {'MEAN DELTA':<26}{'':>10}{'':>10}{'':>10}{md:>+11.4f}")
            if worst is not None:
                verdict = ("inside" if worst <= NOISE_FLOOR_SD else "OUTSIDE")
                print(f"  largest |delta/sd| = {worst:.1f} -- {verdict} the "
                      f"+/-{NOISE_FLOOR_SD:.0f} sd seed noise floor")
            h2["metrics"][metric] = {"per_class": rows, "mean_delta": md,
                                     "max_abs_effect_size": worst}
        report["h2_comparison"] = h2

        w50 = h2["metrics"].get("AP50", {}).get("max_abs_effect_size")
        w95 = h2["metrics"].get("AP50_95", {}).get("max_abs_effect_size")
        if w50 is not None and w95 is not None:
            print()
            if w50 <= NOISE_FLOOR_SD < w95:
                print("  Interpretation: broadening is free WITHIN SEED NOISE at")
                print("  AP50 and not at AP50-95. The write-up must report both")
                print("  and say which claim rests on which; the AP50-only")
                print("  conclusion does not survive the stricter threshold.")
            elif w95 <= NOISE_FLOOR_SD:
                print("  Interpretation: broadening is roughly FREE on the shared")
                print("  classes at both thresholds.")
            else:
                print("  Interpretation: differences exceed the seed noise floor")
                print("  at both thresholds -- quantify and report per class.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(args.out, json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    # ---- LaTeX -------------------------------------------------------------
    tex_dir = args.tex_dir or args.out.parent
    tex_dir.mkdir(parents=True, exist_ok=True)
    if base:
        block = report["configs"][base]
        if block.get("per_class"):
            p = tex_dir / "tierA_per_class.tex"
            write_per_class_tex(p, base, block, hashes)
            print(f"wrote {p}")
        if block.get("per_country"):
            p = tex_dir / "tierA_per_country.tex"
            write_per_country_tex(p, base, block, hashes)
            print(f"wrote {p}")
    else:
        print("  [warn] no *base6* config found -- per-class and per-country "
              "tables not written")
    if report["h2_comparison"]:
        p = tex_dir / "tierA_six_vs_four.tex"
        write_h2_tex(p, report["h2_comparison"], hashes)
        print(f"wrote {p}")
    else:
        print("  [warn] no baseline/control pair found -- delta table not "
              "written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
