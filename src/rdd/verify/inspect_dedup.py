"""
Inspect a de-duplication report before trusting it.

Perceptual hashing has a well-known failure mode: **low-information images**.
A near-uniform frame — blank asphalt, an overexposed or very dark shot — carries
almost no structure, so its pHash collapses toward a common value and it
collides with other unrelated flat frames. Those are false duplicates, and
dropping them silently discards real training data.

The tell is the **group-size distribution**. Genuine video near-duplicates form
small groups (2-4 consecutive frames). A group of 30+ images whose sequence
numbers are scattered across the whole subset is almost certainly a
low-information collision, not a repeated photograph.

Usage
-----
    python -m rdd.verify.inspect_dedup --report dedup_report.json
    python -m rdd.verify.inspect_dedup --report ... --coco rdd2022_train.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

SEQ = re.compile(r"_(\d+)\.")


def seq_of(name: str) -> int | None:
    m = SEQ.search(name)
    return int(m.group(1)) if m else None


def consecutiveness(names: list[str]) -> float | None:
    """Median gap between sorted sequence numbers within a group.

    ~1 means consecutive video frames (a genuine duplicate run).
    Large means scattered across the subset (a likely hash collision).
    """
    seqs = sorted(s for s in (seq_of(n) for n in names) if s is not None)
    if len(seqs) < 2:
        return None
    gaps = [b - a for a, b in zip(seqs, seqs[1:])]
    return statistics.median(gaps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a dedup report.")
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--coco", type=Path, help="also audit the view/country fields")
    ap.add_argument("--suspect-size", type=int, default=5,
                    help="groups at least this large are examined for collisions")
    ap.add_argument("--suspect-gap", type=int, default=10,
                    help="median sequence gap above this looks like a collision")
    args = ap.parse_args(argv)

    r = json.loads(args.report.read_text(encoding="utf-8"))
    groups = r["groups"]

    sizes = collections.Counter(g["size"] for g in groups)
    total_drops = sum((s - 1) * n for s, n in sizes.items())

    print("=" * 66)
    print("GROUP SIZE DISTRIBUTION")
    print("=" * 66)
    print(f"  {'size':>5}{'groups':>9}{'drops':>9}{'% of drops':>12}")
    for s in sorted(sizes):
        d = (s - 1) * sizes[s]
        print(f"  {s:>5}{sizes[s]:>9}{d:>9}{100 * d / max(total_drops, 1):>11.1f}%")
    print(f"\n  total drops: {total_drops}")

    # --- collision screen ---------------------------------------------------
    suspects = []
    for g in groups:
        if g["size"] < args.suspect_size:
            continue
        names = [g["keep"]] + g["drop"]
        med = consecutiveness(names)
        if med is not None and med > args.suspect_gap:
            suspects.append((g["size"], med, g["keep"], names[:4]))

    suspect_drops = sum(s - 1 for s, _, _, _ in suspects)

    print("\n" + "=" * 66)
    print("LIKELY HASH COLLISIONS (large groups, non-consecutive frames)")
    print("=" * 66)
    if not suspects:
        print("  none -- every large group looks like a genuine consecutive run")
    else:
        for size, med, keep, sample in sorted(suspects, reverse=True)[:10]:
            print(f"  size {size:>3}  median seq gap {med:>7.0f}  e.g. {sample[:3]}")
        print(f"\n  {len(suspects)} suspect group(s), accounting for "
              f"{suspect_drops} of {total_drops} drops "
              f"({100 * suspect_drops / max(total_drops, 1):.1f}%)")
        print("\n  These are probably low-information frames (blank asphalt,")
        print("  over/under-exposed) that collide rather than genuine repeats.")
        print("  Consider: lower --threshold, or exclude them from the drop list.")

    # --- genuine-looking runs ------------------------------------------------
    genuine = [g for g in groups
               if g["size"] < args.suspect_size
               or (consecutiveness([g["keep"]] + g["drop"]) or 999) <= args.suspect_gap]
    genuine_drops = sum(g["size"] - 1 for g in genuine)
    print("\n" + "=" * 66)
    print(f"  genuine-looking duplicate drops : {genuine_drops}")
    print(f"  suspect collision drops         : {total_drops - genuine_drops}")
    print("=" * 66)

    if args.coco and args.coco.exists():
        c = json.loads(args.coco.read_text(encoding="utf-8"))
        views = collections.Counter((i["country"], i["view"]) for i in c["images"])
        print("\nCOUNTRY / VIEW as recorded in the COCO store")
        for (country, view), n in sorted(views.items()):
            flag = ""
            if "Drone" in country and view != "topdown":
                flag = "   <-- WRONG: drone imagery marked as forward-facing"
            print(f"  {country:<18} {view:<10} {n:>6}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
