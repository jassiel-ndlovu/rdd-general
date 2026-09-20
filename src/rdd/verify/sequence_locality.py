"""
Test whether RDD2022's sequence numbers carry temporal/spatial locality.

Why this matters twice over
---------------------------
Two parts of the pipeline assume that `<Country>_<NNNNNN>` numbering follows
capture order, so that images with adjacent numbers are the same stretch of road
seconds apart:

  * ``rdd.data.dedup``  -- only merges near-duplicates within ``--max-seq-gap``
    frames, on the grounds that genuine duplicates are temporally adjacent.
  * ``rdd.data.splits`` -- groups images into "segments" by ``sequence // block``
    and assigns whole segments to a split, to stop near-identical frames leaking
    across the train/val boundary.

If the release shuffled the numbering, BOTH are wrong: the dedup guard would
reject real duplicates, and the split grouping would provide no protection at
all while appearing to.

The test
--------
Compare the perceptual-hash distance between **adjacent-numbered pairs** against
**randomly chosen pairs from the same subset**. If numbering is temporal,
adjacent pairs are markedly more similar. If the numbering is shuffled, the two
distributions coincide.

Usage
-----
    python -m rdd.verify.sequence_locality \\
        --coco /opt/rdd/data/coco/rdd2022_train.json \\
        --root /opt/rdd/data/rdd2022/RDD2022 --sample 400
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
from pathlib import Path

SEQ = re.compile(r"_(\d+)\.")


def phash_int(path: Path) -> int | None:
    try:
        from PIL import Image
        import imagehash
        with Image.open(path) as im:
            try:
                im.draft("L", (64, 64))
            except Exception:
                pass
            h = imagehash.phash(im, hash_size=8)
        bits = 0
        for b in h.hash.flatten():
            bits = (bits << 1) | int(bool(b))
        return bits
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is the sequence numbering temporal?")
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=300,
                    help="pairs to sample per country per condition")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    by_country: dict[str, dict[int, str]] = collections.defaultdict(dict)
    for im in coco["images"]:
        m = SEQ.search(im["file_name"])
        if m:
            by_country[im["country"]][int(m.group(1))] = im["rel_path"]

    print("=" * 74)
    print("SEQUENCE LOCALITY TEST")
    print("  adjacent = images numbered n and n+1")
    print("  random   = two images drawn at random from the same subset")
    print("  lower perceptual-hash distance = more visually similar")
    print("=" * 74)
    print(f"  {'subset':<18}{'adjacent':>22}{'random':>22}{'verdict':>11}")

    overall = {"adj": [], "rnd": []}
    for country in sorted(by_country):
        seqs = sorted(by_country[country])
        if len(seqs) < 50:
            continue
        seq_set = set(seqs)

        adj_pairs = [(n, n + 1) for n in seqs if (n + 1) in seq_set]
        if not adj_pairs:
            print(f"  {country:<18} no adjacent pairs")
            continue
        rng.shuffle(adj_pairs)
        adj_pairs = adj_pairs[:args.sample]

        rnd_pairs = [(rng.choice(seqs), rng.choice(seqs)) for _ in range(len(adj_pairs))]
        rnd_pairs = [(a, b) for a, b in rnd_pairs if a != b]

        cache: dict[int, int] = {}

        def h(n: int) -> int | None:
            if n not in cache:
                v = phash_int(args.root / by_country[country][n])
                if v is None:
                    return None
                cache[n] = v
            return cache[n]

        def dists(pairs):
            out = []
            for a, b in pairs:
                ha, hb = h(a), h(b)
                if ha is not None and hb is not None:
                    out.append(hamming(ha, hb))
            return out

        da, dr = dists(adj_pairs), dists(rnd_pairs)
        if not da or not dr:
            continue
        ma, mr = statistics.median(da), statistics.median(dr)
        overall["adj"] += da
        overall["rnd"] += dr

        near_a = 100 * sum(1 for d in da if d <= 4) / len(da)
        near_r = 100 * sum(1 for d in dr if d <= 4) / len(dr)
        verdict = "TEMPORAL" if ma < mr - 2 else "shuffled?"
        print(f"  {country:<18}"
              f"median {ma:>4.1f}  <=4: {near_a:>5.1f}%"
              f"   median {mr:>4.1f}  <=4: {near_r:>5.1f}%"
              f"{verdict:>11}")

    if overall["adj"] and overall["rnd"]:
        ma = statistics.median(overall["adj"])
        mr = statistics.median(overall["rnd"])
        na = 100 * sum(1 for d in overall["adj"] if d <= 4) / len(overall["adj"])
        nr = 100 * sum(1 for d in overall["rnd"] if d <= 4) / len(overall["rnd"])
        print("=" * 74)
        print(f"  OVERALL   adjacent median {ma:.1f} ({na:.1f}% within 4)"
              f"   random median {mr:.1f} ({nr:.1f}% within 4)")
        print()
        if ma < mr - 2:
            print("  VERDICT: numbering carries locality. Adjacent images are")
            print("  measurably more similar than random ones, so both the dedup")
            print("  temporal guard and the segment-based split grouping are sound.")
        else:
            print("  VERDICT: NO detectable locality. Adjacent images are no more")
            print("  similar than random pairs, which means the release shuffled the")
            print("  numbering. BOTH the dedup --max-seq-gap guard AND the segment")
            print("  grouping in splits.py are invalid and must be reworked --")
            print("  the split would give no protection against leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
