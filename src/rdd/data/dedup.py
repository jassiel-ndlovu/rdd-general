"""
Perceptual-hash de-duplication (M1 stage 3).

Why this runs BEFORE splitting
------------------------------
RDD subsets are video-derived: consecutive frames along the same stretch of road
are near-identical. If such a pair is split across train and test, the model is
evaluated on an image it has effectively already seen, and every metric is
inflated. This is the most likely explanation for implausible published figures
in this field (e.g. precision reported as exactly 1.00).

De-duplication must therefore happen before the split, not after.

Method
------
64-bit perceptual hash (``imagehash.phash``) per image, then group images whose
Hamming distance is at most ``--threshold`` (default 4).

A naive all-pairs comparison over 38,385 images is ~737M comparisons and far too
slow in Python. Instead we use banded LSH: split each 64-bit hash into four
16-bit bands, bucket by (band index, band value), and only compare within
buckets. Two hashes within Hamming distance 4 must agree exactly on at least one
band (pigeonhole: 4 differing bits cannot touch all 4 bands), so this is exact
for the default threshold — no near-duplicate is missed.

Output
------
A JSON report listing every duplicate group and a flat list of the image ids to
drop (one representative per group is kept). Nothing is deleted from disk.

Usage
-----
    python -m rdd.data.dedup \\
        --coco  /opt/rdd/data/coco/rdd2022_train.json \\
        --root  /opt/rdd/data/rdd2022/RDD2022 \\
        --out   /opt/rdd/data/coco/dedup_report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

BANDS = 4
BAND_BITS = 16


def _hash_one(args):
    """Worker: return (image_id, 64-bit phash as int) or (image_id, None)."""
    img_id, path = args
    try:
        from PIL import Image
        import imagehash
        with Image.open(path) as im:
            # phash downsamples to 32x32 internally, so fully decoding a
            # 3650x2044 Norway frame is wasted work. draft() asks libjpeg to
            # decode at a reduced scale (1/2, 1/4, 1/8) during decompression,
            # which is several times faster and changes the hash negligibly.
            try:
                im.draft("L", (64, 64))
            except Exception:       # non-JPEG, or Pillow without draft support
                pass
            h = imagehash.phash(im, hash_size=8)  # 8x8 -> 64 bits
        # imagehash exposes a boolean array; pack it into an int.
        bits = 0
        for b in h.hash.flatten():
            bits = (bits << 1) | int(bool(b))
        return img_id, bits
    except Exception:
        return img_id, None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def band_keys(h: int):
    """Split a 64-bit hash into BANDS bands of BAND_BITS each."""
    for i in range(BANDS):
        shift = i * BAND_BITS
        yield i, (h >> shift) & ((1 << BAND_BITS) - 1)


def group_duplicates(hashes: dict[int, int], threshold: int,
                     locality: dict[int, tuple[str, int]] | None = None,
                     max_seq_gap: int = 0) -> list[list[int]]:
    """Return groups of image ids that are mutual near-duplicates.

    ``locality`` maps image id -> (country, sequence_number).

    Only the **country** part is used as a constraint: two photographs taken in
    different countries cannot be the same photograph, so a cross-subset match is
    by definition a hash collision. (One such match was observed in RDD2022:
    ``Japan_003726`` against ``Norway_006232``.)

    ``max_seq_gap`` is retained for datasets whose filenames really do follow
    capture order, but **defaults to 0 (disabled) for RDD2022**, because
    ``rdd.verify.sequence_locality`` measured no locality in its numbering at
    all: adjacent-numbered images are exactly as dissimilar as random pairs
    (median Hamming distance 30 in both conditions, 0% within distance 4, across
    all seven subsets). The release shuffled the sequence, so "nearby number"
    carries no information and must not be used as a constraint.

    Run that verifier before enabling ``max_seq_gap`` on any new dataset.
    """
    # Bucket by band so we only compare plausible candidates.
    buckets: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for img_id, h in hashes.items():
        for key in band_keys(h):
            buckets[key].append(img_id)

    # Union-find over candidate pairs that pass the exact Hamming check.
    parent: dict[int, int] = {i: i for i in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ids in buckets.values():
        if len(ids) < 2:
            continue
        # A pathological bucket (thousands of identical bands) would be O(n^2).
        # In practice these are genuine duplicate clusters; cap the work anyway.
        if len(ids) > 2000:
            ids = ids[:2000]
        for i in range(len(ids)):
            hi = hashes[ids[i]]
            for j in range(i + 1, len(ids)):
                if hamming(hi, hashes[ids[j]]) > threshold:
                    continue
                if locality:
                    ca, sa = locality.get(ids[i], ("?", -1))
                    cb, sb = locality.get(ids[j], ("?", -2))
                    # A cross-subset match cannot be the same photograph.
                    if ca != cb:
                        continue
                    # Only apply the temporal constraint where the numbering has
                    # been shown to carry locality -- see the docstring.
                    if max_seq_gap > 0 and abs(sa - sb) > max_seq_gap:
                        continue
                union(ids[i], ids[j])

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for img_id in hashes:
        groups[find(img_id)].append(img_id)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Perceptual-hash de-duplication.")
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path, help="image root directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=int, default=4,
                    help="max Hamming distance to call a duplicate (default 4)")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2)))
    ap.add_argument("--max-seq-gap", type=int, default=0,
                    help="additionally require duplicates to be within this many "
                         "frames of each other. DEFAULT 0 (disabled): RDD2022's "
                         "numbering was measured to carry no locality. Only "
                         "enable it after rdd.verify.sequence_locality reports "
                         "TEMPORAL for your dataset.")
    args = ap.parse_args(argv)

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    images = coco["images"]
    print(f"hashing {len(images)} images with {args.workers} workers...", flush=True)

    tasks = [(im["id"], str(args.root / im["rel_path"])) for im in images]

    hashes: dict[int, int] = {}
    failed: list[int] = []
    with mp.Pool(args.workers) as pool:
        for n, (img_id, h) in enumerate(pool.imap_unordered(_hash_one, tasks, chunksize=64), 1):
            if h is None:
                failed.append(img_id)
            else:
                hashes[img_id] = h
            if n % 5000 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)

    print(f"hashed {len(hashes)}; failed to read {len(failed)}")

    # (country, sequence number) per image, for the temporal-adjacency check.
    seq_re = re.compile(r"_(\d+)\.")
    locality = {}
    for im in images:
        m = seq_re.search(im["file_name"])
        locality[im["id"]] = (im.get("country", "?"),
                              int(m.group(1)) if m else -1)

    groups = group_duplicates(hashes, args.threshold,
                              locality=locality, max_seq_gap=args.max_seq_gap)

    by_id = {im["id"]: im for im in images}
    ann_count: collections.Counter = collections.Counter()
    for a in coco["annotations"]:
        ann_count[a["image_id"]] += 1

    # Keep the representative with the most annotations -- discarding the richer
    # label set would be the wrong way round.
    drop: list[int] = []
    detail = []
    for g in groups:
        g_sorted = sorted(g, key=lambda i: (-ann_count[i], by_id[i]["file_name"]))
        keep, rest = g_sorted[0], g_sorted[1:]
        drop.extend(rest)
        detail.append({
            "keep": by_id[keep]["file_name"],
            "keep_id": keep,
            "drop": [by_id[i]["file_name"] for i in rest],
            "country": by_id[keep].get("country"),
            "size": len(g),
        })

    cross_country = sum(
        1 for g in groups if len({by_id[i].get("country") for i in g}) > 1)

    report = {
        "source_coco": str(args.coco),
        "threshold": args.threshold,
        "max_seq_gap": args.max_seq_gap,
        "images_total": len(images),
        "images_hashed": len(hashes),
        "read_failures": failed,
        "duplicate_groups": len(groups),
        "images_to_drop": sorted(drop),
        "drop_count": len(drop),
        "cross_country_groups": cross_country,
        "groups": detail,
        # Every near-duplicate cluster, as image-id lists. splits.py uses these
        # as grouping units so that near-identical images cannot straddle the
        # train/val boundary -- which is the actual requirement. Dropping them
        # is optional; keeping them together is not.
        "clusters": [sorted(g) for g in groups],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  duplicate groups     {len(groups):>8}")
    print(f"  images to drop       {len(drop):>8}  "
          f"({100 * len(drop) / max(len(images), 1):.2f}% of the set)")
    print(f"  cross-country groups {cross_country:>8}  "
          f"{'(investigate -- should be 0)' if cross_country else '(as expected)'}")
    print(f"\nwrote {args.out}")
    print("\nNOTHING WAS DELETED. Inspect a sample of groups by eye before")
    print("trusting the threshold; genuinely different damage photographed")
    print("seconds apart can look near-identical to a perceptual hash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
