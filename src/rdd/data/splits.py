"""
Grouped train/validation splits (M1 stage 4).

Why not a plain random split
----------------------------
RDD subsets are video-derived, so the corpus contains near-identical frames of
the same stretch of road. A per-image random split can put such a pair on both
sides of the boundary, and the model is then evaluated on an image it has
effectively already seen — inflating every metric.

Grouping key: perceptual-hash clusters, NOT sequence numbers
------------------------------------------------------------
An earlier version of this module chunked consecutive sequence numbers into
blocks and treated each block as a road segment. **That was wrong and provided
no protection at all.** ``rdd.verify.sequence_locality`` measured RDD2022's
numbering and found no locality whatsoever: adjacent-numbered images are exactly
as dissimilar as random pairs (median perceptual-hash distance 30 in both
conditions, 0% within distance 4, across all seven subsets). The release shuffled
the numbering, so ``sequence // block`` grouped unrelated images while looking
principled.

The correct grouping key needs no ordering assumption. The requirement is simply
that **near-identical images must not straddle the train/val boundary**, so we
group by the near-duplicate clusters that ``rdd.data.dedup`` already computes and
assign whole clusters to a split. Images in no cluster are their own group.

This is strictly stronger than the old proxy: it acts directly on visual
similarity, which is the thing that actually causes leakage, rather than on a
filename convention that turned out to be meaningless.

Stratification
--------------
Segments are shuffled and allocated **per country**, so every country appears in
both splits in roughly its original proportion. Without this, a country-blind
shuffle can starve the validation set of whole countries — which matters because
per-country reporting is part of the evaluation protocol.

Output
------
A manifest containing the image-id lists plus a SHA-256 of each sorted list.
**Every results table must cite that hash**, so a number can always be traced to
the exact data that produced it.

Usage
-----
    python -m rdd.data.splits \\
        --coco  /opt/rdd/data/coco/rdd2022_train.json \\
        --dedup /opt/rdd/data/coco/dedup_report.json \\
        --out   /opt/rdd/data/splits/rdd2022_core.json \\
        --val-fraction 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SEQ_RE = re.compile(r"_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def sequence_number(file_name: str) -> int | None:
    m = SEQ_RE.search(file_name)
    return int(m.group(1)) if m else None


def sha256_of_ids(ids: list[int]) -> str:
    payload = ",".join(str(i) for i in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Country- and segment-grouped splits.")
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--dedup", type=Path, help="dedup report; its drops are excluded")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="also DROP near-duplicates instead of merely keeping "
                         "each cluster within one split. Keeping them is the "
                         "default: it preserves data, and holding a cluster "
                         "together already removes the leakage risk.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-view", default="topdown",
                    help="capture view to exclude, or 'none' (default: topdown, "
                         "which drops the China_Drone aerial subset)")
    args = ap.parse_args(argv)

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    images = coco["images"]

    dropped: set[int] = set()
    clusters: list[list[int]] = []
    if args.dedup and args.dedup.exists():
        rep = json.loads(args.dedup.read_text(encoding="utf-8"))
        clusters = rep.get("clusters", [])
        if args.drop_duplicates:
            dropped = set(rep.get("images_to_drop", []))
            print(f"dropping {len(dropped)} near-duplicate images")
        else:
            print(f"keeping near-duplicates; {len(clusters)} cluster(s) will be "
                  f"held together within a split")

    excluded_view = 0
    kept = []
    for im in images:
        if im["id"] in dropped:
            continue
        if args.exclude_view != "none" and im.get("view") == args.exclude_view:
            excluded_view += 1
            continue
        kept.append(im)
    if excluded_view:
        print(f"excluding {excluded_view} images with view='{args.exclude_view}'")
    print(f"{len(kept)} images eligible for splitting")

    # --- build groups: near-duplicate clusters, singletons otherwise --------
    eligible = {im["id"] for im in kept}
    country_of = {im["id"]: im.get("country", "unknown") for im in kept}

    cluster_of: dict[int, int] = {}
    for ci, members in enumerate(clusters):
        for i in members:
            if i in eligible:
                cluster_of[i] = ci

    groups_map: dict[tuple[str, object], list[int]] = collections.defaultdict(list)
    for img_id in eligible:
        key = cluster_of.get(img_id, ("solo", img_id))
        groups_map[(country_of[img_id], key)].append(img_id)

    by_country: dict[str, list] = collections.defaultdict(list)
    for (country, key), ids in groups_map.items():
        by_country[country].append((str(key), ids))

    multi = sum(1 for v in groups_map.values() if len(v) > 1)
    print(f"{len(groups_map)} split groups across {len(by_country)} countries "
          f"({multi} are multi-image near-duplicate clusters held together)")

    # --- allocate whole segments, per country -------------------------------
    rng = random.Random(args.seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    per_country_rows = []

    for country in sorted(by_country):
        segs = sorted(by_country[country])
        rng.shuffle(segs)
        total = sum(len(ids) for _, ids in segs)
        target_val = total * args.val_fraction

        c_val: list[int] = []
        c_train: list[int] = []

        # Allocate whole segments to val only while doing so moves the running
        # total CLOSER to the target. A greedy "fill until exceeded" overshoots
        # by up to one whole segment, which on a small country is the difference
        # between a 15% and a 40% validation share.
        for idx, (_, ids) in enumerate(segs):
            remaining = len(segs) - idx
            # Reserve at least one segment for train when more than one exists.
            must_stop = (remaining <= 1 and c_train == [] and len(segs) > 1)
            closer = abs(len(c_val) + len(ids) - target_val) < abs(len(c_val) - target_val)
            if closer and not must_stop:
                c_val.extend(ids)
            else:
                c_train.extend(ids)

        # A country with a single segment cannot be split; put it in train and
        # say so, rather than silently handing the whole country to val.
        if not c_train and c_val:
            c_train, c_val = c_val, []
            print(f"  note: {country} has only {len(segs)} group(s); "
                  f"assigned entirely to train")
        elif not c_val and len(segs) > 1:
            print(f"  note: {country} contributed no validation images "
                  f"(only {len(segs)} segments)")

        train_ids.extend(c_train)
        val_ids.extend(c_val)
        per_country_rows.append((country, len(segs), len(c_train), len(c_val),
                                 100 * len(c_val) / max(total, 1)))

    # --- per-class counts, so imbalance is visible up front -----------------
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    split_of = {i: "train" for i in train_ids}
    split_of.update({i: "val" for i in val_ids})
    class_counts = {"train": collections.Counter(), "val": collections.Counter()}
    for a in coco["annotations"]:
        s = split_of.get(a["image_id"])
        if s:
            class_counts[s][cats[a["category_id"]]] += 1

    manifest = {
        "source_coco": str(args.coco),
        "dedup_report": str(args.dedup) if args.dedup else None,
        "seed": args.seed,
        "n_groups": len(groups_map),
        "n_duplicate_clusters": multi,
        "dropped_duplicates": bool(args.drop_duplicates),
        "val_fraction": args.val_fraction,
        "excluded_view": args.exclude_view,
        "grouping": "country + perceptual-hash near-duplicate cluster; whole "
                    "clusters assigned to one split. Sequence numbers are NOT "
                    "used: rdd.verify.sequence_locality showed RDD2022's "
                    "numbering carries no locality.",
        "splits": {
            "train": {"count": len(train_ids), "sha256": sha256_of_ids(train_ids),
                      "image_ids": sorted(train_ids)},
            "val": {"count": len(val_ids), "sha256": sha256_of_ids(val_ids),
                    "image_ids": sorted(val_ids)},
        },
        "class_counts": {k: dict(v) for k, v in class_counts.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- report --------------------------------------------------------------
    print("\nper country")
    print(f"  {'country':<18}{'groups':>8}{'train':>9}{'val':>8}{'val %':>8}")
    for country, nseg, ntr, nva, pct in per_country_rows:
        print(f"  {country:<18}{nseg:>8}{ntr:>9}{nva:>8}{pct:>7.1f}%")

    print("\nper class")
    print(f"  {'class':<26}{'train':>9}{'val':>8}{'val %':>8}")
    for name in cats.values():
        tr = class_counts['train'][name]
        va = class_counts['val'][name]
        tot = tr + va
        print(f"  {name:<26}{tr:>9}{va:>8}{(100*va/tot if tot else 0):>7.1f}%")

    print("\n" + "=" * 62)
    print(f"  train {len(train_ids):>7} images   sha256 {manifest['splits']['train']['sha256'][:16]}...")
    print(f"  val   {len(val_ids):>7} images   sha256 {manifest['splits']['val']['sha256'][:16]}...")
    print(f"\nwrote {args.out}")
    print("Cite these hashes in every results table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
