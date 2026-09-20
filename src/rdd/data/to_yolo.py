"""
Export the canonical COCO store to Ultralytics YOLO layout (M1 stage 6).

The COCO store stays the source of truth; this is a generated view of it. Never
hand-edit the output — regenerate it.

Two details that are easy to get wrong
--------------------------------------
1. **Normalise per image, not by a constant.** RDD2022 mixes 512x512, 600x600,
   640x640, 720x720 and 3650x2044. Dividing every box by a single assumed size
   silently corrupts five of the seven subsets.
2. **Write an empty ``.txt`` for images with no objects.** Ultralytics treats a
   missing label file as "unlabelled" and an empty one as "no objects here".
   The 33% of RDD2022 training images that contain zero damage are genuine
   negatives and must be presented as such.

Images are symlinked where the platform allows it and copied otherwise, so a
14 GB dataset is not duplicated on disk.

Usage
-----
    python -m rdd.data.to_yolo \\
        --coco   /opt/rdd/data/coco/rdd2022_train.json \\
        --splits /opt/rdd/data/splits/rdd2022_core.json \\
        --root   /opt/rdd/data/rdd2022/RDD2022 \\
        --out    /opt/rdd/data/yolo/core
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)
        return "copy"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="COCO -> Ultralytics YOLO export.")
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path, help="image root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    ap.add_argument("--keep-empty-classes", action="store_true",
                    help="keep classes that have zero annotations in this split "
                         "(default: drop them -- see note below)")
    ap.add_argument("--only-classes", default=None,
                    help="comma-separated class names to export, e.g. "
                         "'C1_longitudinal_crack,C2_transverse_crack,"
                         "C3_alligator_crack,C4_pothole'. Used to build the "
                         "four-class control (O5) from the SAME images and the "
                         "SAME splits as the full baseline, so the only variable "
                         "is the label set.")
    args = ap.parse_args(argv)

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    manifest = json.loads(args.splits.read_text(encoding="utf-8"))

    by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    # A class with zero instances cannot be predicted, so its AP is 0 by
    # construction and it drags the mean down for no informational reason.
    # C7_rutting is in the taxonomy but has no RDD2022 data, so by default we
    # export only the classes actually present and record what was dropped.
    in_split = set()
    for split in manifest["splits"].values():
        in_split.update(split["image_ids"])
    present = collections.Counter(
        a["category_id"] for a in coco["annotations"] if a["image_id"] in in_split)

    all_cats = sorted(coco["categories"], key=lambda c: c["id"])

    if args.only_classes:
        wanted = {n.strip() for n in args.only_classes.split(",") if n.strip()}
        unknown = wanted - {c["name"] for c in all_cats}
        if unknown:
            print(f"--only-classes names not in the taxonomy: {sorted(unknown)}",
                  file=sys.stderr)
            return 2
        all_cats = [c for c in all_cats if c["name"] in wanted]

    if args.keep_empty_classes:
        cats, dropped_cats = all_cats, []
    else:
        cats = [c for c in all_cats if present.get(c["id"], 0) > 0]
        dropped_cats = [c["name"] for c in all_cats if present.get(c["id"], 0) == 0]

    # Ultralytics wants contiguous class ids starting at 0, in a fixed order.
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    mode_counts: collections.Counter = collections.Counter()
    stats = {}

    for split_name, split in manifest["splits"].items():
        img_dir = args.out / "images" / split_name
        lbl_dir = args.out / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_img = n_box = n_empty = n_skip = 0
        clipped = 0

        for img_id in split["image_ids"]:
            im = by_id.get(img_id)
            if im is None:
                n_skip += 1
                continue
            src = args.root / im["rel_path"]
            if not src.exists():
                n_skip += 1
                continue

            stem = Path(im["file_name"]).stem
            dst_img = img_dir / im["file_name"]
            if args.copy:
                if not dst_img.exists():
                    shutil.copy2(src, dst_img)
                mode_counts["copy"] += 1
            else:
                mode_counts[link_or_copy(src, dst_img)] += 1

            W, H = im.get("width") or 0, im.get("height") or 0
            lines = []
            for a in anns_by_image.get(img_id, []):
                # With --only-classes, annotations of excluded classes are
                # present in the store but must not be written. The image is
                # still kept: it becomes a negative for the exported classes.
                if a["category_id"] not in cat_index:
                    continue
                x, y, w, h = a["bbox"]
                if not W or not H:
                    continue
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                nw, nh = w / W, h / H
                # Clamp into [0,1]; a handful of source boxes exceed the frame.
                before = (cx, cy, nw, nh)
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
                if (cx, cy, nw, nh) != before:
                    clipped += 1
                if nw <= 0 or nh <= 0:
                    continue
                lines.append(f"{cat_index[a['category_id']]} "
                             f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_box += 1

            # Empty file, deliberately: this image is a negative, not unlabelled.
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            if not lines:
                n_empty += 1
            n_img += 1

        stats[split_name] = dict(images=n_img, boxes=n_box, empty=n_empty,
                                 missing=n_skip, clipped=clipped)

    data_yaml = args.out / "data.yaml"
    data_yaml.write_text(
        "# Generated by rdd.data.to_yolo -- do not edit by hand.\n"
        f"# source COCO : {args.coco}\n"
        f"# split manifest: {args.splits}\n"
        f"# train sha256 : {manifest['splits']['train']['sha256']}\n"
        f"# val   sha256 : {manifest['splits']['val']['sha256']}\n"
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        + (f"# excluded (zero instances in this split): {dropped_cats}\n"
           if dropped_cats else "")
        + f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8")

    print("=" * 66)
    print(f"{'split':<10}{'images':>9}{'boxes':>9}{'empty':>8}{'missing':>9}{'clipped':>9}")
    for s, st in stats.items():
        print(f"{s:<10}{st['images']:>9}{st['boxes']:>9}{st['empty']:>8}"
              f"{st['missing']:>9}{st['clipped']:>9}")
    print("=" * 66)
    print(f"image placement: {dict(mode_counts)}")
    print(f"classes ({len(names)}): {names}")
    if dropped_cats:
        print(f"EXCLUDED (zero instances -- would only deflate mAP): {dropped_cats}")
        print("  pass --keep-empty-classes to include them anyway")
    print(f"\nwrote {data_yaml}")
    if any(st["missing"] for st in stats.values()):
        print("\nWARNING: some images referenced by the split were not on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
