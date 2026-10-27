"""Convert the Stellenbosch pothole set (Supervisely export) to a YOLO val set.

The corpus is Nienaber, Kroon and Booysen's pothole set, released by
Stellenbosch University in 2018 under CC0 and redistributed by Dataset Ninja.

Three properties shape the conversion. It annotates potholes and nothing else,
so the output keeps the project's frozen class names while the labels hold only
C4. Average precision is per class, so C4 is unaffected by predictions for the
other five, which are counted separately as an observation rather than as
error. About 74% of the 11,497 images carry no annotation; these are deliberate
negatives in the original corpus, which makes a false-positive rate computable,
and --sample-negatives writes a list so that can be verified first. Finally it
is an evaluation set only, so the generated data.yaml points its train key at
an empty directory rather than at any South African image.

Usage:
    python -m rdd.data.stellenbosch_to_yolo \
        --src ~/dataset-ninja/road-pothole-images \
        --out /opt/rdd/data/yolo/stellenbosch \
        --sample-negatives 40
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

# Frozen in configs/classmap.yaml. Renumbering invalidates every checkpoint and
# every results table, so these are read as constants, not guessed.
CLASS_NAMES = [
    "C1_longitudinal_crack", "C2_transverse_crack", "C3_alligator_crack",
    "C4_pothole", "C5_white_line_blur", "C6_crosswalk_blur",
]
POTHOLE_ID = 3
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}

# The source vocabulary, as declared for `stellenbosch_potholes` in the class
# map. Anything else present in the export is a reason to stop, not to guess.
ACCEPTED = {"pothole": POTHOLE_ID}


def convert(src: Path, out: Path, link: bool, sample_negatives: int) -> dict:
    ann_files = sorted(src.rglob("ann/*.json"))
    if not ann_files:
        raise SystemExit(f"no Supervisely annotations found under {src}")

    img_dir = out / "images" / "val"
    lbl_dir = out / "labels" / "val"
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "with_boxes": 0, "negatives": 0, "boxes": 0,
             "skipped_missing_image": 0, "degenerate_boxes": 0}
    unknown: dict[str, int] = {}
    negatives: list[str] = []

    for ap in ann_files:
        img_src = ap.parent.parent / "img" / ap.name[:-5]   # drop ".json"
        if not img_src.exists():
            # Supervisely sometimes stores the image name with its own suffix;
            # fall back to a stem match before giving up on the pair.
            cands = list((ap.parent.parent / "img").glob(ap.stem + ".*"))
            if not cands:
                stats["skipped_missing_image"] += 1
                continue
            img_src = cands[0]

        meta = json.loads(ap.read_text(encoding="utf-8"))
        h = meta.get("size", {}).get("height")
        w = meta.get("size", {}).get("width")
        if not h or not w:
            stats["skipped_missing_image"] += 1
            continue

        lines = []
        for obj in meta.get("objects", []):
            title = obj.get("classTitle", "")
            if title not in ACCEPTED:
                unknown[title] = unknown.get(title, 0) + 1
                continue
            if obj.get("geometryType") not in (None, "rectangle"):
                unknown[f"{title}:{obj.get('geometryType')}"] = \
                    unknown.get(f"{title}:{obj.get('geometryType')}", 0) + 1
                continue
            ext = obj.get("points", {}).get("exterior") or []
            if len(ext) < 2:
                stats["degenerate_boxes"] += 1
                continue
            xs = [p[0] for p in ext]
            ys = [p[1] for p in ext]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                stats["degenerate_boxes"] += 1
                continue
            lines.append(
                f"{ACCEPTED[title]} {((x1 + x2) / 2) / w:.6f} "
                f"{((y1 + y2) / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            )

        # The archive holds FOUR Supervisely datasets, ds1_simplex-train,
        # ds1_simplex-test, ds2_complex-train, ds2_complex-test, and image
        # basenames repeat across them: 11,497 files share only 8,563 distinct
        # names. Writing them into one flat directory silently overwrote 2,935
        # images and paired some of the survivors with another copy's labels.
        # Prefixing with the source dataset makes every name unique, and keeps
        # the simplex/complex distinction legible for anyone who wants to
        # report those subsets separately.
        ds_name = ap.parent.parent.name
        stem = f"{ds_name}__{img_src.stem}"
        dst_img = img_dir / f"{stem}{img_src.suffix}"
        if not dst_img.exists():
            if link:
                try:
                    dst_img.symlink_to(img_src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(img_src, dst_img)
            else:
                shutil.copy2(img_src, dst_img)

        # An empty label file is meaningful: it declares a negative image
        # rather than an unlabelled one, which is what makes the
        # false-positive rate computable.
        (lbl_dir / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

        stats["images"] += 1
        if lines:
            stats["with_boxes"] += 1
            stats["boxes"] += len(lines)
        else:
            stats["negatives"] += 1
            if len(negatives) < sample_negatives:
                negatives.append(img_src.name)

    if unknown:
        raise SystemExit(
            "the export contains labels this converter was not told about: "
            f"{unknown}. Add them to configs/classmap.yaml under "
            "`stellenbosch_potholes` (or map them to __IGNORE__) rather than "
            "letting them be dropped silently."
        )

    # Validation only, on purpose: US22 requires that no South African image is
    # ever trained on. Omitting the train key altogether was the first attempt
    # at enforcing that, and it does not work, ultralytics' own
    # check_det_dataset raises "'train' and 'val' are required in all data
    # YAMLs" for validation too, so the guard blocked the very thing it was
    # meant to protect.
    #
    # So train points at a real but deliberately empty directory. The schema is
    # satisfied, validation runs, and anything that tries to train on this set
    # finds no images and fails loudly. The name says so in case anyone reads
    # the YAML rather than this comment.
    empty = out / "images" / "EMPTY_do_not_train_on_south_african_data"
    empty.mkdir(parents=True, exist_ok=True)
    (empty / "README.txt").write_text(
        "Intentionally empty. US22 evaluates on South African imagery and "
        "never trains on it; this directory exists only because ultralytics "
        "requires a train key in every data.yaml.\n", encoding="utf-8")
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out),
        "train": "images/EMPTY_do_not_train_on_south_african_data",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }, sort_keys=False), encoding="utf-8")

    # Count what actually landed on disk. conversion.json previously reported
    # the number of annotations READ, which matched the publisher exactly while
    # the output directory held 2,935 fewer images than that, a partial
    # dataset that would have produced a confident AP figure over an unknown
    # subset. Reading and writing are now reconciled explicitly.
    written_imgs = sum(1 for p in img_dir.iterdir()
                       if p.suffix.lower() in IMG_SUFFIXES)
    written_lbls = sum(1 for _ in lbl_dir.glob("*.txt"))
    stats["written_images"] = written_imgs
    stats["written_labels"] = written_lbls
    if written_imgs != stats["images"] or written_lbls != stats["images"]:
        raise SystemExit(
            f"processed {stats['images']} annotations but wrote "
            f"{written_imgs} images and {written_lbls} labels. Names are "
            f"colliding, so some images would be evaluated against another "
            f"image's labels. Refusing to produce a dataset that would give a "
            f"confident number over an unknown subset."
        )

    stats["negative_sample"] = negatives
    (out / "conversion.json").write_text(json.dumps(stats, indent=2),
                                         encoding="utf-8")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path,
                    help="the downloaded Supervisely export")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking them")
    ap.add_argument("--sample-negatives", type=int, default=40,
                    help="how many unannotated image names to record, so the "
                         "assumption that they are true negatives can be "
                         "checked by eye before the number is used")
    args = ap.parse_args(argv)

    s = convert(args.src, args.out, link=not args.copy,
                sample_negatives=args.sample_negatives)
    print(f"images            {s['images']}")
    print(f"  with potholes   {s['with_boxes']}  ({s['boxes']} boxes)")
    print(f"  negatives       {s['negatives']}")
    print(f"  no image found  {s['skipped_missing_image']}")
    print(f"  degenerate      {s['degenerate_boxes']}")
    print(f"wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
