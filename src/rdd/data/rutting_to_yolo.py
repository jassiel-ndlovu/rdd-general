"""Prepare the Sekilab road rutting dataset (US21) for detection and segmentation.

The corpus is Saha, Arya, Kumar, Maeda and Sekimoto's 949-image rutting set
(Mendeley Data, CC BY 4.0, DOI 10.17632/v8mbnvx8pv.1), the dataset C7 was
waiting on.

It arrives already annotated for both tasks:

    images/                                     949 (675 .jpg + 274 .png)
    labels/object_detection_yolo_format/        949 .txt, already normalised YOLO
    labels/semantic_segmentation_coco_json/     labels_coco.json
    labels/semantic_segmentation_brush_labels/  .npy and .png masks

Three decisions matter for the result.

The output keeps the project's full seven-name taxonomy with only C7 populated,
rather than relabelling rutting as a lone class 0. Ids are frozen in
configs/classmap.yaml, and a model whose class 0 means rutting while every
other model's class 0 means longitudinal crack invites mistakes in a results
table. The cost is six output channels that never receive a gradient.

The split is contiguous rather than random. The images are numbered
sequentially, and sequential road imagery is usually consecutive frames of the
same road, so a random split would place near-duplicate frames on both sides.
This is the same leakage the project avoids in RDD2022 by splitting on country
and road segment. Both file lists are hashed, so the split is reproducible.

These are small crops rather than road photographs: masks run to about 259x194
and the images average 8 KB. Results from them are not comparable with the
640px RDD2022 figures.

Usage:
    python -m rdd.data.rutting_to_yolo \
        --src /opt/rdd/data/rutting-raw/RoadRuttingDatasetSekilab \
        --out /opt/rdd/data/yolo/rutting \
        --val-fraction 0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

CLASS_NAMES = [
    "C1_longitudinal_crack", "C2_transverse_crack", "C3_alligator_crack",
    "C4_pothole", "C5_white_line_blur", "C6_crosswalk_blur", "C7_rutting",
]
RUTTING_ID = 6
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _sha256_of_list(names: list[str]) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists():
        return
    if link:
        try:
            dst.symlink_to(src.resolve())
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dst)


def _coco_segments(coco_path: Path) -> tuple[dict[str, list[list[float]]], dict]:
    """image stem -> list of normalised polygons, from the COCO export."""
    doc = json.loads(coco_path.read_text(encoding="utf-8"))
    by_id = {im["id"]: im for im in doc.get("images", [])}
    out: dict[str, list[list[float]]] = {}
    stats = {"annotations": 0, "polygons": 0, "rle_skipped": 0,
             "degenerate": 0}
    for ann in doc.get("annotations", []):
        stats["annotations"] += 1
        im = by_id.get(ann.get("image_id"))
        if not im:
            continue
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            continue
        seg = ann.get("segmentation")
        # Brush labels exported to COCO are normally polygon lists. RLE is
        # possible, and silently dropping it would understate the masks, so it
        # is counted and reported; the .png masks are the fallback route.
        if isinstance(seg, dict) or not isinstance(seg, list):
            stats["rle_skipped"] += 1
            continue
        stem = Path(im["file_name"]).stem
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                stats["degenerate"] += 1
                continue
            norm = []
            for i in range(0, len(poly) - 1, 2):
                x = min(max(poly[i] / w, 0.0), 1.0)
                y = min(max(poly[i + 1] / h, 0.0), 1.0)
                norm.extend([x, y])
            if len(norm) >= 6:
                out.setdefault(stem, []).append(norm)
                stats["polygons"] += 1
            else:
                stats["degenerate"] += 1
    return out, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path,
                    help="the extracted RoadRuttingDatasetSekilab directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args(argv)

    src: Path = args.src
    img_src = src / "images"
    det_src = src / "labels" / "object_detection_yolo_format"
    coco = src / "labels" / "semantic_segmentation_coco_json" / "labels_coco.json"
    for p in (img_src, det_src):
        if not p.is_dir():
            raise SystemExit(f"expected {p} in the extracted dataset")

    images = sorted(p for p in img_src.iterdir()
                    if p.suffix.lower() in IMG_SUFFIXES)
    if not images:
        raise SystemExit(f"no images under {img_src}")

    # Contiguous, not shuffled, see the module docstring on frame leakage.
    n_val = max(1, int(round(len(images) * args.val_fraction)))
    train_imgs, val_imgs = images[:-n_val], images[-n_val:]

    segs, seg_stats = ({}, {})
    if coco.is_file():
        segs, seg_stats = _coco_segments(coco)
    else:
        seg_stats = {"note": f"no COCO export at {coco}"}

    report = {
        "source": "Saha, Arya, Kumar, Maeda & Sekimoto 2022, "
                  "RoadRuttingDatasetSekilab, Mendeley Data, CC BY 4.0, "
                  "doi:10.17632/v8mbnvx8pv.1",
        "images_total": len(images),
        "split": {"kind": "contiguous_block",
                  "why": "sequential numbering implies consecutive frames of "
                         "the same road; a random split would leak",
                  "train": len(train_imgs), "val": len(val_imgs),
                  "train_sha256": _sha256_of_list([p.name for p in train_imgs]),
                  "val_sha256": _sha256_of_list([p.name for p in val_imgs])},
        "segmentation": seg_stats,
        "tasks": {},
    }

    for task in ("detect", "segment"):
        if task == "segment" and not segs:
            report["tasks"][task] = {"skipped": "no usable polygons"}
            continue
        root = args.out / task
        counts = {"train": {"images": 0, "labelled": 0, "objects": 0},
                  "val": {"images": 0, "labelled": 0, "objects": 0}}
        for split, group in (("train", train_imgs), ("val", val_imgs)):
            idir = root / "images" / split
            ldir = root / "labels" / split
            idir.mkdir(parents=True, exist_ok=True)
            ldir.mkdir(parents=True, exist_ok=True)
            for ip in group:
                _place(ip, idir / ip.name, link=not args.copy)
                lines: list[str] = []
                if task == "detect":
                    sp = det_src / f"{ip.stem}.txt"
                    if sp.is_file():
                        for ln in sp.read_text(encoding="utf-8").splitlines():
                            parts = ln.split()
                            if len(parts) != 5:
                                continue
                            # The source labels everything class 0; remap onto
                            # the project's frozen id for rutting.
                            lines.append(" ".join([str(RUTTING_ID)] + parts[1:]))
                else:
                    for poly in segs.get(ip.stem, []):
                        lines.append(" ".join(
                            [str(RUTTING_ID)] + [f"{v:.6f}" for v in poly]))
                (ldir / f"{ip.stem}.txt").write_text(
                    ("\n".join(lines) + "\n") if lines else "",
                    encoding="utf-8")
                counts[split]["images"] += 1
                if lines:
                    counts[split]["labelled"] += 1
                    counts[split]["objects"] += len(lines)

        (root / "data.yaml").write_text(yaml.safe_dump({
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(CLASS_NAMES)},
            "nc": len(CLASS_NAMES),
        }, sort_keys=False), encoding="utf-8")
        report["tasks"][task] = counts

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preparation.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out / 'preparation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
