"""Count what the six-class detector predicts on rutting imagery (US21).

The detector has no rutting class, so every detection it produces on these
images is either a genuine co-occurring defect, since rutted roads crack as
well, or a false positive triggered by a texture the taxonomy does not cover.

There is no six-class ground truth here, so the output is a distribution rather
than a score. It separates the weak form of the argument for broad category
coverage, that uncovered categories go undetected, from the strong form, that
they are actively mis-detected as a covered class.

Usage:
    python -m rdd.eval.distractor \
        --weights /opt/rdd/runs/tierA_base6_s0/weights/best.pt \
        --images /opt/rdd/data/yolo/rutting/detect/images/val \
        --out /opt/rdd/runs/us21_distractor.json
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

CLASS_NAMES = [
    "C1_longitudinal_crack", "C2_transverse_crack", "C3_alligator_crack",
    "C4_pothole", "C5_white_line_blur", "C6_crosswalk_blur",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="640, matching how the six-class model was trained "
                         "and would be deployed, not the rutting crops' own "
                         "scale")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="the deployment default; this is a count at an "
                         "operating point, not a threshold-swept metric")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from rdd.models import register as register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO

    imgs = sorted(p for p in args.images.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        raise SystemExit(f"no images under {args.images}")

    model = YOLO(str(args.weights))
    per_class = {n: 0 for n in CLASS_NAMES}
    images_with_any = 0
    conf_sum = {n: 0.0 for n in CLASS_NAMES}

    # One path per call: ultralytics sizes the predictor's batch from the
    # source and ignores `batch` for a list, which this project has already
    # run out of GPU memory discovering once.
    for i, p in enumerate(imgs, start=1):
        hit = False
        for r in model.predict(str(p), imgsz=args.imgsz, conf=args.conf,
                               stream=True, verbose=False):
            if r.boxes is None:
                continue
            for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                c = int(c)
                if 0 <= c < len(CLASS_NAMES):
                    per_class[CLASS_NAMES[c]] += 1
                    conf_sum[CLASS_NAMES[c]] += float(cf)
                    hit = True
        if hit:
            images_with_any += 1
        if i % 200 == 0:
            model.predictor = None
            gc.collect()
            print(f"  {i}/{len(imgs)}")

    total = sum(per_class.values())
    report = {
        "weights": str(args.weights),
        "images": len(imgs),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "detections_total": total,
        "images_with_any_detection": images_with_any,
        "images_with_any_detection_frac":
            round(images_with_any / len(imgs), 4),
        "detections_per_class": per_class,
        "mean_confidence_per_class": {
            k: (round(conf_sum[k] / per_class[k], 4) if per_class[k] else None)
            for k in CLASS_NAMES
        },
        "interpretation": "No six-class ground truth exists for this corpus, "
                          "so these are counts, not errors. Rutted roads do "
                          "crack, so some are genuine; the question the "
                          "distribution speaks to is whether an uncovered "
                          "category is actively mis-detected as a covered one.",
    }

    print(f"\nimages                     {len(imgs)}")
    print(f"images with a detection    {images_with_any} "
          f"({report['images_with_any_detection_frac']:.1%})")
    print(f"detections                 {total}")
    for k in CLASS_NAMES:
        if per_class[k]:
            print(f"  {k:24} {per_class[k]:5}  "
                  f"mean conf {report['mean_confidence_per_class'][k]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
