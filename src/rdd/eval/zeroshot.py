"""US22: zero-shot evaluation on held-out South African imagery.

Every other number in this project comes from RDD2022, whose train and
validation splits share countries, cameras and road authorities even though
they share no road segment. This puts the detector in front of a country,
camera and annotation team it has never seen: the Stellenbosch pothole set
(Nienaber, Kroon and Booysen, 2018, CC0).

The set annotates potholes and nothing else, so C4 is the only class with
ground truth. Average precision is per class, so C4 is unaffected by
predictions for the other five, and those are not errors either, since the
images contain cracks nobody annotated. They are counted as an observation.

Two numbers are produced: C4 detection quality on the annotated images, in the
same terms as every other result here so it can be set against the RDD2022
figure; and false positives per negative image, over the images the corpus
marks as containing no pothole. The second is what matters for deployment, and
nothing in the RDD2022 evaluation tests it.

Usage:
    python -m rdd.eval.zeroshot \
        --variant baseline=/opt/rdd/runs/tierA_base6_s0/weights/best.pt \
        --data /opt/rdd/data/yolo/stellenbosch/data.yaml \
        --fp-pass --out /opt/rdd/runs/us22_zeroshot.json
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
POTHOLE_ID = 3


def _c4_metrics(res) -> dict:
    """Pull C4's row out of an ultralytics val result."""
    out = {}
    try:
        idx = list(res.ap_class_index).index(POTHOLE_ID)
    except (ValueError, AttributeError):
        return {"note": "C4 absent from the validation result"}
    b = res.box
    out["AP50"] = round(float(b.ap50[idx]), 4)
    out["AP50_95"] = round(float(b.ap[idx]), 4)
    out["precision"] = round(float(b.p[idx]), 4)
    out["recall"] = round(float(b.r[idx]), 4)
    return out


def _negative_images(data_yaml: Path) -> list[Path]:
    """Images whose label file exists and is empty: the corpus's negatives."""
    import yaml
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(d.get("path", data_yaml.parent))
    img_dir = root / d["val"]
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/")
                   .replace("\\images\\", "\\labels\\"))
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lp = lbl_dir / f"{p.stem}.txt"
        if lp.exists() and not lp.read_text(encoding="utf-8").strip():
            out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", action="append", default=[], required=True,
                    metavar="NAME=PATH")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence for the false-positive pass. 0.25 is the "
                         "deployment default; the AP figures are unaffected by "
                         "it since AP sweeps all thresholds")
    ap.add_argument("--fp-pass", action="store_true",
                    help="also count predictions on the corpus's negative "
                         "images. Costs a second pass over them")
    ap.add_argument("--fp-limit", type=int, default=0,
                    help="cap the negative images used, 0 for all")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    # A checkpoint holding a custom module cannot be unpickled without this.
    from rdd.models import register as register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO

    negatives = _negative_images(args.data) if args.fp_pass else []
    if args.fp_pass:
        if args.fp_limit:
            negatives = negatives[:args.fp_limit]
        print(f"negative images for the false-positive pass: {len(negatives)}")

    results: dict[str, dict] = {}
    for spec in args.variant:
        name, _, path = spec.partition("=")
        print(f"\n=== {name}")
        model = YOLO(path)

        res = model.val(data=str(args.data), imgsz=args.imgsz,
                        batch=args.batch, verbose=False, plots=False)
        entry = {"weights": str(path), "C4": _c4_metrics(res)}
        c4 = entry["C4"]
        print(f"  C4 zero-shot   AP50 {c4.get('AP50')}  "
              f"AP50-95 {c4.get('AP50_95')}  "
              f"P {c4.get('precision')}  R {c4.get('recall')}")

        if negatives:
            # One path per call. Ultralytics sizes the predictor's batch from
            # the source: given a list it sets bs = len(source) and ignores
            # `batch` entirely, which is how this project exhausted a T4 once
            # already.
            per_class = {n: 0 for n in CLASS_NAMES}
            images_with_c4 = 0
            for i, p in enumerate(negatives, start=1):
                for r in model.predict(str(p), imgsz=args.imgsz,
                                       conf=args.conf, stream=True,
                                       verbose=False):
                    cls = [int(c) for c in r.boxes.cls.tolist()] if r.boxes is not None else []
                    for c in cls:
                        if 0 <= c < len(CLASS_NAMES):
                            per_class[CLASS_NAMES[c]] += 1
                    if POTHOLE_ID in cls:
                        images_with_c4 += 1
                if i % 500 == 0:
                    model.predictor = None
                    gc.collect()
                    print(f"    {i}/{len(negatives)} negatives")
            entry["negatives"] = {
                "images": len(negatives),
                "conf": args.conf,
                "predictions_per_class": per_class,
                "images_with_a_pothole_prediction": images_with_c4,
                "false_positive_rate_per_image":
                    round(per_class["C4_pothole"] / max(1, len(negatives)), 4),
                "images_with_a_pothole_prediction_frac":
                    round(images_with_c4 / max(1, len(negatives)), 4),
            }
            n = entry["negatives"]
            print(f"  on {n['images']} negative images at conf {args.conf}: "
                  f"{n['predictions_per_class']['C4_pothole']} pothole "
                  f"predictions over {n['images_with_a_pothole_prediction']} "
                  f"images ({n['images_with_a_pothole_prediction_frac']:.1%})")
            # Reported, not scored: these classes have no ground truth here.
            other = {k: v for k, v in n["predictions_per_class"].items()
                     if k != "C4_pothole" and v}
            if other:
                print(f"  other classes predicted (no ground truth for them "
                      f"in this corpus, so an observation only): {other}")

        results[name] = entry
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                             # noqa: BLE001
            pass

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "dataset": "stellenbosch_potholes",
            "source": "Nienaber, Kroon & Booysen 2018, Stellenbosch "
                       "University, CC0, via Dataset Ninja",
            "note": "C4 is the only annotated class; predictions for other "
                    "classes are observations, not errors",
            "imgsz": args.imgsz,
            "variants": results,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
