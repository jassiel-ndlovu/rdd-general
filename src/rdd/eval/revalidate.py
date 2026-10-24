"""Re-evaluate a trained checkpoint without retraining.

Produces the two things Tier A's run summaries never recorded:

  1. Per-class metrics *within each country*. The summaries hold per-class
     figures and per-country figures but never the two crossed, which is the
     single most useful missing number in the project -- it is what separates
     "potholes are hard" from "potholes are hard in Norway".
     (US13 task 13.5.)

  2. The pothole false negatives, with their boxes, so that failures can be
     categorised by size and country instead of inferred from a curve.
     (US14b.)

Both come out of one prediction pass, so they cost the same as either alone.

Inference only. Nothing here trains, and nothing here writes to the canonical
store or the splits.

Usage:
    rddpython -m rdd.eval.revalidate \
        --weights /opt/rdd/runs/tierA_base6_s0/weights/best.pt \
        --data /opt/rdd/data/yolo/core/data.yaml \
        --out  /opt/rdd/runs/tierA_base6_s0/reval \
        --fn-class C4_pothole

Why predictions are matched here rather than read off `model.val()`:
Ultralytics reports aggregate metrics, not which ground-truth box went
unmatched. The false-negative question needs the latter, so this module runs
`predict` at a low confidence floor and does its own IoU matching. The matching
is deliberately simple -- greedy, by descending confidence, within a class -- so
that the numbers it produces can be reasoned about. It is not a replacement for
the mAP computation, which still comes from `model.val()`.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Confidence floor for the prediction pass. Low enough to capture detections
# that the default threshold would discard, because the distinction between
# "scored too low to report" and "never detected at all" is the whole point of
# the false-negative breakdown.
CONF_FLOOR = 0.001

# The operating point the project reports at, so that FN counts can be given
# both at the reported threshold and at the floor.
REPORT_CONF = 0.25

IOU_MATCH = 0.5


def country_of(stem: str) -> str:
    """RDD2022 filenames are <Country>_<sequence>."""
    return stem.rsplit("_", 1)[0]


def load_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file as (class, x1, y1, x2, y2) in pixels.

    An absent file means a negative image, which is a legitimate state: the
    export writes empty label files for images with no annotations.
    """
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
        out.append((
            cls,
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        ))
    return out


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(gts: list[tuple], preds: list[tuple], min_conf: float) -> tuple[set[int], set[int]]:
    """Greedy per-class IoU matching.

    Returns (indices of matched ground truths, indices of used predictions).
    Predictions are consumed in descending confidence, each taking the best
    still-unmatched ground truth of its own class above the IoU threshold.
    """
    matched_gt: set[int] = set()
    used_pred: set[int] = set()
    ordered = sorted(
        (i for i, p in enumerate(preds) if p[5] >= min_conf),
        key=lambda i: -preds[i][5],
    )
    for pi in ordered:
        pcls = preds[pi][0]
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            if gi in matched_gt or g[0] != pcls:
                continue
            v = iou(g[1:5], preds[pi][1:5])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= IOU_MATCH:
            matched_gt.add(best_gi)
            used_pred.add(pi)
    return matched_gt, used_pred


def per_class_from(metrics, names: dict[int, str]) -> dict:
    """Pull the per-class block out of an Ultralytics metrics object.

    `ap_class_index` matters: a class with no instances in the subset under
    evaluation is omitted from the arrays rather than reported as zero, so the
    index is the only reliable way to attribute a figure to a class.
    """
    out = {}
    for i, _ in enumerate(metrics.box.ap50):
        ci = int(metrics.box.ap_class_index[i])
        out[names.get(ci, str(ci))] = {
            "AP50": round(float(metrics.box.ap50[i]), 4),
            "AP50_95": round(float(metrics.box.ap[i]), 4),
            "precision": round(float(metrics.box.p[i]), 4),
            "recall": round(float(metrics.box.r[i]), 4),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="data.yaml")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fn-class", default="C4_pothole",
                    help="class whose false negatives are dumped in full")
    ap.add_argument("--skip-fn", action="store_true",
                    help="metrics only; skip the prediction pass")
    args = ap.parse_args(argv)

    import yaml
    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    report: dict = {
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "iou_match": IOU_MATCH,
        "data_yaml_header": "".join(
            ln + "\n" for ln in args.data.read_text(encoding="utf-8").splitlines()
            if ln.startswith("#")
        ),
    }

    # ---- 1. pooled and per-class, over the whole validation set ------------
    print("== global validation ==")
    m = model.val(data=str(args.data), imgsz=args.imgsz, plots=False, verbose=False)
    report["pooled"] = {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }
    report["per_class"] = per_class_from(m, names)
    for c, v in report["per_class"].items():
        print(f"  {c:26s} AP50={v['AP50']:.4f} R={v['recall']:.4f}")

    # ---- 2. per-class within each country (US13 task 13.5) ----------------
    # Reuses the scratch-tree approach already proven in train_yolo.py rather
    # than inventing a second one.
    from rdd.train.train_yolo import per_country_eval

    print("\n== per-country validation ==")
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    val_img_dir = root / cfg["val"]
    val_lbl_dir = root / cfg["val"].replace("images", "labels")

    report["per_country"] = per_country_eval(model, args.data, args.out, args.imgsz)

    # The crossed block. per_country_eval builds one data.yaml per country and
    # leaves it behind, so re-validating against it is cheap and needs no
    # second scratch tree.
    crossed: dict = {}
    for country in sorted(report["per_country"]):
        cyaml = args.out / "_per_country" / country / "data.yaml"
        if not cyaml.exists():
            continue
        cm = model.val(data=str(cyaml), imgsz=args.imgsz, split="val",
                       verbose=False, plots=False)
        crossed[country] = per_class_from(cm, names)
        pot = crossed[country].get(args.fn_class)
        if pot:
            print(f"  {country:18s} {args.fn_class} AP50={pot['AP50']:.4f} R={pot['recall']:.4f}")
    report["per_class_by_country"] = crossed

    # ---- 3. false negatives for one class (US14b) -------------------------
    if not args.skip_fn:
        want = {v: k for k, v in names.items()}.get(args.fn_class)
        if want is None:
            print(f"\n  {args.fn_class} is not in this checkpoint's class list; skipping FN dump")
        else:
            print(f"\n== false negatives for {args.fn_class} ==")
            images = sorted(val_img_dir.glob("*.jpg"))
            fns: list[dict] = []
            tally = defaultdict(lambda: {"gt": 0, "fn_report": 0, "fn_floor": 0})

            for res in model.predict(
                [str(p) for p in images], imgsz=args.imgsz, conf=CONF_FLOOR,
                stream=True, verbose=False,
            ):
                stem = Path(res.path).stem
                h, w = res.orig_shape
                gts = load_labels(val_lbl_dir / f"{stem}.txt", w, h)
                preds = [
                    (int(c), *(float(v) for v in xyxy), float(cf))
                    for c, xyxy, cf in zip(
                        res.boxes.cls.tolist(),
                        res.boxes.xyxy.tolist(),
                        res.boxes.conf.tolist(),
                    )
                ]
                at_report, _ = match(gts, preds, REPORT_CONF)
                at_floor, _ = match(gts, preds, CONF_FLOOR)
                country = country_of(stem)

                for gi, g in enumerate(gts):
                    if g[0] != want:
                        continue
                    tally[country]["gt"] += 1
                    missed_report = gi not in at_report
                    missed_floor = gi not in at_floor
                    if missed_report:
                        tally[country]["fn_report"] += 1
                    if missed_floor:
                        tally[country]["fn_floor"] += 1
                    if missed_report:
                        bw, bh = g[3] - g[1], g[4] - g[2]
                        fns.append({
                            "image": stem,
                            "country": country,
                            "box_xyxy": [round(v, 1) for v in g[1:5]],
                            "box_w": round(bw, 1),
                            "box_h": round(bh, 1),
                            "box_area_px": round(bw * bh, 1),
                            "box_area_frac": round(bw * bh / (w * h), 6),
                            "image_w": w,
                            "image_h": h,
                            # The distinction that matters: a box the detector
                            # found but under-scored is a different problem
                            # from one it never produced at all.
                            "detected_at_floor": not missed_floor,
                        })

            report["false_negatives"] = {
                "class": args.fn_class,
                "report_conf": REPORT_CONF,
                "conf_floor": CONF_FLOOR,
                "by_country": {k: dict(v) for k, v in sorted(tally.items())},
                "count": len(fns),
            }
            (args.out / f"false_negatives_{args.fn_class}.json").write_text(
                json.dumps(fns, indent=1), encoding="utf-8")

            total_gt = sum(v["gt"] for v in tally.values())
            total_fr = sum(v["fn_report"] for v in tally.values())
            total_ff = sum(v["fn_floor"] for v in tally.values())
            if total_gt:
                print(f"  ground truth boxes          {total_gt}")
                print(f"  missed at conf {REPORT_CONF}         {total_fr}"
                      f"  ({total_fr / total_gt:.1%})")
                print(f"  missed at conf {CONF_FLOOR}        {total_ff}"
                      f"  ({total_ff / total_gt:.1%})  <- never detected at all")
                print(f"  under-scored rather than absent  {total_fr - total_ff}")

    out_json = args.out / "revalidation.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
