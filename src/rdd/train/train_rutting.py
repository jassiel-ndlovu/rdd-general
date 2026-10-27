"""Train C7 rutting detection or segmentation on the Sekilab rutting set (US21).

Separate from rdd.train.train_yolo rather than a flag on it: that module has no
segmentation path and derives country from the filename, which this corpus has
no notion of.

The corpus is 949 small crops, 503 of which carry an annotation, with a
contiguous split to avoid leaking consecutive frames. One seed at an image
scale well below the project's RDD2022 work, so this is a feasibility baseline
for the seventh class rather than a benchmark result.

Saha et al. (IEEE Big Data 2022) report YOLOX-s at 61.6 mAP@0.5 for detection
and PSPNet-ResNet50 at 54.69 IoU on this data. Their split is not published, so
these figures are indicative of the same task rather than a head-to-head.

Usage:
    python -m rdd.train.train_rutting --task detect \
        --data /opt/rdd/data/yolo/rutting/detect/data.yaml \
        --epochs 100 --imgsz 320 --name us21_rutting_det_s0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

RUTTING_ID = 6


def set_determinism(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _class_row(metrics, attr: str) -> dict:
    """Pull C7's row out of an ultralytics metrics object."""
    m = getattr(metrics, attr, None)
    if m is None:
        return {}
    try:
        idx = list(metrics.ap_class_index).index(RUTTING_ID)
    except (ValueError, AttributeError):
        return {"note": "C7 absent from the validation result"}
    try:
        return {"AP50": round(float(m.ap50[idx]), 4),
                "AP50_95": round(float(m.ap[idx]), 4),
                "precision": round(float(m.p[idx]), 4),
                "recall": round(float(m.r[idx]), 4)}
    except Exception:                                 # noqa: BLE001
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--task", choices=("detect", "segment"), required=True)
    ap.add_argument("--model", default=None,
                    help="defaults to yolo11s.pt or yolo11s-seg.pt by task")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=320,
                    help="320, not the project's usual 640: these are small "
                         "crops and 640 would upscale two to three times")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", type=Path, default=Path("/opt/rdd/runs"))
    ap.add_argument("--s3", default="")
    args = ap.parse_args(argv)

    set_determinism(args.seed)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    weights = args.model or (
        "yolo11s.pt" if args.task == "detect" else "yolo11s-seg.pt")

    print("=" * 66)
    print(f"  task       {args.task}")
    print(f"  model      {weights}")
    print(f"  data       {args.data}")
    print(f"  epochs     {args.epochs}   imgsz {args.imgsz}   batch {args.batch}")
    print(f"  seed       {args.seed}")
    print(f"  device     cuda={torch.cuda.is_available()}")
    print("=" * 66)

    # A stock checkpoint needs no custom modules, but registering is idempotent
    # and keeps this usable if a project variant is ever passed as --model.
    from rdd.models import register as register_custom_modules
    register_custom_modules()

    model = YOLO(weights)
    stats = {}
    try:
        stats = {"parameters": int(get_num_params(model.model)),
                 "gflops": round(float(get_flops(model.model, imgsz=args.imgsz)), 2)}
        print(f"  parameters {stats['parameters']:,}  GFLOPs {stats['gflops']}")
    except Exception as e:                            # noqa: BLE001
        print(f"  [warn] could not read model info: {e}")

    out_dir = args.project / args.name
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, seed=args.seed, patience=args.patience,
        workers=args.workers, project=str(args.project), name=args.name,
        exist_ok=True, deterministic=True, verbose=True, plots=False,
        # Orientation is definitional for cracks in this project, and rutting
        # is by definition a longitudinal wheel-path feature, so the same ban
        # on vertical flips and rotation applies here.
        flipud=0.0, degrees=0.0,
    )

    best = out_dir / "weights" / "best.pt"
    res = YOLO(str(best)).val(data=str(args.data), imgsz=args.imgsz,
                              batch=args.batch, verbose=False, plots=False)

    report = {
        "task": args.task,
        "weights_from": weights,
        "data": str(args.data),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "seed": args.seed,
        "model_stats": stats,
        "C7_box": _class_row(res, "box"),
        "source": "Saha, Arya, Kumar, Maeda & Sekimoto 2022, "
                  "RoadRuttingDatasetSekilab, Mendeley Data, CC BY 4.0",
        "reference_points": {
            "YOLOX-s mAP@0.5 (Saha et al. 2022)": 0.616,
            "PSPNet-ResNet50 IoU (Saha et al. 2022)": 0.5469,
            "note": "their train/test split is unpublished, so these are "
                    "indicative of the same task, not a head-to-head",
        },
    }
    if args.task == "segment":
        report["C7_mask"] = _class_row(res, "seg")

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2),
                                          encoding="utf-8")
    print("\n" + json.dumps({k: report[k] for k in
                             ("C7_box", "C7_mask") if k in report}, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")

    if args.s3:
        dest = f"{args.s3.rstrip('/')}/{args.name}"
        subprocess.run(["aws", "s3", "sync", str(out_dir), dest,
                        "--only-show-errors"], check=False)
        print(f"synced to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
