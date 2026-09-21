"""
Tier A baseline: one-stage CNN detector on the RDD2022 core taxonomy (M2).

Follows the proposal's protocol:
  * one-stage CNN (YOLO family, small variant), COCO-initialised
  * 640 px, 100-150 epochs
  * three seeds, mean +/- std reported
  * SINGLE MODEL -- no ensembling, no test-time augmentation
  * per-class AND per-country metrics, never one pooled number

Two things this script does that a stock `yolo train` call does not
-------------------------------------------------------------------
1. **Checkpoints to S3 after every epoch.** Required for Spot instances, which
   terminate on two minutes' notice. Without it, an interruption at hour 9 of a
   10-hour run loses the run.
2. **Per-country evaluation after training.** The proposal's H3 is about domain
   variance, and the CRDDC leaderboards show >0.25 F1 spread between countries
   inside a single model. A pooled number hides exactly the effect we are
   measuring.

Augmentation constraints (enforced here, not left to defaults)
--------------------------------------------------------------
Vertical flip and large rotation are DISABLED. "Longitudinal" and "transverse"
are defined by orientation relative to the direction of travel, so a 90-degree
rotation turns a C1 into a C2 while leaving the label untouched — it teaches the
model the opposite of the truth. Horizontal flip and colour jitter are safe and
are kept.

Usage
-----
    python -m rdd.train.train_yolo \\
        --data /opt/rdd/data/yolo/core/data.yaml \\
        --model yolo11s.pt --epochs 100 --seed 0 \\
        --name tierA_s0 --s3 s3://BUCKET/runs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def set_determinism(seed: int) -> None:
    """Seed every source of randomness we can reach.

    Framework-level `seed=` is necessary but not sufficient: cuDNN picks
    algorithms non-deterministically unless told otherwise, and Python's hash
    randomisation affects set/dict iteration order.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def s3_sync(local: Path, remote: str) -> None:
    """Push run artefacts to S3.

    The per-country evaluation builds a scratch tree of symlinked validation
    images inside the run directory. Those images already live in S3 via the
    country archives, so syncing them would re-upload tens of thousands of
    files per run for no benefit. Exclude the scratch tree and the dataset
    caches; keep weights, metrics, plots and the summary.

    Both `_per_country/*` and `*/_per_country/*` are needed. The AWS CLI matches
    these patterns against the path RELATIVE TO THE SYNC SOURCE, so which one
    fires depends on whether the source is the run directory or its parent.
    Getting this wrong is silent: the sync succeeds and simply uploads 1.7 GB of
    duplicated images. It has happened once already -- see infra/run-tierA.sh.
    """
    if not remote:
        return
    subprocess.run(
        ["aws", "s3", "sync", str(local), remote, "--only-show-errors",
         "--exclude", "_per_country/*",
         "--exclude", "*/_per_country/*",
         "--exclude", "*.cache"],
        check=False)


def per_country_eval(model, data_yaml: Path, out_dir: Path, imgsz: int) -> dict:
    """Re-evaluate the trained model on each country's validation images.

    Builds a temporary data.yaml per country whose val/ contains only that
    country's images, then runs validation. Inference-only, so cheap.
    """
    import yaml as _yaml
    cfg = _yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    val_img_dir = root / cfg["val"]
    val_lbl_dir = root / cfg["val"].replace("images", "labels")

    by_country: dict[str, list[Path]] = {}
    for p in sorted(val_img_dir.glob("*.jpg")):
        # Filenames are <Country>_<sequence>.jpg
        country = p.stem.rsplit("_", 1)[0]
        by_country.setdefault(country, []).append(p)

    results = {}
    scratch = out_dir / "_per_country"
    scratch.mkdir(parents=True, exist_ok=True)

    for country, paths in sorted(by_country.items()):
        if len(paths) < 10:
            results[country] = {"n_images": len(paths), "skipped": "too few images"}
            continue
        cdir = scratch / country
        (cdir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (cdir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        for p in paths:
            dst = cdir / "images" / "val" / p.name
            if not dst.exists():
                try:
                    os.symlink(p, dst)
                except OSError:
                    import shutil
                    shutil.copy2(p, dst)
            lab_src = val_lbl_dir / f"{p.stem}.txt"
            lab_dst = cdir / "labels" / "val" / f"{p.stem}.txt"
            if lab_src.exists() and not lab_dst.exists():
                try:
                    os.symlink(lab_src, lab_dst)
                except OSError:
                    import shutil
                    shutil.copy2(lab_src, lab_dst)

        ccfg = dict(cfg)
        ccfg["path"] = str(cdir.resolve())
        ccfg["train"] = "images/val"     # unused, but the key must exist
        ccfg["val"] = "images/val"
        cyaml = cdir / "data.yaml"
        cyaml.write_text(_yaml.safe_dump(ccfg), encoding="utf-8")

        try:
            m = model.val(data=str(cyaml), imgsz=imgsz, split="val",
                          verbose=False, plots=False)
            results[country] = {
                "n_images": len(paths),
                "mAP50": round(float(m.box.map50), 4),
                "mAP50_95": round(float(m.box.map), 4),
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
                "f1": round(float(2 * m.box.mp * m.box.mr / (m.box.mp + m.box.mr))
                            if (m.box.mp + m.box.mr) else 0.0, 4),
            }
        except Exception as e:                       # noqa: BLE001
            results[country] = {"n_images": len(paths), "error": str(e)[:200]}
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier A YOLO baseline.")
    ap.add_argument("--data", required=True, type=Path, help="data.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", type=Path, default=Path("/opt/rdd/runs"))
    ap.add_argument("--s3", default=os.environ.get("RDD_S3_RUNS", ""),
                    help="s3://bucket/prefix to sync checkpoints to each epoch")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="train on this fraction of the data. Use a small value "
                         "(e.g. 0.05) for a smoke test that exercises the whole "
                         "path -- callbacks, S3 sync, per-country eval, summary "
                         "-- in minutes rather than hours. NEVER use <1.0 for a "
                         "reported result.")
    args = ap.parse_args(argv)

    if args.fraction < 1.0:
        print(f"\n  *** SMOKE TEST: fraction={args.fraction} -- "
              f"results are NOT valid for reporting ***\n")

    run_name = args.name or f"{Path(args.model).stem}_s{args.seed}"
    set_determinism(args.seed)

    from ultralytics import YOLO
    import torch

    print("=" * 66)
    print(f"  model      {args.model}")
    print(f"  data       {args.data}")
    print(f"  epochs     {args.epochs}   imgsz {args.imgsz}   batch {args.batch}")
    print(f"  seed       {args.seed}  (deterministic cuDNN on)")
    print(f"  device     {args.device}  cuda={torch.cuda.is_available()}"
          f" {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"  run        {args.project / run_name}")
    print(f"  s3         {args.s3 or '(none -- checkpoints are local only)'}")
    print("=" * 66)

    model = YOLO(args.model)
    out_dir = args.project / run_name
    remote = f"{args.s3.rstrip('/')}/{run_name}" if args.s3 else ""

    # Push weights to S3 after each epoch so a Spot reclaim costs one epoch.
    def on_epoch_end(trainer):
        try:
            s3_sync(Path(trainer.save_dir), remote)
        except Exception as e:                        # noqa: BLE001
            print(f"  [warn] S3 sync failed: {e}")

    if remote:
        model.add_callback("on_fit_epoch_end", on_epoch_end)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        project=str(args.project),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        cache=False,          # RAM cache reintroduces non-determinism
        val=True,
        plots=True,
        # --- augmentation: orientation-safe only ---
        fliplr=0.5,
        flipud=0.0,           # MUST stay 0: would swap C1 <-> C2
        degrees=0.0,          # MUST stay 0: same reason
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        mosaic=1.0,
        close_mosaic=10,      # disable mosaic for the last 10 epochs
    )

    print("\nfinal validation (pooled)")
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, plots=False)

    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    per_class = {}
    try:
        for i, ap50 in enumerate(metrics.box.ap50):
            ci = int(metrics.box.ap_class_index[i])
            per_class[names.get(ci, str(ci))] = {
                "AP50": round(float(ap50), 4),
                "AP50_95": round(float(metrics.box.ap[i]), 4),
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
            }
    except Exception as e:                            # noqa: BLE001
        print(f"  [warn] could not extract per-class metrics: {e}")

    print("\nper-country validation")
    per_country = per_country_eval(model, args.data, out_dir, args.imgsz)
    for c, r in per_country.items():
        if "mAP50" in r:
            print(f"  {c:<18} n={r['n_images']:<6} mAP50={r['mAP50']:.3f}  "
                  f"F1={r['f1']:.3f}")
        else:
            print(f"  {c:<18} {r}")

    summary = {
        "run": run_name,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "fraction": args.fraction,
        "valid_for_reporting": args.fraction >= 1.0,
        "data": str(args.data),
        "data_yaml_header": args.data.read_text(encoding="utf-8").split("path:")[0],
        "pooled": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class": per_class,
        "per_country": per_country,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"  pooled mAP50 {summary['pooled']['mAP50']:.4f}   "
          f"mAP50-95 {summary['pooled']['mAP50_95']:.4f}")
    print(f"  wrote {out_dir / 'summary.json'}")
    s3_sync(out_dir, remote)
    if remote:
        print(f"  synced to {remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
