"""
Tier A baseline: one-stage CNN detector on the RDD2022 core taxonomy (M2).

Follows the proposal's protocol:
  * one-stage CNN (YOLO family, small variant), COCO-initialised
  * 640 px, 100-150 epochs
  * three seeds, mean +/- std reported
  * SINGLE MODEL, no ensembling, no test-time augmentation
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
rotation turns a C1 into a C2 while leaving the label untouched, it teaches the
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
import gc
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
    duplicated images. It has happened once already, see infra/run-tierA.sh.
    """
    if not remote:
        return
    subprocess.run(
        ["aws", "s3", "sync", str(local), remote, "--only-show-errors",
         "--exclude", "_per_country/*",
         "--exclude", "*/_per_country/*",
         "--exclude", "*.cache"],
        check=False)


def per_country_eval(model, data_yaml: Path, out_dir: Path, imgsz: int,
                     names: dict[int, str] | None = None) -> dict:
    """Re-evaluate the trained model on each country's validation images.

    Builds a temporary data.yaml per country whose val/ contains only that
    country's images, then runs validation. Inference-only, so cheap.

    Pass `names` to also record the per-class breakdown within each country.
    That block is what US13 task 13.5 needs, and it is captured here rather
    than by re-validating afterwards because each validation pass retains a
    few hundred megabytes: a second pass over six countries is what pushed a
    16 GB instance into the OOM killer.
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
            entry = {
                "n_images": len(paths),
                "mAP50": round(float(m.box.map50), 4),
                "mAP50_95": round(float(m.box.map), 4),
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
                "f1": round(float(2 * m.box.mp * m.box.mr / (m.box.mp + m.box.mr))
                            if (m.box.mp + m.box.mr) else 0.0, 4),
            }
            if names is not None:
                # ap_class_index matters: a class with no instances in this
                # country is omitted from the arrays rather than reported as
                # zero, so the index is the only safe attribution.
                per_class = {}
                for i, _ in enumerate(m.box.ap50):
                    ci = int(m.box.ap_class_index[i])
                    per_class[names.get(ci, str(ci))] = {
                        "AP50": round(float(m.box.ap50[i]), 4),
                        "AP50_95": round(float(m.box.ap[i]), 4),
                        "precision": round(float(m.box.p[i]), 4),
                        "recall": round(float(m.box.r[i]), 4),
                    }
                entry["per_class"] = per_class
            results[country] = entry
            # Each pass holds on to its own validator state. Release it before
            # the next country rather than at the end of the loop.
            del m
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                        # noqa: BLE001
                pass
        except Exception as e:                       # noqa: BLE001
            results[country] = {"n_images": len(paths), "error": str(e)[:200]}
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier A YOLO baseline.")
    ap.add_argument("--data", required=True, type=Path, help="data.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--transfer-from", default=None,
                    help="checkpoint to transfer matching weights from when "
                         "--model is an architecture .yaml. Without this a "
                         ".yaml builds random weights, and comparing that "
                         "against a COCO-pretrained baseline would confound "
                         "architecture with initialisation")
    ap.add_argument("--transfer-max-shift", type=int, default=6,
                    help="how far back transfer_weights may look for a "
                         "renumbered layer. Each inserted module shifts every "
                         "later layer by one, so a config combining several "
                         "modifications needs a larger value than the default "
                         "-- the combined US17 variant shifts Detect by 10. "
                         "Too small and the deepest layers silently fall back "
                         "to random initialisation")
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
                         "path, callbacks, S3 sync, per-country eval, summary "
                         "-- in minutes rather than hours. NEVER use <1.0 for a "
                         "reported result.")
    args = ap.parse_args(argv)

    if args.fraction < 1.0:
        print(f"\n  *** SMOKE TEST: fraction={args.fraction}, "
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
    print(f"  s3         {args.s3 or '(none, checkpoints are local only)'}")
    print("=" * 66)

    # Custom US17 modules must be resolvable before a YAML is parsed or a
    # checkpoint containing them is unpickled.
    from rdd.models import register as register_custom_modules
    register_custom_modules()

    model = YOLO(args.model)
    model_transfer = {}
    if args.transfer_from:
        # Transfers every layer whose shape matches and leaves the rest at
        # their initialised values, which for an added detection head is the
        # head itself. This is what keeps an architecture variant comparable
        # with a baseline that started from the same pretrained weights.
        if not str(args.model).endswith((".yaml", ".yml")):
            print(f"  [warn] --transfer-from with a non-yaml model "
                  f"({args.model}); weights would be overwritten, ignoring")
        else:
            # Not model.load(): it matches state-dict keys verbatim, and keys
            # carry the layer index, so inserting a module renumbers every
            # layer after it and roughly 40% of the pretrained weights are
            # silently dropped. See rdd.models.transfer_weights.
            from rdd.models import transfer_weights
            n, tot = transfer_weights(model, args.transfer_from,
                                      max_shift=args.transfer_max_shift)
            model_transfer = {"from": str(args.transfer_from),
                              "tensors": n, "of": tot,
                              "max_shift": args.transfer_max_shift}
            print(f"  source: {args.transfer_from}")

            # The transfer has to survive into the trainer, and by default it
            # does not. Model.train() reuses the module it is already holding
            # only when self.ckpt is set, which happens for a .pt and not for
            # a .yaml; otherwise the trainer falls back to the YAML path in
            # overrides, rebuilds from scratch, and every transferred tensor
            # is discarded. Measured on ultralytics 8.4.131: without the line
            # below the trainer's first conv does not match the pretrained
            # one, so the variant trains from random initialisation while the
            # Tier A baseline: launched from yolo11s.pt, did not. That is
            # the same confound transfer_weights exists to remove, one layer
            # further out. Setting ckpt puts a YAML-built model on the same
            # path a checkpoint-loaded one already takes; ultralytics then
            # reports its own "Transferred N/M items" line, which is the
            # number to trust.
            model.ckpt = {"model": model.model}

    # Parameters and FLOPs, which US17 asks for per variant.
    # Not model_info(): it early-returns None unless verbose=True, despite a
    # docstring promising a tuple. The primitives below actually return values.
    model_stats = {}
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        n_p = get_num_params(model.model)
        flops = get_flops(model.model, imgsz=args.imgsz)
        n_l = sum(1 for m in model.model.modules() if not list(m.children()))
        model_stats = {"layers": n_l, "parameters": int(n_p),
                       "gflops": round(float(flops), 2)}
        print(f"  layers {n_l}, parameters {n_p:,}, GFLOPs {flops:.1f}")
    except Exception as e:                            # noqa: BLE001
        print(f"  [warn] could not read model info: {e}")

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
        "transfer_from": args.transfer_from,
        "model_stats": model_stats,
        "weight_transfer": model_transfer,
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
