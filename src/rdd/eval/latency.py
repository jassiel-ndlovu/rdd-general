"""Measure inference latency per US17 variant.

US17 asks for parameters, FLOPs and latency per architecture. The trainer
records the first two; this is the third.

Two numbers per variant, because they do not move together here: forward, one
batch through the network on the GPU; and end to end, what predict takes on a
real image including letterboxing, device transfer, decoding and NMS. A variant
can be modest in FLOPs and still cost more to deploy, which a FLOP count hides.

Measured on an idle GPU, since a box training something else reports contention
rather than the model, so this refuses to run if another process holds the
device. Timing uses torch.cuda.synchronize around each iteration because CUDA
launches are asynchronous. cudnn.benchmark is left on and warmup iterations are
discarded to pay for kernel selection. Median and p95 sit beside the mean,
since latency distributions are right-skewed.

Usage:
    python -m rdd.eval.latency \
        --variant baseline=/opt/rdd/runs/tierA_base6_s0/weights/best.pt \
        --variant us17a=/opt/rdd/runs/us17a_p2head_s0/weights/best.pt \
        --images /opt/rdd/data/yolo/core/images/val \
        --out /opt/rdd/runs/us17_latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _busy_gpu_processes() -> list[str]:
    """Other processes holding the GPU, which would corrupt the measurement."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:                                 # noqa: BLE001
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _timed(fn, iters: int, warmup: int, sync) -> dict:
    """Run fn repeatedly and summarise the per-call wall time in ms."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "mean_ms": round(statistics.fmean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
        "std_ms": round(statistics.pstdev(samples), 3) if len(samples) > 1 else 0.0,
        "min_ms": round(samples[0], 3),
        "fps": round(1000.0 / statistics.fmean(samples), 1),
        "iters": iters,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH",
                    required=True, help="repeatable; a trained .pt to measure")
    ap.add_argument("--images", type=Path, default=None,
                    help="directory of real images for the end-to-end timing. "
                         "Without it only the forward pass is measured")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1,
                    help="1 is the deployment-relevant case and the one US17 "
                         "reports; larger batches measure throughput instead")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--half", action="store_true",
                    help="measure in FP16. Off by default: FP32 is the cleaner "
                         "architecture comparison, since FP16 speedups depend "
                         "on which kernels happen to have tensor-core paths")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="measure even with another process on the GPU. The "
                         "numbers then describe contention, not the model")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch
    # Custom modules must be resolvable before a checkpoint holding one is
    # unpickled, not only before a YAML is parsed. See rdd.models.register.
    from rdd.models import register as register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    if not torch.cuda.is_available():
        print("no CUDA device; latency on CPU would not be comparable to the "
              "numbers US17 reports")
        return 2

    busy = _busy_gpu_processes()
    if busy and not args.allow_busy_gpu:
        print("GPU is in use, so a latency measurement here would describe "
              "contention rather than the model:")
        for b in busy:
            print(f"  {b}")
        print("Wait for the device to be idle, or pass --allow-busy-gpu "
              "knowing the numbers are not comparable.")
        return 3

    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda:0")
    sync = torch.cuda.synchronize
    gpu = torch.cuda.get_device_name(0)

    imgs: list[Path] = []
    if args.images:
        imgs = sorted(p for p in Path(args.images).iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not imgs:
            print(f"no images under {args.images}")
            return 1

    results: dict[str, dict] = {}
    for spec in args.variant:
        name, _, path = spec.partition("=")
        print(f"\n=== {name}  ({path})")
        model = YOLO(path)
        net = model.model.to(dev).eval()
        if args.half:
            net = net.half()
        n_p = get_num_params(net)
        flops = get_flops(net, imgsz=args.imgsz)

        x = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=dev,
                        dtype=torch.half if args.half else torch.float)

        with torch.inference_mode():
            fwd = _timed(lambda: net(x), args.iters, args.warmup, sync)
        print(f"  params {n_p:,}   GFLOPs {flops:.1f}")
        print(f"  forward      {fwd['mean_ms']:7.2f} ms mean   "
              f"{fwd['median_ms']:7.2f} median   {fwd['p95_ms']:7.2f} p95   "
              f"{fwd['fps']:6.1f} fps")

        entry = {"weights": str(path), "parameters": int(n_p),
                 "gflops": round(float(flops), 2), "forward": fwd}

        if imgs:
            # Cycle through real images so the NMS cost reflects real detection
            # counts. A single image would fix the candidate count and flatter
            # whichever variant happens to predict least on that one image.
            n = min(len(imgs), args.iters)
            picks = [str(imgs[i % len(imgs)]) for i in range(n)]
            state = {"i": 0}

            def one_predict():
                p = picks[state["i"] % n]
                state["i"] += 1
                model.predict(p, imgsz=args.imgsz, device=0, verbose=False,
                              half=args.half)

            e2e = _timed(one_predict, n, min(args.warmup, n), sync)
            print(f"  end to end   {e2e['mean_ms']:7.2f} ms mean   "
                  f"{e2e['median_ms']:7.2f} median   {e2e['p95_ms']:7.2f} p95   "
                  f"{e2e['fps']:6.1f} fps   (over {n} real images)")
            entry["end_to_end"] = e2e
            entry["end_to_end_images"] = n

        results[name] = entry
        del model, net, x
        torch.cuda.empty_cache()

    names = list(results)
    base = results.get("baseline")
    print("\n" + "=" * 78)
    print(f"US17 latency: {gpu}, imgsz {args.imgsz}, batch {args.batch}, "
          f"{'FP16' if args.half else 'FP32'}")
    print("=" * 78)
    print(f"{'variant':12} {'GFLOPs':>8} {'fwd ms':>9} {'e2e ms':>9} "
          f"{'e2e fps':>8} {'vs base':>9}")
    for nm in names:
        r = results[nm]
        e2e = r.get("end_to_end", {})
        rel = ""
        if base and base.get("end_to_end") and e2e:
            rel = f"x{e2e['mean_ms'] / base['end_to_end']['mean_ms']:.2f}"
        print(f"{nm:12} {r['gflops']:8.1f} {r['forward']['mean_ms']:9.2f} "
              f"{e2e.get('mean_ms', float('nan')):9.2f} "
              f"{e2e.get('fps', float('nan')):8.1f} {rel:>9}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"gpu": gpu, "imgsz": args.imgsz, "batch": args.batch,
             "precision": "fp16" if args.half else "fp32",
             "iters": args.iters, "warmup": args.warmup,
             "variants": results}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
