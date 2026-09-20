"""
Live Tier A progress reporter and completion-time estimator.

Runs ON the instance. Reads what Ultralytics writes to disk as it trains, so it
needs no cooperation from the training process and cannot disturb it.

Why not read cost from Cost Explorer
------------------------------------
AWS Cost Explorer lags roughly 24 hours, so it reports nothing useful about a
run in progress. Cost here is computed from **instance uptime multiplied by the
published hourly rate**, which is exact and immediate. It is reported as *gross*
(list price); while the promotional credit is live the net charge is zero.

Usage
-----
    python -m rdd.verify.progress --runs /opt/rdd/runs
    python -m rdd.verify.progress --runs /opt/rdd/runs --json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Tier A: 3 baseline seeds + 1 four-class control.
EXPECTED_RUNS = ["tierA_base6_s0", "tierA_base6_s1", "tierA_base6_s2", "tierA_ctrl4_s0"]
RATE_USD_PER_HOUR = 0.526          # g4dn.xlarge on-demand, us-east-1


def hms(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "--"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    return f"{h:02d}h {m:02d}m {s:02d}s"


def instance_uptime_seconds() -> float | None:
    """Seconds since this boot, from /proc/uptime."""
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except Exception:
        return None


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        u, mu, mt, t = [x.strip() for x in out.split(",")]
        return {"util_pct": int(u), "mem_used_mib": int(mu),
                "mem_total_mib": int(mt), "temp_c": int(t)}
    except Exception:
        return {}


def read_run(run_dir: Path) -> dict:
    """Parse one run's results.csv into progress + timing."""
    info = {"name": run_dir.name, "state": "unknown", "epochs_done": 0,
            "total_epochs": None, "sec_per_epoch": None, "elapsed_s": None,
            "eta_s": None, "best_map50": None, "last_map50": None}

    summary = run_dir / "summary.json"
    if summary.exists():
        info["state"] = "complete"
        try:
            d = json.loads(summary.read_text(encoding="utf-8"))
            info["total_epochs"] = d.get("epochs")
            info["epochs_done"] = d.get("epochs")
            info["best_map50"] = d.get("pooled", {}).get("mAP50")
            info["valid_for_reporting"] = d.get("valid_for_reporting", True)
        except Exception:
            pass
        return info

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        info["state"] = "starting"
        return info

    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    except Exception:
        return info
    if not rows:
        info["state"] = "starting"
        return info

    def col(row, *names):
        for n in names:
            for k in row:
                if k.strip() == n:
                    try:
                        return float(row[k])
                    except (TypeError, ValueError):
                        return None
        return None

    info["state"] = "running"
    info["epochs_done"] = len(rows)
    info["elapsed_s"] = col(rows[-1], "time")
    if info["elapsed_s"]:
        info["sec_per_epoch"] = info["elapsed_s"] / len(rows)

    maps = [col(r, "metrics/mAP50(B)") for r in rows]
    maps = [m for m in maps if m is not None]
    if maps:
        info["last_map50"] = round(maps[-1], 4)
        info["best_map50"] = round(max(maps), 4)

    # total epochs from the saved args
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        for line in args_yaml.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("epochs:"):
                try:
                    info["total_epochs"] = int(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
                break

    if info["total_epochs"] and info["sec_per_epoch"]:
        remaining = max(0, info["total_epochs"] - info["epochs_done"])
        info["eta_s"] = remaining * info["sec_per_epoch"]
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier A progress and ETA.")
    ap.add_argument("--runs", type=Path, default=Path("/opt/rdd/runs"))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--rate", type=float, default=RATE_USD_PER_HOUR)
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc)
    runs = {}
    for name in EXPECTED_RUNS:
        d = args.runs / name
        runs[name] = read_run(d) if d.exists() else {
            "name": name, "state": "pending", "epochs_done": 0,
            "total_epochs": None, "sec_per_epoch": None, "eta_s": None,
            "best_map50": None, "last_map50": None}

    # --- overall projection --------------------------------------------------
    rates = [r["sec_per_epoch"] for r in runs.values() if r.get("sec_per_epoch")]
    # Fall back to the measured rate if no run has produced timing yet.
    sec_per_epoch = sum(rates) / len(rates) if rates else 600.0
    default_epochs = next((r["total_epochs"] for r in runs.values()
                           if r.get("total_epochs")), 100)

    remaining_s = 0.0
    for r in runs.values():
        if r["state"] == "complete":
            continue
        total = r.get("total_epochs") or default_epochs
        remaining_s += max(0, total - r.get("epochs_done", 0)) * sec_per_epoch

    finish_at = now + timedelta(seconds=remaining_s)

    up = instance_uptime_seconds()
    cost_gross = (up / 3600.0 * args.rate) if up else None
    gpu = gpu_state()

    if args.json:
        print(json.dumps({
            "now_utc": now.isoformat(), "runs": runs,
            "sec_per_epoch": sec_per_epoch, "remaining_s": remaining_s,
            "eta_utc": finish_at.isoformat(),
            "uptime_s": up, "cost_gross_usd": cost_gross, "gpu": gpu}, indent=2))
        return 0

    done = sum(1 for r in runs.values() if r["state"] == "complete")
    tot_ep = sum((r.get("total_epochs") or default_epochs) for r in runs.values())
    don_ep = sum(r.get("epochs_done", 0) for r in runs.values())
    pct = 100 * don_ep / max(tot_ep, 1)

    bar_w = 42
    filled = int(bar_w * pct / 100)
    bar = "#" * filled + "." * (bar_w - filled)

    print("=" * 72)
    print(f"  TIER A PROGRESS          {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 72)
    print(f"  [{bar}] {pct:5.1f}%")
    print(f"  {don_ep} / {tot_ep} epochs   |   {done} / {len(runs)} runs complete")
    print()
    print(f"  {'run':<20}{'state':<10}{'epochs':>10}{'mAP50':>9}{'eta':>16}")
    print("  " + "-" * 66)
    for name in EXPECTED_RUNS:
        r = runs[name]
        ep = (f"{r.get('epochs_done', 0)}/{r.get('total_epochs') or default_epochs}")
        m = r.get("best_map50")
        mstr = f"{m:.4f}" if isinstance(m, float) else "--"
        eta = hms(r["eta_s"]) if r.get("eta_s") is not None else (
            "done" if r["state"] == "complete" else "--")
        print(f"  {name:<20}{r['state']:<10}{ep:>10}{mstr:>9}{eta:>16}")

    print()
    print(f"  pace                 {sec_per_epoch/60:.1f} min/epoch")
    print(f"  TIER A REMAINING     {hms(remaining_s)}")
    print(f"  ESTIMATED FINISH     {finish_at.strftime('%a %d %b %Y, %H:%M')} UTC"
          f"   ({(finish_at + timedelta(hours=2)).strftime('%H:%M')} SAST)")
    print()
    if up is not None:
        print(f"  instance uptime      {hms(up)}")
        print(f"  cost at list price   ${cost_gross:,.2f}"
              f"   (net $0.00 while the promotional credit is live)")
    if gpu:
        print(f"  gpu                  {gpu['util_pct']}% util, "
              f"{gpu['mem_used_mib']}/{gpu['mem_total_mib']} MiB, {gpu['temp_c']}C")
        if gpu["util_pct"] < 50:
            print("    ^ low utilisation: training may have stalled or finished")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
