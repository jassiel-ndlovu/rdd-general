#!/bin/bash
# =============================================================================
#  US17: latency for every variant, measured on one idle GPU
# =============================================================================
#      bash infra/latency.sh
#
#  All variants are measured on one box. Latency depends on the architecture
#  rather than on training progress, so a variant still training elsewhere can
#  be measured from its current checkpoint. Measuring each on its own machine
#  would compare four GPUs in four thermal states.
#
#  Must run on an idle device. Both this script and rdd.eval.latency refuse
#  while a trainer is running.
#
#  Writes us17_latency.json beside the runs and pushes it to S3.
# =============================================================================
set -euo pipefail

RUNS=/opt/rdd/runs
DATA=/opt/rdd/data
WORK=/opt/rdd/latency
S3=s3://rdd-general-299284685700-us-east-1/runs
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

if pgrep -f 'rdd.train.train_yolo' > /dev/null; then
  echo "a trainer is running on this instance; latency measured here would" >&2
  echo "describe GPU contention rather than the models. Run it on an idle box." >&2
  exit 1
fi

# The idle watchdog halts a box with no GPU load, and a latency run is mostly
# short bursts. Hold it off for the duration and release it on the way out.
HELD=0
if [ ! -f "$KEEP" ]; then touch "$KEEP"; HELD=1; fi
trap '[ "$HELD" = 1 ] && rm -f "$KEEP"; echo "watchdog restored at $(date -Is)"' EXIT

mkdir -p "$WORK"
echo "== fetching checkpoints  $(date -Is)"
for r in tierA_base6_s0 us17a_p2head_s0 us17b_strip_s0 us17c_dysample_s0 us17d_combined_s0; do
  if [ ! -s "$WORK/$r.pt" ]; then
    aws s3 cp "$S3/$r/weights/best.pt" "$WORK/$r.pt" --only-show-errors \
      && echo "  ok   $r" \
      || echo "  skip $r (no checkpoint yet)"
  else
    echo "  have $r"
  fi
done

ARGS=()
[ -s "$WORK/tierA_base6_s0.pt" ]    && ARGS+=(--variant "baseline=$WORK/tierA_base6_s0.pt")
[ -s "$WORK/us17a_p2head_s0.pt" ]   && ARGS+=(--variant "us17a=$WORK/us17a_p2head_s0.pt")
[ -s "$WORK/us17b_strip_s0.pt" ]    && ARGS+=(--variant "us17b=$WORK/us17b_strip_s0.pt")
[ -s "$WORK/us17c_dysample_s0.pt" ] && ARGS+=(--variant "us17c=$WORK/us17c_dysample_s0.pt")
[ -s "$WORK/us17d_combined_s0.pt" ] && ARGS+=(--variant "us17d=$WORK/us17d_combined_s0.pt")

if [ "${#ARGS[@]}" -eq 0 ]; then
  echo "no checkpoints to measure" >&2
  exit 1
fi

echo
echo "== measuring  $(date -Is)"
rddpython -m rdd.eval.latency "${ARGS[@]}" \
  --images "$DATA/yolo/core/images/val" \
  --imgsz 640 --batch 1 --iters 100 --warmup 20 \
  --out "$RUNS/us17_latency.json"

aws s3 cp "$RUNS/us17_latency.json" "$S3/us17_latency.json" --only-show-errors
echo
echo "=== latency done  $(date -Is) ==="
