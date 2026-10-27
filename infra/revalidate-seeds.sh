#!/bin/bash
# =============================================================================
#  Revalidate Tier A seeds 1 and 2, for the spreads that are currently missing
# =============================================================================
#      bash infra/revalidate-seeds.sh [1 2]
#
#  The Tier A aggregate carries a standard deviation for pooled mAP50 and for
#  per-class AP50, but not for per-class-within-country, because only seed 0
#  had been through the revalidation pass that produces that breakdown. US14a
#  and US17 both quote figures at that granularity, so the spreads are needed.
#
#  Inference only, using the existing weights. Two passes over the 5,398-image
#  validation set, roughly one to two hours.
# =============================================================================
set -euo pipefail

if [ "$#" -gt 0 ]; then SEEDS=("$@"); else SEEDS=(1 2); fi

RUNS=/opt/rdd/runs
DATA=/opt/rdd/data
S3=s3://rdd-general-299284685700-us-east-1/runs
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

if pgrep -f 'rdd.train.train_yolo' > /dev/null; then
  echo "a trainer is running here; this would contend with it for the GPU" >&2
  exit 1
fi

HELD=0
if [ ! -f "$KEEP" ]; then touch "$KEEP"; HELD=1; fi
trap '[ "$HELD" = 1 ] && rm -f "$KEEP"; echo "watchdog restored"' EXIT

for s in "${SEEDS[@]}"; do
  run="tierA_base6_s${s}"
  echo
  echo "=============================================================="
  echo " revalidating $run   $(date -Is)"
  echo "=============================================================="

  if [ ! -s "$RUNS/$run/weights/best.pt" ]; then
    mkdir -p "$RUNS/$run/weights"
    aws s3 cp "$S3/$run/weights/best.pt" "$RUNS/$run/weights/best.pt" \
      --only-show-errors
  fi

  # --stage metrics only: the false-negative dump is a seed-0 artefact for
  # US14b and is not what the spreads need, and skipping it halves the work.
  rddpython -m rdd.eval.revalidate --stage metrics \
    --weights "$RUNS/$run/weights/best.pt" \
    --data "$DATA/yolo/core/data.yaml" \
    --out "$RUNS/$run/reval" \
    --imgsz 640 --fn-class C4_pothole

  aws s3 cp "$RUNS/$run/reval/revalidation.json" \
    "$S3/$run/reval/revalidation.json" --only-show-errors
  echo "--- $run done  $(date -Is) ---"
done

echo
echo "=== seed revalidation complete  $(date -Is) ==="
echo "Aggregate the spreads with rdd.eval.aggregate once seed 0, 1 and 2 are all present."
