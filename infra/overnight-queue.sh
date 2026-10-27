#!/bin/bash
# =============================================================================
#  Overnight queue: US21 detection at 640, then variant B's zero-shot
# =============================================================================
#      nohup bash infra/overnight-queue.sh > /opt/rdd/logs/overnight.log 2>&1 &
#
#  Runs on whichever box already holds the prepared rutting views and the
#  Stellenbosch export, to avoid re-downloading 24 GB.
#
#  Two jobs, in order:
#
#    1. US21 rutting detection at imgsz 640, repeating the 320 run at a larger
#       scale. The detection boxes have a median area of 0.8% of the image, so
#       320 leaves a typical target around 28x28 pixels. In the event this made
#       almost no difference, which rules resolution out as the constraint.
#
#    2. Variant B's zero-shot evaluation from its final weights. It waits for
#       B's evaluation to appear in S3, which signals that best.pt is final.
#
#  Everything runs under nohup on the instance, so nothing local needs to stay
#  alive.
# =============================================================================
set -uo pipefail

RUNS=/opt/rdd/runs
DATA=/opt/rdd/data
S3=s3://rdd-general-299284685700-us-east-1/runs
BUCKET=rdd-general-299284685700-us-east-1
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

# Held for the whole queue and released at the end so the box stops itself.
# The idle counter persists in /var/tmp across a stop/start, so it is cleared
# as well as the flag being set.
echo 0 > /var/tmp/rdd-idle-count 2>/dev/null || true
touch "$KEEP"
trap 'echo 0 > /var/tmp/rdd-idle-count 2>/dev/null || true; rm -f "$KEEP"; echo "watchdog released $(date -Is)"' EXIT

echo "=== overnight queue starting $(date -Is) ==="

# ---------------------------------------------------------------------------
# 1. US21 rutting detection at 640
# ---------------------------------------------------------------------------
echo
echo "== 1/2 rutting detection at imgsz 640  $(date -Is)"
if [ ! -s "$DATA/yolo/rutting/detect/data.yaml" ]; then
  echo "   rutting data missing; re-preparing from the archive"
  if [ -d "$DATA/rutting-raw/x/RoadRuttingDatasetSekilab" ]; then
    rddpython -m rdd.data.rutting_to_yolo \
      --src "$DATA/rutting-raw/x/RoadRuttingDatasetSekilab" \
      --out "$DATA/yolo/rutting" --val-fraction 0.2
  else
    echo "   archive missing too; running the full US21 fetch"
    bash infra/run-us21.sh || echo "   US21 fetch failed"
  fi
fi

if [ -s "$DATA/yolo/rutting/detect/data.yaml" ]; then
  rddpython -m rdd.train.train_rutting --task detect \
    --data "$DATA/yolo/rutting/detect/data.yaml" \
    --epochs 100 --imgsz 640 --batch 16 --seed 0 \
    --name us21_rutting_det640_s0 --project "$RUNS" --s3 "$S3"
  echo "   detection-640 exit $?"
else
  echo "   skipped: no rutting detection view available"
fi

# ---------------------------------------------------------------------------
# 2. Variant B's zero-shot, once B has finished
# ---------------------------------------------------------------------------
echo
echo "== 2/2 waiting for variant B to finish  $(date -Is)"
READY=0
for i in $(seq 1 240); do          # up to 8 hours, checked every 2 minutes
  if aws s3 ls "$S3/us17b_strip_s0/reval/revalidation.json" > /dev/null 2>&1; then
    echo "   B's evaluation is in S3 after ~$((i * 2)) minutes; weights are final"
    READY=1
    break
  fi
  sleep 120
done

if [ "$READY" = "1" ]; then
  mkdir -p "$RUNS/us17b_strip_s0/weights"
  aws s3 cp "$S3/us17b_strip_s0/weights/best.pt" \
    "$RUNS/us17b_strip_s0/weights/best.pt" --only-show-errors
  if [ -s "$DATA/yolo/stellenbosch/data.yaml" ]; then
    echo
    echo "== variant B zero-shot on the Stellenbosch set  $(date -Is)"
    rddpython -m rdd.eval.zeroshot \
      --variant "us17b=$RUNS/us17b_strip_s0/weights/best.pt" \
      --data "$DATA/yolo/stellenbosch/data.yaml" \
      --imgsz 640 --batch 16 --fp-pass --conf 0.25 \
      --out "$RUNS/us22_zeroshot_us17b_final.json"
    aws s3 cp "$RUNS/us22_zeroshot_us17b_final.json" \
      "$S3/us22_zeroshot_us17b_final.json" --only-show-errors
    echo "   B zero-shot exit $?"
  else
    echo "   the Stellenbosch view is missing from this box; re-run US22 first" >&2
  fi
else
  echo "   B never reported an evaluation within the window" >&2
fi

echo
echo "=== overnight queue done $(date -Is) ==="
