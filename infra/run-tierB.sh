#!/bin/bash
# =============================================================================
#  Tier B: tiled resolution front end, one seed
# =============================================================================
#  Run on the instance:
#      tmux new -s tierB
#      bash infra/run-tierB.sh
#
#  Exports the tiled view of the canonical store, then trains on it with the
#  epoch count step-matched to Tier A. About 15 hours unattended.
#
#  Configuration is argued in docs/13-tier-b-plan.md section 5: tile 1280 with
#  20% overlap, applied only to images whose larger side exceeds 1200 px.
#  Tiling at 640 would leave 83% of tiles empty.
#
#  The watchdog is held off for the whole run because the export phase is CPU
#  only and would otherwise look idle. The trap re-arms it. A SIGKILL skips the
#  trap and leaves the instance running, so check infra/status.sh after an
#  abnormal exit.
# =============================================================================
set -euo pipefail

KEEP=/opt/rdd/KEEP_ALIVE
touch "$KEEP"

# The progress daemon is idempotent and outlives this script, so the tail of a
# run and its upload are recorded. It holds no KEEP_ALIVE.
bash infra/progress-daemon.sh >/dev/null 2>&1 &
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed at $(date -Is)"' EXIT

OUT=/opt/rdd/data/yolo/tiled1280
S3=s3://rdd-general-299284685700-us-east-1/runs
cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

echo "=== 1/3 tiled export  $(date -Is) ==="
rddpython -m rdd.data.to_yolo_tiled \
  --coco   /opt/rdd/data/coco/rdd2022_train.json \
  --splits /opt/rdd/data/splits/rdd2022_core.json \
  --root   /opt/rdd/data/rdd2022/RDD2022 \
  --out    "$OUT" \
  --tile 1280 --overlap 0.2 --tile-above 1200

N_TRAIN=$(ls "$OUT/images/train" | wc -l)
N_VAL=$(ls "$OUT/images/val" | wc -l)
echo "tiled train tiles: $N_TRAIN   val tiles: $N_VAL"

# Step matching. Tier A saw 100 epochs over 30,586 images; holding total
# gradient updates fixed means dividing the epoch count by the sample
# multiplier, so that any gain is attributable to resolution and not to a
# larger training budget. See docs/13-tier-b-plan.md section 5.
EPOCHS=$(rddpython -c "print(max(1, round(100*30586/$N_TRAIN)))")
echo "step-matched epochs: $EPOCHS  (k = $(rddpython -c "print(round($N_TRAIN/30586,3))"))"

echo
echo "=== 2/3 training  $(date -Is) ==="
rddpython -m rdd.train.train_yolo \
  --data "$OUT/data.yaml" \
  --model /opt/rdd/code/yolo11s.pt \
  --epochs "$EPOCHS" --imgsz 640 --batch 32 --seed 0 \
  --workers 4 --patience 30 \
  --name tierB_tiled1280_s0 \
  --project /opt/rdd/runs \
  --s3 "$S3"

echo
echo "=== 3/3 done  $(date -Is) ==="
aws s3 sync /opt/rdd/runs/tierB_tiled1280_s0 "$S3/tierB_tiled1280_s0" \
  --exclude "_per_country/*" --exclude "*/_per_country/*" --only-show-errors
echo "synced"
