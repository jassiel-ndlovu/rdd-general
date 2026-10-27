#!/bin/bash
# =============================================================================
#  Tier B: evaluate, once training finishes
# =============================================================================
#  Run on the instance:
#      nohup bash infra/run-tierB-eval.sh > /opt/rdd/logs/tierb_eval.log 2>&1 &
#
#  Waits for the training run to exit, then produces the two things the write-up
#  needs:
#
#    1. The Tier A comparable number for the tiled checkpoint. Tiled inference
#       over the ORIGINAL whole validation images, detections merged back to
#       full-image coordinates, scored against whole-image ground truth. The
#       tiled view's own metrics are not comparable with Tier A, because both
#       halves of that split are tiled; this is.
#
#    2. The pothole false negatives for US14b, from the Tier A checkpoint.
#       Written long ago, never run.
#
#  Safe to start while training is still going; it waits.
# =============================================================================
set -uo pipefail

RUNS=/opt/rdd/runs
DATA=/opt/rdd/data
S3=s3://rdd-general-299284685700-us-east-1/runs
TIER_B=tierB_tiled1280_s0
TIER_A=tierA_base6_s0
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

# run-tierB.sh releases KEEP_ALIVE from its own exit trap, and the evaluation
# below is partly CPU-bound, which looks idle to the watchdog. Re-asserting it
# on every poll is simpler than coordinating with the other script's trap.
touch "$KEEP"

# Record progress for the whole run. The daemon is idempotent (one pidfile) and
# outlives this script on purpose, so the tail of a run and its upload are
# recorded too. It holds no KEEP_ALIVE, so it cannot keep the instance awake.
bash infra/progress-daemon.sh >/dev/null 2>&1 &
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed at $(date -Is)"' EXIT

echo "=== waiting for training to finish  $(date -Is) ==="
while pgrep -f "rdd.train.train_yolo" >/dev/null 2>&1; do
    touch "$KEEP"
    sleep 120
done
touch "$KEEP"
echo "training process has exited  $(date -Is)"

BEST="$RUNS/$TIER_B/weights/best.pt"
if [ ! -f "$BEST" ]; then
    echo "no checkpoint at $BEST, training did not produce one. Stopping."
    exit 1
fi
echo "checkpoint: $BEST  ($(du -h "$BEST" | cut -f1))"

# -----------------------------------------------------------------------------
echo
echo "=== 1/3 tiled inference, Tier A comparable  $(date -Is) ==="
# --data is the UNTILED view: its val images are what we tile, and its val
# labels are the ground truth. --tiling-from takes tile size, overlap and the
# pass-through threshold from the training export's own header, so evaluation
# cannot silently drift from the configuration the model was trained on.
rddpython -m rdd.eval.tiled_predict \
    --weights "$BEST" \
    --data "$DATA/yolo/core/data.yaml" \
    --tiling-from "$DATA/yolo/tiled1280/data.yaml" \
    --out "$RUNS/$TIER_B/tiled_eval" \
    --dump "$RUNS/$TIER_B/tiled_eval/dets.jsonl" \
    --imgsz 640 --ap-engine both
RC=$?
echo "tiled_predict exit: $RC"

# -----------------------------------------------------------------------------
echo
echo "=== 2/3 the same checkpoint, untiled  $(date -Is) ==="
# The honest control. Without it, a difference between this model and Tier A
# confounds two changes: the model was trained on tiles AND is being evaluated
# with tiling. Running it untiled separates the two.
rddpython -m rdd.eval.tiled_predict \
    --weights "$BEST" \
    --data "$DATA/yolo/core/data.yaml" \
    --untiled \
    --out "$RUNS/$TIER_B/untiled_eval" \
    --imgsz 640
echo "untiled control exit: $?"

# -----------------------------------------------------------------------------
echo
echo "=== 3/3 pothole false negatives, Tier A checkpoint (US14b)  $(date -Is) ==="
# Merges into the existing revalidation.json rather than replacing it.
rddpython -m rdd.eval.revalidate --stage fn \
    --weights "$RUNS/$TIER_A/weights/best.pt" \
    --data "$DATA/yolo/core/data.yaml" \
    --out "$RUNS/$TIER_A/reval" \
    --imgsz 640 --fn-class C4_pothole
echo "false-negative stage exit: $?"

# -----------------------------------------------------------------------------
echo
echo "=== syncing  $(date -Is) ==="
for d in "$TIER_B" "$TIER_A"; do
    aws s3 sync "$RUNS/$d" "$S3/$d" \
        --exclude "_per_country/*" --exclude "*/_per_country/*" \
        --only-show-errors && echo "  synced $d"
done

echo
echo "=== done  $(date -Is) ==="
