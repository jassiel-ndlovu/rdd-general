#!/bin/bash
# =============================================================================
#  US17 variant A -- stride-4 detection head
# =============================================================================
#  Run on the instance:
#      nohup bash infra/run-us17a.sh > /opt/rdd/logs/us17a.log 2>&1 &
#
#  Trains yolo11s with a fourth detection head at P2/4 on the UNTILED core
#  view, under settings otherwise identical to Tier A, then evaluates it the
#  same way Tier A was evaluated.
#
#  Why this variant first. Tier A's weakest classes are the elongated cracks,
#  and Tier B established that tiling cannot fix them because tiling cuts long
#  objects across seams. A stride-4 head attacks the same deficit without
#  introducing seams at all: it gives the detector a feature map at four-pixel
#  stride rather than eight, so a two or three pixel crack still has a cell to
#  be detected in. It is also the only one of US17's three modifications that
#  needs no new modules, so it can be measured before the others are written.
#
#  Comparability. Same data, same split, same 100 epochs, same imgsz 640,
#  batch 32, seed 0, and the same COCO-pretrained starting weights as Tier A
#  via --transfer-from. Only the architecture differs. Without the transfer a
#  .yaml builds random weights and the comparison would confound architecture
#  with initialisation.
#
#  Cost. 29.8 GFLOPs against the baseline's 21.8, so expect roughly 20 hours
#  rather than 15. Parameters barely move: 9.63M against 9.46M.
#
#  Watch the first epoch. Batch 32 at 640 used 7.6 GB on the baseline; the P2
#  head adds a 160x160 feature map and may not fit 15 GB. If it runs out of
#  memory, the honest fix is to drop the batch size AND rerun the Tier A
#  baseline at the same batch, because batch size is not a free variable.
# =============================================================================
set -euo pipefail

RUNS=/opt/rdd/runs
DATA=/opt/rdd/data
S3=s3://rdd-general-299284685700-us-east-1/runs
NAME=us17a_p2head_s0
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

touch "$KEEP"
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed at $(date -Is)"' EXIT

bash infra/progress-daemon.sh >/dev/null 2>&1 &

echo "=== 1/2 training  $(date -Is) ==="
rddpython -m rdd.train.train_yolo \
    --data "$DATA/yolo/core/data.yaml" \
    --model /opt/rdd/code/configs/yolo11s-p2.yaml \
    --transfer-from /opt/rdd/code/yolo11s.pt \
    --epochs 100 --imgsz 640 --batch 32 --seed 0 \
    --workers 4 --patience 30 \
    --name "$NAME" \
    --project "$RUNS" \
    --s3 "$S3"

echo
echo "=== 2/2 evaluation  $(date -Is) ==="
# Untiled, exactly as Tier A was evaluated. No tiling anywhere in US17.
rddpython -m rdd.eval.revalidate --stage metrics \
    --weights "$RUNS/$NAME/weights/best.pt" \
    --data "$DATA/yolo/core/data.yaml" \
    --out "$RUNS/$NAME/reval" \
    --imgsz 640 --fn-class C4_pothole

aws s3 sync "$RUNS/$NAME" "$S3/$NAME" \
    --exclude "_per_country/*" --exclude "*/_per_country/*" --only-show-errors

echo
echo "=== done  $(date -Is) ==="
