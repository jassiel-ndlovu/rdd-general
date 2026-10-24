#!/bin/bash
# =============================================================================
#  Tier B, second attempt -- balanced tiled export, then a longer run
# =============================================================================
#  Run on the instance:
#      nohup bash infra/run-tierB-v2.sh > /opt/rdd/logs/tierb_v2.log 2>&1 &
#
#  The first attempt (tierB_tiled1280_s0) lost 0.035 pooled mAP50 against
#  Tier A. The untiled control decomposed that into -0.0277 from training and
#  -0.0073 from tiled inference, so most of the damage was in the training set
#  rather than the merge.
#
#  The cause: of Norway's 55,496 training tiles, 49,784 carry no label. That
#  single subset became 63% of the whole training set, and Norway's share of
#  training went from 22.7% to 70.1%. The corpus-wide negative ratio looked
#  acceptable at 2.36:1 only because the pass-through countries diluted it;
#  within Norway it is 8.7:1.
#
#  This run drops 88.5% of the grid-generated empty tiles, which brings Norway
#  to roughly one empty tile per labelled one and its share of training to
#  about a third. Pass-through images are untouched, so the negatives Tier A
#  trained on are all still there.
#
#  ONE variable changes against the first attempt, plus the epoch count. The
#  clipping rule is deliberately left alone: the evidence points at the
#  distribution, not at fragment dropping, and changing both would make the
#  result unattributable.
# =============================================================================
set -euo pipefail

OUT=/opt/rdd/data/yolo/tiled1280_bal
S3=s3://rdd-general-299284685700-us-east-1/runs
NAME=tierB_tiled1280_bal_s0
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

touch "$KEEP"
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed at $(date -Is)"' EXIT

echo "=== 1/3 balanced tiled export  $(date -Is) ==="
rm -rf "$OUT"
rddpython -m rdd.data.to_yolo_tiled \
    --coco   /opt/rdd/data/coco/rdd2022_train.json \
    --splits /opt/rdd/data/splits/rdd2022_core.json \
    --root   /opt/rdd/data/rdd2022/RDD2022 \
    --out    "$OUT" \
    --tile 1280 --overlap 0.2 --tile-above 1200 \
    --drop-empty-fraction 0.885

N_TRAIN=$(ls "$OUT/images/train" | wc -l)
N_VAL=$(ls "$OUT/images/val" | wc -l)
echo "train tiles: $N_TRAIN   val tiles: $N_VAL"

echo
echo "country mix of the new training set:"
for c in China_MotorBike Czech India Japan Norway United_States; do
    n=$(ls "$OUT/images/train" | grep -c "^${c}_" || true)
    printf '  %-18s %6d  %5.1f%%\n' "$c" "$n" \
        "$(rddpython -c "print(100*$n/$N_TRAIN)")"
done
echo "  (Tier A for comparison: Norway 22.7%, Japan 29.2%, India 21.4%,"
echo "   United_States 13.4%, Czech 7.9%, China_MotorBike 5.5%)"

# 100 epochs over this set is about 1.15x Tier A's gradient budget -- near
# enough to step parity to report both ways, and far enough past the first
# attempt's 39 epochs to actually converge. The first attempt's best epoch was
# its last, so it was still climbing when the budget ran out.
EPOCHS=100
STEPS=$(rddpython -c "print(round($EPOCHS*$N_TRAIN/32))")
echo
echo "epochs: $EPOCHS   gradient steps: $STEPS   (Tier A: 95581)"

echo
echo "=== 2/3 training  $(date -Is) ==="
rddpython -m rdd.train.train_yolo \
    --data "$OUT/data.yaml" \
    --model /opt/rdd/code/yolo11s.pt \
    --epochs "$EPOCHS" --imgsz 640 --batch 32 --seed 0 \
    --workers 4 --patience 30 \
    --name "$NAME" \
    --project /opt/rdd/runs \
    --s3 "$S3"

echo
echo "=== 3/3 evaluation  $(date -Is) ==="
BEST=/opt/rdd/runs/$NAME/weights/best.pt
rddpython -m rdd.eval.tiled_predict \
    --weights "$BEST" \
    --data /opt/rdd/data/yolo/core/data.yaml \
    --tiling-from "$OUT/data.yaml" \
    --out /opt/rdd/runs/$NAME/tiled_eval \
    --dump /opt/rdd/runs/$NAME/tiled_eval/dets.jsonl \
    --imgsz 640 --ap-engine both

rddpython -m rdd.eval.tiled_predict \
    --weights "$BEST" \
    --data /opt/rdd/data/yolo/core/data.yaml \
    --untiled \
    --out /opt/rdd/runs/$NAME/untiled_eval \
    --imgsz 640

aws s3 sync /opt/rdd/runs/$NAME "$S3/$NAME" \
    --exclude "_per_country/*" --exclude "*/_per_country/*" --only-show-errors

echo
echo "=== done  $(date -Is) ==="
