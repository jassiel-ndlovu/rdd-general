#!/bin/bash
# =============================================================================
#  US17 -- architecture variants for thin cracks
# =============================================================================
#      bash infra/run-us17.sh a      # stride-4 detection head
#      bash infra/run-us17.sh b      # strip attention
#      bash infra/run-us17.sh c      # dynamic upsampling
#      bash infra/run-us17.sh abc    # all three, in sequence
#
#  Trains a variant on the UNTILED core view under settings otherwise
#  identical to Tier A, then evaluates it the way Tier A was evaluated. No
#  tiling anywhere: Tier B established that tiling cannot help elongated
#  damage because it cuts it, so US17 attacks the same deficit from the
#  architecture side instead.
#
#  Comparability rests on three things being held fixed against Tier A: the
#  same data and split, the same 100 epochs at imgsz 640 batch 32 seed 0, and
#  the same COCO-pretrained starting weights.
#
#  That last one needs care. Inserting a module renumbers every layer after it,
#  so Ultralytics' own .load() silently drops the pretrained weights for all of
#  them -- 40% of the model, for two of these three variants. The trainer uses
#  rdd.models.transfer_weights instead, which retries a shifted index and
#  recovers them. Each run records how many tensors transferred, so the
#  write-up can state it rather than assume it.
#
#  Measured costs against the 21.8 GFLOP baseline:
#    a  29.8 GFLOPs (x1.36), 9.63M params, 461/593 tensors transferred
#    b  23.1 GFLOPs (x1.06), 10.16M params, 499/523 transferred
#    c  21.8 GFLOPs (x1.00),  9.46M params, 499/503 transferred
# =============================================================================
set -euo pipefail

WHICH="${1:-}"
[ -n "$WHICH" ] || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 1; }

RUNS=/opt/rdd/runs
DATA=/opt/rdd/data
S3=s3://rdd-general-299284685700-us-east-1/runs
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

touch "$KEEP"
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed at $(date -Is)"' EXIT

bash infra/progress-daemon.sh >/dev/null 2>&1 &

run_variant() {
    local key="$1" cfg name
    case "$key" in
        a) cfg=configs/yolo11s-p2.yaml;       name=us17a_p2head_s0 ;;
        b) cfg=configs/yolo11s-strip.yaml;    name=us17b_strip_s0 ;;
        c) cfg=configs/yolo11s-dysample.yaml; name=us17c_dysample_s0 ;;
        *) echo "unknown variant '$key'"; return 1 ;;
    esac

    echo
    echo "=============================================================="
    echo " US17 variant ${key^^}  --  $name   $(date -Is)"
    echo "=============================================================="

    rddpython -m rdd.train.train_yolo \
        --data "$DATA/yolo/core/data.yaml" \
        --model "/opt/rdd/code/$cfg" \
        --transfer-from /opt/rdd/code/yolo11s.pt \
        --epochs 100 --imgsz 640 --batch 32 --seed 0 \
        --workers 4 --patience 30 \
        --name "$name" \
        --project "$RUNS" \
        --s3 "$S3"

    echo
    echo "--- evaluating $name ---"
    rddpython -m rdd.eval.revalidate --stage metrics \
        --weights "$RUNS/$name/weights/best.pt" \
        --data "$DATA/yolo/core/data.yaml" \
        --out "$RUNS/$name/reval" \
        --imgsz 640 --fn-class C4_pothole

    aws s3 sync "$RUNS/$name" "$S3/$name" \
        --exclude "_per_country/*" --exclude "*/_per_country/*" --only-show-errors
    echo "--- $name complete  $(date -Is) ---"
}

for ((i = 0; i < ${#WHICH}; i++)); do
    run_variant "${WHICH:$i:1}"
done

echo
echo "=== US17 done  $(date -Is) ==="
