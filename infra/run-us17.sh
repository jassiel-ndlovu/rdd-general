#!/bin/bash
# =============================================================================
#  US17: architecture variants for thin cracks
# =============================================================================
#      bash infra/run-us17.sh a      # stride-4 detection head
#      bash infra/run-us17.sh b      # strip attention
#      bash infra/run-us17.sh c      # dynamic upsampling
#      bash infra/run-us17.sh d      # all three combined
#      bash infra/run-us17.sh abc    # all three, in sequence
#
#  Trains a variant on the untiled core view under settings otherwise identical
#  to Tier A, then evaluates it the same way. No tiling is used anywhere: Tier B
#  established that tiling cannot help elongated damage because it cuts it.
#
#  Comparability rests on the same data and split, the same 100 epochs at imgsz
#  640 batch 32 seed 0, and the same COCO-pretrained starting weights.
#
#  The weights need care in two places. Inserting a module renumbers every
#  later layer, so ultralytics' .load() drops their pretrained weights without
#  warning; rdd.models.transfer_weights retries a shifted index. The transfer
#  then has to survive into the trainer, which it does not by default. See the
#  note in train_yolo.py. Each run logs the count ultralytics reports, which is
#  the number to quote.
#
#  A relaunch pulls whatever the variant has already pushed to S3 and continues
#  from last.pt, so a spot interruption costs the epoch in progress. If the
#  checkpoint is not resumable the run falls back to a fresh start rather than
#  failing.
#
#  Measured costs against the 21.8 GFLOP baseline:
#    a  29.8 GFLOPs (x1.36), 9.63M params, 525/593 tensors transferred
#    b  23.1 GFLOPs (x1.06), 10.16M params, 517/523 transferred
#    c  21.8 GFLOPs (x1.00),  9.46M params, 497/503 transferred
#    d  30.9 GFLOPs (x1.42), 10.29M params, 563/631 transferred
# =============================================================================
set -euo pipefail

WHICH="${1:-}"
[ -n "$WHICH" ] || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 1; }

RUNS=/opt/rdd/runs
DATA=/opt/rdd/data
S3=s3://rdd-general-299284685700-us-east-1/runs
KEEP=/opt/rdd/KEEP_ALIVE
EPOCHS=100

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

touch "$KEEP"
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed at $(date -Is)"' EXIT

bash infra/progress-daemon.sh >/dev/null 2>&1 &

train_fresh() {
    local cfg="$1" name="$2" shift_max="$3"
    rddpython -m rdd.train.train_yolo \
        --data "$DATA/yolo/core/data.yaml" \
        --model "/opt/rdd/code/$cfg" \
        --transfer-from /opt/rdd/code/yolo11s.pt \
        --transfer-max-shift "$shift_max" \
        --epochs "$EPOCHS" --imgsz 640 --batch 32 --seed 0 \
        --workers 4 --patience 30 \
        --name "$name" \
        --project "$RUNS" \
        --s3 "$S3"
}

train_resume() {
    local name="$1"
    # --model is the checkpoint rather than the YAML, since ultralytics resumes
    # from the run directory recorded inside it. No --transfer-from: the
    # checkpoint already carries the transferred weights.
    rddpython -m rdd.train.train_yolo \
        --data "$DATA/yolo/core/data.yaml" \
        --model "$RUNS/$name/weights/last.pt" \
        --resume \
        --epochs "$EPOCHS" --imgsz 640 --batch 32 --seed 0 \
        --workers 4 --patience 30 \
        --name "$name" \
        --project "$RUNS" \
        --s3 "$S3"
}

run_variant() {
    local key="$1" cfg name shift_max
    # shift_max is how far back the weight transfer looks for a renumbered
    # layer. The combined variant inserts ten ahead of Detect, so it needs
    # more than the default of six.
    case "$key" in
        a) cfg=configs/yolo11s-p2.yaml;       name=us17a_p2head_s0;   shift_max=6 ;;
        b) cfg=configs/yolo11s-strip.yaml;    name=us17b_strip_s0;    shift_max=6 ;;
        c) cfg=configs/yolo11s-dysample.yaml; name=us17c_dysample_s0; shift_max=6 ;;
        d) cfg=configs/yolo11s-combined.yaml; name=us17d_combined_s0; shift_max=12 ;;
        *) echo "unknown variant '$key'"; return 1 ;;
    esac

    echo
    echo "=============================================================="
    echo " US17 variant ${key^^},  $name   $(date -Is)"
    echo "=============================================================="

    # Anything this variant has already pushed. Harmless when there is nothing.
    aws s3 sync "$S3/$name" "$RUNS/$name" --only-show-errors 2>/dev/null || true

    local done_ep=0
    if [ -s "$RUNS/$name/results.csv" ]; then
        done_ep=$(tail -1 "$RUNS/$name/results.csv" | cut -d, -f1 | tr -dc '0-9')
        done_ep=${done_ep:-0}
    fi

    if [ -s "$RUNS/$name/weights/last.pt" ] && [ "$done_ep" -gt 0 ] \
       && [ "$done_ep" -lt "$EPOCHS" ]; then
        echo "  found a checkpoint at epoch $done_ep of $EPOCHS; resuming"
        if ! train_resume "$name"; then
            echo "  resume failed; discarding the partial run and starting fresh"
            rm -rf "$RUNS/$name"
            train_fresh "$cfg" "$name" "$shift_max"
        fi
    elif [ "$done_ep" -ge "$EPOCHS" ]; then
        echo "  already trained to epoch $done_ep; skipping to evaluation"
    else
        echo "  no checkpoint; training from the pretrained initialisation"
        train_fresh "$cfg" "$name" "$shift_max"
    fi

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
