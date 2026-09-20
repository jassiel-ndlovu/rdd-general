#!/usr/bin/env bash
# =============================================================================
#  RDD project -- Tier A experiment set
# =============================================================================
#  Answers RQ1, RQ2, RQ3 and RQ5 and tests H1 and H2.
#
#    Baseline : 6-class detector (C1-C6), 3 seeds       ~30 GPU-h
#    Control  : 4-class detector (C1-C4), 1 seed        ~10 GPU-h   <- tests H2
#    Eval     : per-class and per-country for all runs   included
#
#  The control uses the SAME images and the SAME split manifest as the
#  baseline. Only the label set differs, so any accuracy change on the shared
#  classes is attributable to broadening the taxonomy and to nothing else.
#  That is the whole point of H2.
#
#  RUN ON THE INSTANCE, inside tmux:
#      tmux new -s tierA
#      bash infra/run-tierA.sh
#      # detach with Ctrl-b d
# =============================================================================
set -euo pipefail

RDD_BUCKET="${RDD_BUCKET:?set RDD_BUCKET}"
HOME_DIR="${RDD_HOME:-/opt/rdd}"
CODE="$HOME_DIR/code"
DATA="$HOME_DIR/data"
PY="${RDD_PY:-rddpython}"

# Python block-buffers stdout when it is not a TTY, so progress lines sit
# invisibly in an 8 KB buffer for the whole run. Force line buffering.
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-yolo11s.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
SEEDS="${SEEDS:-0 1 2}"
S3_RUNS="s3://$RDD_BUCKET/runs"

CORE_CLASSES="C1_longitudinal_crack,C2_transverse_crack,C3_alligator_crack,C4_pothole"

# Training keeps the GPU busy so the watchdog is satisfied, but the export and
# evaluation phases are CPU-only. Hold the watchdog off for the whole run.
KEEP="$HOME_DIR/KEEP_ALIVE"
touch "$KEEP"
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed"' EXIT

cd "$CODE"
export PYTHONPATH=src
export RDD_S3_RUNS="$S3_RUNS"

say() { printf '\n\033[1;36m### %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
say "Building the 4-class control export (same splits, fewer labels)"
# ---------------------------------------------------------------------------
if [ ! -f "$DATA/yolo/control4/data.yaml" ]; then
  "$PY" -m rdd.data.to_yolo \
    --coco   "$DATA/coco/rdd2022_train.json" \
    --splits "$DATA/splits/rdd2022_core.json" \
    --root   "$DATA/rdd2022/RDD2022" \
    --out    "$DATA/yolo/control4" \
    --only-classes "$CORE_CLASSES"
else
  echo "  already built"
fi

# ---------------------------------------------------------------------------
say "BASELINE -- 6 classes, seeds: $SEEDS"
# ---------------------------------------------------------------------------
for s in $SEEDS; do
  NAME="tierA_base6_s${s}"
  if [ -f "$HOME_DIR/runs/$NAME/summary.json" ]; then
    echo "  $NAME already complete, skipping"
    continue
  fi
  say "  training $NAME"
  "$PY" -m rdd.train.train_yolo \
    --data "$DATA/yolo/core/data.yaml" \
    --model "$MODEL" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
    --seed "$s" --name "$NAME" --s3 "$S3_RUNS"
done

# ---------------------------------------------------------------------------
say "CONTROL -- 4 classes, seed 0 (tests H2)"
# ---------------------------------------------------------------------------
NAME="tierA_ctrl4_s0"
if [ -f "$HOME_DIR/runs/$NAME/summary.json" ]; then
  echo "  $NAME already complete, skipping"
else
  "$PY" -m rdd.train.train_yolo \
    --data "$DATA/yolo/control4/data.yaml" \
    --model "$MODEL" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
    --seed 0 --name "$NAME" --s3 "$S3_RUNS"
fi

# ---------------------------------------------------------------------------
say "AGGREGATE"
# ---------------------------------------------------------------------------
"$PY" -m rdd.eval.aggregate --runs "$HOME_DIR/runs" --out "$HOME_DIR/runs/tierA_summary.json" || \
  echo "  (aggregate step unavailable; per-run summary.json files are still written)"

aws s3 sync "$HOME_DIR/runs" "$S3_RUNS" --only-show-errors
say "TIER A COMPLETE -- results at $S3_RUNS"
