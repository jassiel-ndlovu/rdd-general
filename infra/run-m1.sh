#!/usr/bin/env bash
# =============================================================================
#  RDD project -- run M1 stages 3-6 end to end
# =============================================================================
#  Assumes stages 0-2 are already done, i.e. the dataset is extracted and the
#  canonical COCO store exists at data/coco/rdd2022_train.json.
#
#  Stage 3  de-duplicate        perceptual hash, banded LSH
#  Stage 4  grouped splits      by country + derived road segment
#  Stage 6  YOLO export         generated view of the COCO store
#
#  RUN ON THE INSTANCE:  bash infra/run-m1.sh
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

# All of this is CPU-only, so the GPU watchdog would read the box as idle and
# stop it mid-run. Suspend, and release however we exit.
KEEP="$HOME_DIR/KEEP_ALIVE"
touch "$KEEP"
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed"' EXIT

cd "$CODE"
export PYTHONPATH=src

say() { printf '\n\033[1;36m### %s\033[0m\n' "$*"; }

say "STAGE 2 -- VOC to canonical COCO"
"$PY" -m rdd.data.voc_to_coco \
  --root "$DATA/rdd2022/RDD2022" \
  --source rdd2022 --split train \
  --out "$DATA/coco/rdd2022_train.json"

say "STAGE 3 -- de-duplication"
"$PY" -m rdd.data.dedup \
  --coco "$DATA/coco/rdd2022_train.json" \
  --root "$DATA/rdd2022/RDD2022" \
  --out  "$DATA/coco/dedup_report.json" \
  --threshold "${DEDUP_THRESHOLD:-4}"
# NOTE: --max-seq-gap is deliberately NOT passed. It defaults to 0 (disabled)
# because rdd.verify.sequence_locality measured no locality in RDD2022's
# filename numbering. Set MAX_SEQ_GAP and add the flag back only for a dataset
# where that verifier reports TEMPORAL.

say "STAGE 4 -- grouped splits"
"$PY" -m rdd.data.splits \
  --coco  "$DATA/coco/rdd2022_train.json" \
  --dedup "$DATA/coco/dedup_report.json" \
  --out   "$DATA/splits/rdd2022_core.json" \
  --val-fraction "${VAL_FRACTION:-0.15}" \
  --seed "${SPLIT_SEED:-0}"

say "STAGE 6 -- YOLO export"
"$PY" -m rdd.data.to_yolo \
  --coco   "$DATA/coco/rdd2022_train.json" \
  --splits "$DATA/splits/rdd2022_core.json" \
  --root   "$DATA/rdd2022/RDD2022" \
  --out    "$DATA/yolo/core"

say "SYNC artefacts to S3"
aws s3 cp "$DATA/coco/rdd2022_train.json"     "s3://$RDD_BUCKET/coco/"   --only-show-errors
aws s3 cp "$DATA/coco/dedup_report.json"      "s3://$RDD_BUCKET/coco/"   --only-show-errors
aws s3 cp "$DATA/splits/rdd2022_core.json"    "s3://$RDD_BUCKET/splits/" --only-show-errors
aws s3 cp "$DATA/yolo/core/data.yaml"         "s3://$RDD_BUCKET/yolo/"   --only-show-errors

say "M1 COMPLETE"
echo "  split hashes (cite these in every results table):"
grep -E '"sha256"' "$DATA/splits/rdd2022_core.json" | sed 's/^/    /'
