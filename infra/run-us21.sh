#!/bin/bash
# =============================================================================
#  US21: C7 rutting: detection, segmentation, and the distractor probe
# =============================================================================
#      bash infra/run-us21.sh             # fetch, prepare, train both, probe
#      bash infra/run-us21.sh --no-fetch   # reuse an existing download
#
#  C7_rutting has sat in configs/classmap.yaml as PROVISIONAL since the
#  taxonomy was frozen, with the note "949 images, instance count unpublished"
#  and a data-availability finding that the set had not been located. It has
#  now been located: Saha, Arya, Kumar, Maeda and Sekimoto (Mendeley Data,
#  CC BY 4.0, doi:10.17632/v8mbnvx8pv.1), from the same lab as RDD2022.
#
#  Three jobs, in increasing order of what can go wrong:
#    1. detection, the source ships YOLO-format labels; remap and train
#    2. segmentation, from the COCO polygon export; the task the class map
#                         assigns C7
#    3. distractor probe, run the SIX-class model over these images and count
#                         what it predicts, since it has no rutting class
#
#  Scale note: these are small crops, masks around 259x194, so training runs at
#  imgsz 320 rather than the project's usual 640. Upscaling by three would not
#  make the numbers comparable with the RDD2022 work, and pretending otherwise
#  would be the error. The probe, by contrast, runs at 640, because that is how
#  the six-class model was trained and would be deployed.
# =============================================================================
set -euo pipefail

FETCH=1
[ "${1:-}" = "--no-fetch" ] && FETCH=0

RAW=/opt/rdd/data/rutting-raw
YOLO=/opt/rdd/data/yolo/rutting
RUNS=/opt/rdd/runs
S3=s3://rdd-general-299284685700-us-east-1/runs
KEEP=/opt/rdd/KEEP_ALIVE
ZIP_URL="https://data.mendeley.com/public-files/datasets/v8mbnvx8pv/files/56114c84-bfc9-47d5-80a8-36172e5e6282/file_downloaded"
ZIP_SHA=ba14829a4b581522f2011952fc585b48eaebef2d112ea7e2927045ef6705c58a

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

if pgrep -f 'rdd.train.train_yolo' > /dev/null; then
  echo "a trainer is running here; US21 would contend with it for the GPU" >&2
  exit 1
fi

HELD=0
if [ ! -f "$KEEP" ]; then touch "$KEEP"; HELD=1; fi
trap '[ "$HELD" = 1 ] && rm -f "$KEEP"; echo "watchdog restored"' EXIT

if [ "$FETCH" = "1" ]; then
  echo "== fetching the rutting dataset  $(date -Is)"
  mkdir -p "$RAW"
  if [ ! -s "$RAW/rutting.zip" ]; then
    curl -sS -L -o "$RAW/rutting.zip" "$ZIP_URL"
  fi
  # Mendeley publishes a hash for this file, so verify rather than assume.
  GOT=$(sha256sum "$RAW/rutting.zip" | cut -d' ' -f1)
  if [ "$GOT" != "$ZIP_SHA" ]; then
    echo "archive hash mismatch:" >&2
    echo "  expected $ZIP_SHA" >&2
    echo "  got      $GOT" >&2
    exit 1
  fi
  echo "   hash matches the published value"
  rm -rf "$RAW/x"; mkdir -p "$RAW/x"
  unzip -q "$RAW/rutting.zip" -d "$RAW/x"
fi

SRC="$RAW/x/RoadRuttingDatasetSekilab"
[ -d "$SRC" ] || { echo "extracted dataset not found at $SRC" >&2; exit 1; }

echo
echo "== preparing both views  $(date -Is)"
rddpython -m rdd.data.rutting_to_yolo --src "$SRC" --out "$YOLO" --val-fraction 0.2

echo
echo "== 1/3 detection  $(date -Is)"
rddpython -m rdd.train.train_rutting --task detect \
  --data "$YOLO/detect/data.yaml" \
  --epochs 100 --imgsz 320 --batch 16 --seed 0 \
  --name us21_rutting_det_s0 --project "$RUNS" --s3 "$S3"

if [ -s "$YOLO/segment/data.yaml" ]; then
  echo
  echo "== 2/3 segmentation  $(date -Is)"
  rddpython -m rdd.train.train_rutting --task segment \
    --data "$YOLO/segment/data.yaml" \
    --epochs 100 --imgsz 320 --batch 16 --seed 0 \
    --name us21_rutting_seg_s0 --project "$RUNS" --s3 "$S3"
else
  echo "   (no segmentation view prepared; skipping)"
fi

echo
echo "== 3/3 distractor probe with the six-class baseline  $(date -Is)"
if [ ! -s "$RUNS/tierA_base6_s0/weights/best.pt" ]; then
  mkdir -p "$RUNS/tierA_base6_s0/weights"
  aws s3 cp "$S3/tierA_base6_s0/weights/best.pt" \
    "$RUNS/tierA_base6_s0/weights/best.pt" --only-show-errors
fi
rddpython -m rdd.eval.distractor \
  --weights "$RUNS/tierA_base6_s0/weights/best.pt" \
  --images "$YOLO/detect/images/val" \
  --imgsz 640 --conf 0.25 \
  --out "$RUNS/us21_distractor.json"

aws s3 cp "$RUNS/us21_distractor.json" "$S3/us21_distractor.json" --only-show-errors
aws s3 cp "$YOLO/preparation.json" "$S3/us21_preparation.json" --only-show-errors
echo
echo "=== US21 done  $(date -Is) ==="
