#!/bin/bash
# =============================================================================
#  US22: zero-shot evaluation on held-out South African imagery
# =============================================================================
#      bash infra/run-us22.sh            # download, convert, evaluate
#      bash infra/run-us22.sh --no-fetch  # reuse an existing download
#
#  The corpus is Nienaber, Kroon and Booysen's pothole set, released by
#  Stellenbosch University in 2018 under CC0, redistributed by Dataset Ninja.
#  11,497 images, bounding boxes, one class, from a different country, camera
#  and annotation team to anything in training.
#
#  Inference only. No South African imagery is ever trained on, and the
#  generated data.yaml points its train key at an empty directory so it cannot
#  be handed to a trainer by accident.
#
#  The archive is untrusted third-party data, so it lands in its own empty
#  directory and the converter is invoked by absolute path from the code tree,
#  so nothing inside it can shadow a module on import.
# =============================================================================
set -euo pipefail

FETCH=1
[ "${1:-}" = "--no-fetch" ] && FETCH=0

RAW=/opt/rdd/data/stellenbosch-raw
YOLO=/opt/rdd/data/yolo/stellenbosch
DS_URL="https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMTY1MV9Sb2FkIFBvdGhvbGUgSW1hZ2VzL3JvYWQtcG90aG9sZS1pbWFnZXMtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAiOUFTN25NbHBENDlmdGtRTndxc3Jpb1J1aWFKOG9kR1pHaHl0YUV3YVB6RT0ifQ==?response-content-disposition=attachment"
RUNS=/opt/rdd/runs
S3=s3://rdd-general-299284685700-us-east-1/runs
KEEP=/opt/rdd/KEEP_ALIVE

cd /opt/rdd/code
export PYTHONPATH=src PYTHONUNBUFFERED=1

if pgrep -f 'rdd.train.train_yolo' > /dev/null; then
  echo "a trainer is running here; US22 would contend with it for the GPU" >&2
  exit 1
fi

# The idle watchdog keeps its count in /var/tmp, which survives a stop/start,
# so a box stopped for being idle comes back at the threshold and stops again
# within a minute. Reset the count as well as setting the flag.
echo 0 > /var/tmp/rdd-idle-count 2>/dev/null || true
HELD=0
if [ ! -f "$KEEP" ]; then touch "$KEEP"; HELD=1; fi
trap '[ "$HELD" = 1 ] && rm -f "$KEEP"; echo "watchdog restored"' EXIT

if [ "$FETCH" = "1" ]; then
  echo "== fetching the dataset into its own directory  $(date -Is)"
  mkdir -p "$RAW"
  # Fetched directly rather than through the dataset-tools package, whose
  # download path needs Supervisely API credentials and otherwise reports a
  # zero-byte success. DS_URL is the signed asset link the Dataset Ninja page
  # serves; the Dropbox link that package ships now returns an HTML page.
  #
  # That host rejects HEAD with a 404 while serving GET normally, so probe it
  # with a ranged GET rather than curl -I. If the signature expires, re-scrape
  # the link from https://datasetninja.com/road-pothole-images
  #
  # There is no publisher hash for this archive, so size and completeness are
  # checked instead.
  TAR="$RAW/road-pothole-images-DatasetNinja.tar"

  # A single 23 GB transfer does not reliably survive, so resume rather than
  # restart. A completed file makes the server answer 416, which curl reports
  # as an error even though it means there is nothing left to fetch.
  ATTEMPT=0
  while : ; do
    if curl -sS -L --fail -C - --retry 5 --retry-delay 10 --retry-all-errors \
            -o "$TAR" "$DS_URL"; then
      echo "   transfer complete"
      break
    fi
    RC=$?
    SZ=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
    # 33 = server will not accept a range request; 36 = bad download resume.
    # Combined with a plausible size, both mean there is nothing left to fetch.
    if [ "$RC" = "33" ] || [ "$RC" = "36" ]; then
      echo "   server refused a further range; treating $SZ bytes as complete"
      break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "   interrupted (curl $RC) at $(numfmt --to=iec "$SZ" 2>/dev/null || echo "$SZ")"
    if [ "$ATTEMPT" -ge 20 ]; then
      echo "gave up after $ATTEMPT resume attempts" >&2
      exit 1
    fi
    sleep 10
  done

  BYTES=$(stat -c%s "$TAR")
  echo "   archive: $(numfmt --to=iec "$BYTES" 2>/dev/null || echo "$BYTES bytes")"
  if [ "$BYTES" -lt 100000000 ]; then
    echo "archive is only $BYTES bytes. That is a truncated transfer or an" >&2
    echo "error page served as a file, not the dataset." >&2
    head -c 200 "$TAR" >&2; echo >&2
    exit 1
  fi

  # Byte size cannot distinguish a complete archive from a truncated one.
  # Reading the whole tar index is the actual test and is cheap on local disk.
  echo "   verifying the archive is complete"
  if ! tar -tf "$TAR" > /dev/null 2>"$RAW/tar-check.err"; then
    echo "the archive is truncated or corrupt:" >&2
    tail -3 "$RAW/tar-check.err" >&2
    echo "Delete $TAR to start the download again, or re-run to resume it." >&2
    exit 1
  fi

  rm -rf "$RAW/x"; mkdir -p "$RAW/x"
  tar -xf "$TAR" -C "$RAW/x"
  echo "   unpacked:"
  du -sh "$RAW/x" || true

  # The publisher states 11,497 images. Checking here means a partial dataset
  # fails now rather than becoming a wrong AP figure later.
  FOUND=$(find "$RAW/x" -name "*.json" -path "*ann*" | wc -l)
  echo "   annotation files found: $FOUND (expected about 11497)"
  if [ "$FOUND" -lt 11000 ]; then
    echo "far fewer annotations than the published 11,497, the extract is" >&2
    echo "incomplete, so the evaluation would be computed over a subset" >&2
    echo "without saying so. Stopping." >&2
    exit 1
  fi
fi

echo
echo "== converting to a YOLO validation split  $(date -Is)"
rddpython -m rdd.data.stellenbosch_to_yolo \
  --src "$RAW" --out "$YOLO" --sample-negatives 40

# The false-positive rate is only meaningful if the unannotated images really
# contain no pothole. Code cannot settle that, so the sampled negatives are
# staged where they can be inspected, with a few annotated positives for
# comparison.
echo
echo "== staging a review set for the negative-image assumption  $(date -Is)"
REVIEW=s3://rdd-general-299284685700-us-east-1/review/us22
rddpython - "$YOLO" <<'PYREVIEW'
import json, shutil, sys
from pathlib import Path

root = Path(sys.argv[1])
conv = json.loads((root / "conversion.json").read_text(encoding="utf-8"))
img = root / "images" / "val"
lbl = root / "labels" / "val"
out_neg = root / "review" / "negatives"
out_pos = root / "review" / "positives"
for d in (out_neg, out_pos):
    d.mkdir(parents=True, exist_ok=True)

for name in conv.get("negative_sample", []):
    src = img / name
    if src.exists():
        shutil.copy2(src, out_neg / name)

# A dozen annotated images, so the reviewer can see what this corpus counts as
# a pothole before judging what it counts as nothing.
n = 0
for p in sorted(img.iterdir()):
    lp = lbl / f"{p.stem}.txt"
    if lp.exists() and lp.read_text(encoding="utf-8").strip():
        shutil.copy2(p, out_pos / p.name)
        n += 1
        if n >= 12:
            break
print(f"staged {len(list(out_neg.iterdir()))} negatives and {n} positives")
PYREVIEW
aws s3 sync "$YOLO/review" "$REVIEW" --only-show-errors
echo "   review set: $REVIEW"
echo "   negatives/, the question is only 'can I see a pothole here?'"
echo "   positives/, what this corpus does annotate, for calibration"

echo
echo "== evaluating  $(date -Is)"
ARGS=()
add() { [ -s "$2" ] && ARGS+=(--variant "$1=$2") || echo "   (no $1 checkpoint yet)"; }
# The baseline is the comparison that matters: it is the model the project's
# headline numbers describe. The variants are included when present so the
# generalisation question can be asked of them too.
for pair in "baseline:tierA_base6_s0" "us17a:us17a_p2head_s0" \
            "us17b:us17b_strip_s0" "us17c:us17c_dysample_s0" \
            "us17d:us17d_combined_s0"; do
  nm="${pair%%:*}"; run="${pair##*:}"
  if [ ! -s "$RUNS/$run/weights/best.pt" ]; then
    aws s3 cp "$S3/$run/weights/best.pt" "$RUNS/$run/weights/best.pt" \
      --only-show-errors 2>/dev/null || true
  fi
  add "$nm" "$RUNS/$run/weights/best.pt"
done

# The false-positive pass runs over every negative image, which is the number
# that decides whether this detector is deployable rather than merely accurate.
rddpython -m rdd.eval.zeroshot "${ARGS[@]}" \
  --data "$YOLO/data.yaml" --imgsz 640 --batch 16 \
  --fp-pass --conf 0.25 \
  --out "$RUNS/us22_zeroshot.json"

aws s3 cp "$RUNS/us22_zeroshot.json" "$S3/us22_zeroshot.json" --only-show-errors
aws s3 cp "$YOLO/conversion.json" "$S3/us22_conversion.json" --only-show-errors
echo
echo "=== US22 done  $(date -Is) ==="
