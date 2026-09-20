#!/usr/bin/env bash
# =============================================================================
#  RDD project -- fetch source datasets into S3
# =============================================================================
#  RUN THIS ON THE EC2 INSTANCE, not on your laptop.
#
#  RDD2022 is a 13.3 GB download. From an EC2 instance in us-east-1 it lands in
#  roughly 10-20 minutes and the S3 upload is free and in-region. From a home
#  connection it can take hours and then has to be uploaded again.
#
#  Source of record
#  ----------------
#  The canonical URLs in the sekilab/RoadDamageDetector README
#  (mycityreport.s3... and bigdatacup.s3...) now return HTTP 403 AccessDenied,
#  verified 2026-09-19. The figshare record below is the working official
#  mirror, and is the one the ORDDC'2024 organisers link to.
#
#      RDD2022   doi:10.6084/m9.figshare.21431547
#      13,264,172,619 bytes, CC BY 4.0
#
#  Usage:
#      bash fetch-datasets.sh rdd2022
#      bash fetch-datasets.sh rdd2022 --verify-only
# =============================================================================
set -euo pipefail

RDD_BUCKET="${RDD_BUCKET:?set RDD_BUCKET (exported by the instance bootstrap)}"
WORK="${RDD_HOME:-/opt/rdd}/data"
mkdir -p "$WORK"

# The idle watchdog samples *GPU* utilisation, and everything this script does
# (download, unzip, S3 sync) is CPU-only. Without this guard the instance would
# stop itself mid-extraction after 45 minutes of apparent idleness. Released on
# exit, however the script ends.
KEEP="${RDD_HOME:-/opt/rdd}/KEEP_ALIVE"
touch "$KEEP"
trap 'rm -f "$KEEP"; echo "idle watchdog re-armed"' EXIT
echo "idle watchdog suspended for the duration of this script"

RDD2022_URL="https://ndownloader.figshare.com/files/38030910"
RDD2022_BYTES=13264172619
RDD2022_ZIP="$WORK/RDD2022.zip"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

fetch_rdd2022() {
  say "RDD2022 -- 13.3 GB from figshare"

  if [ -f "$RDD2022_ZIP" ] && [ "$(stat -c %s "$RDD2022_ZIP")" = "$RDD2022_BYTES" ]; then
    echo "    already on this disk at the correct size; not re-fetching"
  elif aws s3 ls "s3://$RDD_BUCKET/raw/rdd2022/RDD2022.zip" >/dev/null 2>&1; then
    echo "    already in S3; pulling from there instead (in-region, much faster)"
    aws s3 cp "s3://$RDD_BUCKET/raw/rdd2022/RDD2022.zip" "$RDD2022_ZIP"
  else
    echo "    downloading (resumable; re-run if it drops)"
    curl -L --fail --retry 5 --retry-delay 10 --continue-at - \
         -o "$RDD2022_ZIP" "$RDD2022_URL"

    ACTUAL=$(stat -c %s "$RDD2022_ZIP")
    if [ "$ACTUAL" != "$RDD2022_BYTES" ]; then
      echo "    SIZE MISMATCH: got $ACTUAL, expected $RDD2022_BYTES" >&2
      echo "    Delete the file and re-run; a truncated archive will fail to unzip." >&2
      exit 1
    fi
    echo "    size verified: $ACTUAL bytes"

    say "Archiving the pristine zip to S3 (so nobody re-downloads 13 GB)"
    aws s3 cp "$RDD2022_ZIP" "s3://$RDD_BUCKET/raw/rdd2022/RDD2022.zip" --only-show-errors
    sha256sum "$RDD2022_ZIP" | tee "$WORK/RDD2022.zip.sha256"
    aws s3 cp "$WORK/RDD2022.zip.sha256" "s3://$RDD_BUCKET/raw/rdd2022/" --only-show-errors
  fi

  # RDD2022.zip is a ZIP OF ZIPS: it holds seven per-country archives, not the
  # image tree. Extracting one level yields only Japan.zip, Norway.zip and so on.
  say "Extracting outer archive"
  mkdir -p "$WORK/rdd2022"
  unzip -q -o "$RDD2022_ZIP" -d "$WORK/rdd2022"

  INNER_DIR="$WORK/rdd2022/RDD2022"
  say "Extracting the seven country archives"
  for z in "$INNER_DIR"/*.zip; do
    [ -e "$z" ] || continue
    name=$(basename "$z" .zip)
    echo "    $name ($(du -h "$z" | cut -f1))"

    # Cache each country zip in S3 separately. Seven objects instead of 47,420,
    # so a later instance can pull one country in seconds rather than issuing
    # tens of thousands of GET requests.
    aws s3 cp "$z" "s3://$RDD_BUCKET/raw/rdd2022/countries/$name.zip" --only-show-errors

    unzip -q -o "$z" -d "$INNER_DIR"
    rm -f "$z"        # reclaim the space; the copy in S3 is the archive
  done

  say "Counting what actually arrived"
  IMGS=$(find "$INNER_DIR" -name "*.jpg" | wc -l)
  XMLS=$(find "$INNER_DIR" -name "*.xml" | wc -l)
  echo "    images : $IMGS   (expected 47420)"
  echo "    xmls   : $XMLS   (expected 38385)"
  if [ "$IMGS" != "47420" ] || [ "$XMLS" != "38385" ]; then
    echo "    COUNT MISMATCH -- do not train on this until it is explained" >&2
  else
    echo "    counts reconcile with Arya et al. (2024)"
  fi

  say "Per-country breakdown"
  for d in "$INNER_DIR"/*/; do
    c=$(basename "$d")
    printf '    %-18s train img %6s  train xml %6s  test img %6s\n' "$c" \
      "$(find "$d" -path "*train*" -name "*.jpg" | wc -l)" \
      "$(find "$d" -path "*train*" -name "*.xml" | wc -l)" \
      "$(find "$d" -path "*test*"  -name "*.jpg" | wc -l)"
  done

  echo
  echo "    The per-country zips are cached at s3://$RDD_BUCKET/raw/rdd2022/countries/"
  echo "    The extracted tree stays on this instance only; it is cheap to recreate."
}

fetch_metadata() {
  say "RDD2022 metadata (tiny -- safe to fetch anywhere)"
  mkdir -p "$WORK/meta"
  curl -sL -o "$WORK/meta/label_map.pbtxt"          "https://ndownloader.figshare.com/files/38030820"
  curl -sL -o "$WORK/meta/File_List.txt"            "https://ndownloader.figshare.com/files/38030826"
  curl -sL -o "$WORK/meta/Directory_Structure.txt"  "https://ndownloader.figshare.com/files/38030823"
  aws s3 sync "$WORK/meta" "s3://$RDD_BUCKET/raw/rdd2022/meta/" --only-show-errors
  echo "    label map:"; sed 's/^/      /' "$WORK/meta/label_map.pbtxt"
}

case "${1:-}" in
  rdd2022)  fetch_metadata; fetch_rdd2022 ;;
  metadata) fetch_metadata ;;
  *)
    cat <<USAGE
Usage: bash fetch-datasets.sh <target>

  metadata   label map, file list, directory structure  (~3 MB, any machine)
  rdd2022    the full 13.3 GB archive                   (run on EC2)

Note: RDD2022 alone supplies every class the project currently trains.
Earlier versions of this script warned that a second dataset was required
for the marking classes. That is no longer true -- see configs/classmap.yaml
for the label map actually in use, and do not substitute the one shipped
with the dataset.
USAGE
    exit 2 ;;
esac
