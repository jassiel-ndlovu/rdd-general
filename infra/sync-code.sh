#!/usr/bin/env bash
# =============================================================================
#  RDD project -- push local code to S3, where instances pick it up
# =============================================================================
#  Code travels laptop -> S3 -> instance. The instance never needs credentials
#  for your laptop, and a fresh instance always starts from the same artefact.
#
#  On the instance:  aws s3 sync s3://$RDD_BUCKET/code/ /opt/rdd/code/
#  (run automatically at first boot, and by 'rdd-pull' afterwards)
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
ROOT="$(cd "$HERE/.." && pwd)"

# Guard: a shell script with CRLF line endings fails on Linux with the
# baffling error `set: pipefail: invalid option name`, because bash reads the
# trailing \r as part of the option. Editing on Windows makes this easy to do
# by accident, so refuse to ship it.
BAD=""
for f in "$ROOT"/infra/*.sh; do
  [ -f "$f" ] || continue
  if grep -qU $'\r' "$f" 2>/dev/null; then BAD="$BAD $(basename "$f")"; fi
done
if [ -n "$BAD" ]; then
  echo "REFUSING TO SYNC -- these scripts have CRLF line endings:" >&2
  for b in $BAD; do echo "    infra/$b" >&2; done
  echo >&2
  echo "  Fix with:  sed -i 's/\\r\$//' infra/*.sh" >&2
  exit 1
fi

SRC="$(nativepath "$ROOT")"
echo "Pushing $SRC -> s3://$RDD_BUCKET/code/"
aws s3 sync "$SRC" "s3://$RDD_BUCKET/code/" \
  --exclude ".git/*" \
  --exclude "__pycache__/*" \
  --exclude "*/__pycache__/*" \
  --exclude "*.pyc" \
  --exclude ".venv/*" \
  --exclude "data/*" \
  --exclude "runs/*" \
  --exclude "*.pt" \
  --exclude "*.zip" \
  --delete \
  --only-show-errors

echo "Done. Manifest:"
aws s3 ls "s3://$RDD_BUCKET/code/" --recursive --summarize | tail -3
