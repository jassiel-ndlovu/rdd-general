#!/usr/bin/env bash
# =============================================================================
#  Restart a US17 variant in place, on the instance already running it
# =============================================================================
#      bash infra/restart-us17.sh <instance-id> <variant>
#
#  Stops the trainer, removes the local run directory, takes the latest code
#  from S3 and starts the variant again. Only the local copy is removed;
#  anything already pushed to S3 stays, so copy it aside first if it is needed.
# =============================================================================
set -euo pipefail

ID="${1:-}"; VARIANT="${2:-}"
[ -n "$ID" ] && [ -n "$VARIANT" ] || { sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 1; }
case "$VARIANT" in a|b|c|d) ;; *) echo "variant must be a, b, c or d" >&2; exit 1 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

case "$VARIANT" in
  a) RUN=us17a_p2head_s0 ;;
  b) RUN=us17b_strip_s0 ;;
  c) RUN=us17c_dysample_s0 ;;
  d) RUN=us17d_combined_s0 ;;
esac

# Only state changes here; no sleep loops, because a payload that waits exceeds
# the SSM command-executed waiter and the call fails even though the work runs.
read -r -d '' PAYLOAD <<PL || true
set -x
pkill -f 'rdd.train.train_yolo' || true
pkill -f 'run-us17.sh' || true
pkill -f 'progress-daemon.sh' || true
rm -f /opt/rdd/logs/progress-daemon.pid /opt/rdd/KEEP_ALIVE
rm -rf /opt/rdd/runs/$RUN
aws s3 sync s3://$RDD_BUCKET/code/ /opt/rdd/code/ --only-show-errors
cd /opt/rdd/code
sed -i 's/\r\$//' infra/*.sh
nohup bash infra/run-us17.sh $VARIANT > /opt/rdd/logs/us17$VARIANT.log 2>&1 &
PL

printf '\n== Restarting variant %s on %s (run %s)\n' "${VARIANT^^}" "$ID" "$RUN"

PF="$(mktemp)"
# A real file, not process substitution: the CLI needs to seek the payload.
# Both python arguments go through nativepath, python here is the Windows
# interpreter and a Git Bash /tmp/... path means nothing to it.
printf '%s' "$PAYLOAD" > "$PF.cmd"
python -c "
import json,sys,io
body=io.open(sys.argv[1],encoding='utf-8').read()
io.open(sys.argv[2],'w',encoding='utf-8').write(json.dumps({'commands':[body]}))
" "$(nativepath "$PF.cmd")" "$(nativepath "$PF")"

CID=$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$ID" \
      --document-name AWS-RunShellScript \
      --parameters "file://$(nativepath "$PF")" \
      --query 'Command.CommandId' --output text)
rm -f "$PF" "$PF.cmd"
printf '  ok    sent %s\n\n  Confirm with:  bash infra/track.sh\n\n' "$CID"
