#!/usr/bin/env bash
# =============================================================================
#  RDD project -- Tier A progress and completion-time estimate
# =============================================================================
#  Run this FROM YOUR LAPTOP at any time. Read-only; it cannot disturb training.
#
#      bash infra/progress.sh              # one snapshot
#      bash infra/progress.sh --watch      # refresh every 60s until Ctrl-C
#      bash infra/progress.sh --json       # machine-readable
#
#  Cost is computed from instance uptime x the published hourly rate, not from
#  Cost Explorer, which lags ~24h and reports nothing about a run in progress.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

WATCH=0
EXTRA=""
INTERVAL=60
while [ $# -gt 0 ]; do
  case "$1" in
    --watch)    WATCH=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --json)     EXTRA="--json"; shift ;;
    -h|--help)  sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

find_instance() {
  aws ec2 describe-instances --region "$AWS_REGION" \
    --filters "Name=tag:Project,Values=$PROJECT_TAG" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null
}

snapshot() {
  local iid="$1"
  local cmd
  cmd=$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$iid" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"cd /opt/rdd/code && aws s3 sync s3://$RDD_BUCKET/code/ /opt/rdd/code/ --only-show-errors >/dev/null 2>&1; PYTHONPATH=src rddpython -m rdd.verify.progress --runs /opt/rdd/runs $EXTRA\"]" \
        --query 'Command.CommandId' --output text)
  # Poll until the command lands.
  for _ in $(seq 1 30); do
    st=$(aws ssm get-command-invocation --region "$AWS_REGION" \
         --command-id "$cmd" --instance-id "$iid" \
         --query Status --output text 2>/dev/null || echo Pending)
    case "$st" in Success|Failed) break ;; esac
    sleep 3
  done
  aws ssm get-command-invocation --region "$AWS_REGION" \
    --command-id "$cmd" --instance-id "$iid" \
    --query 'StandardOutputContent' --output text 2>/dev/null
}

IID=$(find_instance)
if [ -z "$IID" ] || [ "$IID" = "None" ]; then
  cat <<EMPTY

  No running instance tagged Project=$PROJECT_TAG.

  Either Tier A has finished and the instance stopped itself, or it was
  stopped manually. Check what landed in S3:

      aws s3 ls s3://$RDD_BUCKET/runs/ --recursive | grep summary.json

EMPTY
  exit 0
fi

if [ "$WATCH" = "1" ]; then
  echo "watching $IID every ${INTERVAL}s -- Ctrl-C to stop"
  while true; do
    clear 2>/dev/null || true
    snapshot "$IID"
    echo
    echo "  (refreshing every ${INTERVAL}s -- Ctrl-C to stop)"
    sleep "$INTERVAL"
  done
else
  snapshot "$IID"
fi
