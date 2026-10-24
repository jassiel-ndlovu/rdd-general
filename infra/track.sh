#!/usr/bin/env bash
# =============================================================================
#  Track what is happening on AWS
# =============================================================================
#      bash infra/track.sh              # one report
#      bash infra/track.sh --watch      # refresh every 5 minutes
#      bash infra/track.sh --watch 60   # refresh every 60 seconds
#      bash infra/track.sh --log        # + the last meaningful log lines
#      bash infra/track.sh --cost       # + month-to-date spend
#
#  Read-only. Starts nothing, stops nothing, changes nothing.
#
#  Commands run on the instance through SSM Session Manager, because the
#  security group has no inbound ports at all and there is no SSH.
#
#  Four things here are workarounds for problems that cost real time, and are
#  commented where they appear rather than here: MSYS path rewriting, the
#  console codec against Ultralytics' emoji, process substitution not reaching
#  a native aws.exe, and progress bars written with carriage returns.
# =============================================================================
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-RDD-General}"
REMOTE_RUNS="${REMOTE_RUNS:-/opt/rdd/runs}"
REMOTE_LOGS="${REMOTE_LOGS:-/opt/rdd/logs}"

# Git Bash rewrites arguments that look like Unix paths before they reach a
# native .exe, so a remote path like /opt/rdd becomes C:/Program Files/Git/opt/rdd.
export MSYS_NO_PATHCONV=1
# Ultralytics prints emoji. On a cp1252 console that raises UnicodeEncodeError
# inside the AWS CLI and loses the entire report, not just the emoji.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

WATCH=0; INTERVAL=300; SHOW_LOG=0; SHOW_COST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=1; shift
             case "${1:-}" in ''|--*) ;; *) INTERVAL="$1"; shift ;; esac ;;
    --log)   SHOW_LOG=1; shift ;;
    --cost)  SHOW_COST=1; shift ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

hdr()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[0;32m%s\033[0m\n' "$*"; }
warn() { printf '  \033[0;33m%s\033[0m\n' "$*"; }
err()  { printf '  \033[0;31m%s\033[0m\n' "$*"; }

# --- the script that runs on the instance ------------------------------------
# A quoted heredoc, so nothing here is expanded locally. Placeholders in the
# form __NAME__ are substituted just before sending.
read -r -d '' REMOTE_MAIN <<'EOS'
RUNS=__RUNS__

echo "### gpu"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
           --format=csv,noheader 2>/dev/null || echo "no GPU reported"

echo "### procs"
FOUND=0
for p in train_yolo to_yolo_tiled revalidate tiled_predict run-tierB.sh run-tierA.sh; do
  if pgrep -f "$p" >/dev/null 2>&1; then echo "$p"; FOUND=1; fi
done
[ "$FOUND" = "0" ] && echo "(nothing running)"

echo "### runs"
R=$(ls -1t "$RUNS"/*/results.csv 2>/dev/null | head -1)
if [ -n "$R" ]; then
  D=$(dirname "$R")
  NAME=$(basename "$D")
  N=$(( $(wc -l < "$R") - 1 ))
  TOTAL=$(awk '/^epochs:/ {print $2; exit}' "$D/args.yaml" 2>/dev/null)
  [ -z "$TOTAL" ] && TOTAL="?"
  echo "run      : $NAME"
  echo "epochs   : $N / $TOTAL"
  awk -F, -v n="$N" -v tot="$TOTAL" '
    NR>1 { t=$2; e=$1; m=$8; m95=$9 }
    END {
      if (n > 0) {
        r = t / n
        printf "elapsed  : %.2f h\n", t/3600
        printf "perEpoch : %.0f s (%.1f min)\n", r, r/60
        if (tot ~ /^[0-9]+$/ && tot > n)
          printf "remaining: %d epochs, %.2f h\n", tot-n, r*(tot-n)/3600
        printf "latest   : epoch %d  mAP50 %.4f  mAP50-95 %.4f\n", e, m, m95
      }
    }' "$R"
  awk -F, 'NR>1 && $8>b { b=$8; be=$1 }
           END { if (b>0) printf "best     : mAP50 %.4f at epoch %d\n", b, be }' "$R"
else
  echo "(no results.csv yet)"
fi

echo "### housekeeping"
df -h / | tail -1 | awk '{print "disk     : " $4 " free, " $5 " used"}'
if [ -f /opt/rdd/KEEP_ALIVE ]; then
  echo "watchdog : HELD OFF (KEEP_ALIVE present)"
else
  echo "watchdog : armed"
fi
EOS

read -r -d '' REMOTE_LOG <<'EOS'
L=$(ls -1t __LOGS__/*.log 2>/dev/null | head -1)
if [ -z "$L" ]; then echo "no logs"; exit 0; fi
echo "($L)"
# Ultralytics draws progress bars with carriage returns. Without the tr, a
# tail of this file is thousands of redrawn progress lines and nothing useful.
tr '\r' '\n' < "$L" \
  | grep -aE '^=== |^## |AP50=|^epochs |Traceback|[Ee]rror|^wrote |PASSED|FAILED' \
  | tr -cd '\11\12\15\40-\176' \
  | tail -20
EOS

check_session() {
  if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1; then
    err "AWS session has expired."
    echo "  Run:  aws login"
    echo "  (it opens a browser; nothing here can do that for you)"
    exit 1
  fi
}

find_instance() {
  aws ec2 describe-instances --region "$AWS_REGION" \
    --filters "Name=tag:Project,Values=$PROJECT_TAG" \
              "Name=instance-state-name,Values=running,pending,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType]' \
    --output text 2>/dev/null | head -1
}

# Send a script to the instance and print its stdout.
# The payload goes via a real file: process substitution (file://<(...)) does
# not work here, because aws.exe is a native binary and cannot see /dev/fd.
remote() {
  local iid="$1" body="$2"
  local bodyf paramsf cid winbody winparams
  bodyf="$(mktemp)"; paramsf="$(mktemp)"
  printf '%s' "$body" > "$bodyf"

  # MSYS_NO_PATHCONV, set above so that remote paths survive the trip to
  # aws.exe, also stops Git Bash converting /tmp/... for python.exe -- which is
  # equally a native binary and cannot open a POSIX path. Convert explicitly
  # for the native tools; the shell keeps using the POSIX form.
  winbody="$(cygpath -m "$bodyf" 2>/dev/null || echo "$bodyf")"
  winparams="$(cygpath -m "$paramsf" 2>/dev/null || echo "$paramsf")"

  python -c "
import json, sys
body = open(sys.argv[1], encoding='utf-8').read()
open(sys.argv[2], 'w', encoding='utf-8').write(
    json.dumps({'commands': [body], 'executionTimeout': ['300']}))
" "$winbody" "$winparams" \
    || { rm -f "$bodyf" "$paramsf"; err "could not build the request"; return 1; }

  cid=$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$iid" \
        --document-name AWS-RunShellScript --timeout-seconds 300 \
        --parameters "file://$winparams" \
        --query 'Command.CommandId' --output text 2>/dev/null)
  rm -f "$bodyf" "$paramsf"
  [ -n "$cid" ] || { err "send-command failed (is SSM registered yet?)"; return 1; }

  # The waiter blocks server-side rather than sleeping here. It gives up after
  # about 100 seconds, which is why the payloads above only ever read state.
  aws ssm wait command-executed --region "$AWS_REGION" \
      --command-id "$cid" --instance-id "$iid" 2>/dev/null
  aws ssm get-command-invocation --region "$AWS_REGION" \
      --command-id "$cid" --instance-id "$iid" \
      --query 'StandardOutputContent' --output text 2>/dev/null
}

section() { echo "$1" | sed -n "/^### $2\$/,/^###/p" | grep -v '^###' | sed 's/^/  /'; }

report() {
  printf '\033[1m%s\033[0m\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

  hdr "Instance"
  local info iid state itype out payload
  info="$(find_instance)"
  if [ -z "$info" ]; then
    warn "no instance tagged Project=$PROJECT_TAG"
    return 0
  fi
  iid=$(echo "$info" | cut -f1)
  state=$(echo "$info" | cut -f2)
  itype=$(echo "$info" | cut -f3)
  echo "  $iid   $itype   $state"
  if [ "$state" != "running" ]; then
    warn "not running -- nothing to track"
    [ "$state" = "stopped" ] && echo "  note: the EBS volume bills while stopped"
    return 0
  fi

  payload="${REMOTE_MAIN//__RUNS__/$REMOTE_RUNS}"
  out="$(remote "$iid" "$payload")" || return 1

  hdr "GPU";          section "$out" gpu
  hdr "Running";      section "$out" procs
  hdr "Training";     section "$out" runs
  hdr "Housekeeping"; section "$out" housekeeping

  if [ "$SHOW_LOG" = "1" ]; then
    hdr "Log"
    remote "$iid" "${REMOTE_LOG//__LOGS__/$REMOTE_LOGS}" | sed 's/^/  /'
  fi

  if [ "$SHOW_COST" = "1" ]; then
    hdr "Spend, month to date (before credit)"
    local today
    today="$(date -u +%Y-%m-%d)"
    aws ce get-cost-and-usage \
      --time-period "Start=$(date -u +%Y-%m-01),End=$today" \
      --granularity MONTHLY --metrics UnblendedCost \
      --filter '{"Not":{"Dimensions":{"Key":"RECORD_TYPE","Values":["Credit"]}}}' \
      --group-by Type=DIMENSION,Key=SERVICE \
      --query 'ResultsByTime[].Groups[].[Keys[0],Metrics.UnblendedCost.Amount]' \
      --output text 2>/dev/null \
      | awk '$2+0 > 0 { printf "  %-42s $%.2f\n", $1, $2 }' \
      || warn "Cost Explorer unavailable"
  fi
}

check_session

if [ "$WATCH" = "1" ]; then
  echo "tracking every ${INTERVAL}s -- Ctrl-C to stop"
  while true; do
    clear 2>/dev/null || true
    report
    echo
    echo "(refreshing in ${INTERVAL}s)"
    sleep "$INTERVAL"
  done
else
  report
fi
