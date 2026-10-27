#!/bin/bash
# =============================================================================
#  Re-run US22 once the current chain on this box has finished
# =============================================================================
#      nohup bash infra/after-chain-us22.sh > /opt/rdd/logs/us22b.log 2>&1 &
#
#  Waits for the chain rather than racing it: the seed revalidation and US21
#  both need the GPU, and a second evaluation alongside them would measure
#  contention. Runs entirely on the instance, so nothing local needs to stay
#  alive for it to complete.
# =============================================================================
set -uo pipefail

CHAIN_LOG=/opt/rdd/logs/d-chain.log
KEEP=/opt/rdd/KEEP_ALIVE
BUCKET=rdd-general-299284685700-us-east-1

echo "waiting for the main chain to finish  $(date -Is)"
for i in $(seq 1 480); do          # up to 8 hours
  if grep -q CHAIN-DONE "$CHAIN_LOG" 2>/dev/null; then
    echo "chain finished after ~$((i)) minutes of waiting"
    break
  fi
  sleep 60
done

if ! grep -q CHAIN-DONE "$CHAIN_LOG" 2>/dev/null; then
  echo "the chain never reported CHAIN-DONE; not starting US22" >&2
  exit 1
fi

# Hold the watchdog across the whole of US22, and release it afterwards so the
# box can stop itself and stop costing money.
touch "$KEEP"
trap 'rm -f "$KEEP"; echo "watchdog released  $(date -Is)"' EXIT

cd /opt/rdd/code
echo
echo "== taking the latest code (the pip fix is newer than this box)"
aws s3 sync "s3://$BUCKET/code/" /opt/rdd/code/ --only-show-errors
sed -i 's/\r$//' infra/*.sh

echo
echo "== US22  $(date -Is)"
bash infra/run-us22.sh
echo "US22B-EXIT $?"
echo "US22B-DONE $(date -Is)"
