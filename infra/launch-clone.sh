#!/usr/bin/env bash
# =============================================================================
#  Launch a clone of the training instance and run one US17 variant on it
# =============================================================================
#      bash infra/launch-clone.sh <ami-id> <variant> [--spot]
#      bash infra/launch-clone.sh ami-0123 b --spot
#      bash infra/launch-clone.sh ami-0123 c --dry-run
#
#  Why a clone rather than launch-instance.sh. That script provisions a fresh
#  box from the Deep Learning AMI and installs packages on first boot, so two
#  instances launched a day apart can end up with different driver, torch or
#  ultralytics builds. For a comparison whose whole purpose is to attribute a
#  difference to the architecture, a drifting software environment is exactly
#  the confound to avoid. An image taken from the instance that trained the
#  baseline guarantees the same stack -- and brings the 26 GB extracted dataset
#  with it, so there is nothing to download.
#
#  Make the image first, with the source instance still running:
#      aws ec2 create-image --instance-id <id> --name rdd-us17-clone --no-reboot
#
#  Spot. Each training run checkpoints to S3 every epoch, so an interruption
#  costs one epoch. Spot is about 37% cheaper and is the sensible default for
#  a variant that can be restarted; use on-demand for anything on the critical
#  path of a deadline.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

AMI="${1:-}"; VARIANT="${2:-}"; shift 2 || true
SPOT=0; DRY=0
for a in "$@"; do
  case "$a" in
    --spot)    SPOT=1 ;;
    --dry-run) DRY=1 ;;
    *) die "unknown option: $a" ;;
  esac
done
[ -n "$AMI" ] && [ -n "$VARIANT" ] || { sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 1; }
case "$VARIANT" in a|b|c) ;; *) die "variant must be a, b or c" ;; esac

NAME="rdd-us17${VARIANT}"

# First boot: take the latest code, clear state the image inherited from the
# source instance, and start this variant. Everything else is already in place.
UD=$(cat <<EOF
#!/bin/bash
exec > /var/log/rdd-clone-boot.log 2>&1
set -x
# The image was taken from a running instance, so it carries that instance's
# daemon pidfile and a held watchdog. Both refer to processes that do not
# exist here.
rm -f /opt/rdd/logs/progress-daemon.pid /opt/rdd/KEEP_ALIVE
# Partial run directories from the source instance would otherwise be synced
# back to S3 over the real ones.
rm -rf /opt/rdd/runs/us17a_p2head_s0 /opt/rdd/runs/us17b_strip_s0 /opt/rdd/runs/us17c_dysample_s0
aws s3 sync s3://$RDD_BUCKET/code/ /opt/rdd/code/ --only-show-errors
cd /opt/rdd/code
sed -i 's/\r\$//' infra/*.sh
chown -R ubuntu:ubuntu /opt/rdd/code 2>/dev/null || true
nohup bash infra/run-us17.sh $VARIANT > /opt/rdd/logs/us17${VARIANT}.log 2>&1 &
EOF
)

SG_ID=$(aws ec2 describe-security-groups --region "$AWS_REGION" \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text)

say "Clone -> variant ${VARIANT^^}"
cat <<INFO
  image         $AMI
  instance type $INSTANCE_TYPE
  market        $([ "$SPOT" = 1 ] && echo "spot" || echo "on-demand")
  name          $NAME
  runs          infra/run-us17.sh $VARIANT, started from user-data
INFO

if [ "$DRY" = "1" ]; then ok "--dry-run: nothing launched"; exit 0; fi

ARGS=(
  --region "$AWS_REGION"
  --image-id "$AMI"
  --instance-type "$INSTANCE_TYPE"
  --security-group-ids "$SG_ID"
  --iam-instance-profile "Name=$IAM_PROFILE"
  --user-data "$UD"
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled"
  --tag-specifications
    "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$PROJECT_TAG},{Key=Job,Value=us17$VARIANT},{Key=ManagedBy,Value=rdd-clone}]"
    "ResourceType=volume,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$PROJECT_TAG}]"
  --query 'Instances[0].InstanceId' --output text
)
# Spot cannot take instance-initiated-shutdown-behavior=stop unless the request
# is persistent, so that flag is on-demand only.
if [ "$SPOT" = "1" ]; then
  ARGS+=(--instance-market-options '{"MarketType":"spot"}')
else
  ARGS+=(--instance-initiated-shutdown-behavior stop)
fi

IID=$(aws ec2 run-instances "${ARGS[@]}")
ok "launched $IID as $NAME"
cat <<DONE

  It boots, syncs the code and starts variant ${VARIANT^^} on its own. Give it
  two or three minutes before expecting to see anything.

      bash infra/track.sh            # all instances, including this one
      aws ec2 get-console-output --instance-id $IID --region $AWS_REGION --output text | tail -30

DONE
