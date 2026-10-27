#!/usr/bin/env bash
# =============================================================================
#  Launch a clone of the training instance and run one US17 variant on it
# =============================================================================
#      bash infra/launch-clone.sh <ami-id> <variant> [--spot]
#      bash infra/launch-clone.sh ami-0123 b --spot --az us-east-1b
#      bash infra/launch-clone.sh ami-0123 d --dry-run
#
#  A clone rather than launch-instance.sh, which provisions a fresh box and
#  installs packages on first boot. Two boxes created a day apart can end up
#  with different driver, torch or ultralytics builds, which is a confound for
#  a comparison meant to attribute differences to architecture. An image of the
#  box that trained the baseline guarantees the same stack and carries the
#  extracted dataset with it.
#
#  Make the image first, with the source instance still running:
#      aws ec2 create-image --instance-id <id> --name rdd-us17-clone --no-reboot
#
#  Spot is about 37% cheaper and reasonable for a restartable run, since each
#  run checkpoints to S3 every epoch and run-us17.sh resumes from it. Use
#  on-demand on a deadline's critical path. Nothing replaces a reclaimed spot
#  instance: the request is one-time, and a persistent request risks a
#  stop/start loop against the idle watchdog.
#
#  Spot capacity varies sharply by availability zone, so --az is worth setting.
#  Check before launching:
#      aws ec2 get-spot-placement-scores --instance-types g4dn.xlarge \
#        --target-capacity 1 --target-capacity-unit-type units \
#        --region-names us-east-1 --single-availability-zone
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

# infra/config.sh carries settings and two path helpers, not output helpers --
# those live in the cluster's config, which is a different file.
say()  { printf '
== %s
' "$*"; }
ok()   { printf '  ok    %s
' "$*"; }
die()  { printf '  error %s
' "$*" >&2; exit 1; }

AMI="${1:-}"; VARIANT="${2:-}"; shift 2 || true
SPOT=0; DRY=0; AZ=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --spot)    SPOT=1 ;;
    --dry-run) DRY=1 ;;
    --az)      AZ="${2:-}"; [ -n "$AZ" ] || die "--az needs a zone"; shift ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done
[ -n "$AMI" ] && [ -n "$VARIANT" ] || { sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 1; }
case "$VARIANT" in a|b|c|d) ;; *) die "variant must be a, b, c or d" ;; esac

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
rm -rf /opt/rdd/runs/us17a_p2head_s0 /opt/rdd/runs/us17b_strip_s0 /opt/rdd/runs/us17c_dysample_s0 /opt/rdd/runs/us17d_combined_s0
# A volume restored from a snapshot hydrates lazily: the first touch of each
# block is a round trip to S3. Measured on the first pair of clones, the
# trainer's own reads crawled at about 8 MB/s and it sat in disk sleep with
# the GPU at 0% for the whole of what should have been the first epoch. The
# bottleneck is latency rather than bandwidth, so a parallel sweep at high
# queue depth pulls the blocks in far faster than the trainer reaches them
# one at a time. Backgrounded, so training still starts immediately.
nohup bash -c 'find /opt/rdd/data/yolo/core -type f -print0 | xargs -0 -P 32 -n 16 cat > /dev/null 2>&1; echo warmed-data \$(date -Is) >> /opt/rdd/logs/warm.log' > /dev/null 2>&1 &
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
  zone          ${AZ:-(let AWS choose)}
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
[ -n "$AZ" ] && ARGS+=(--placement "AvailabilityZone=$AZ")
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
