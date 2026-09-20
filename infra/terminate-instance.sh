#!/usr/bin/env bash
# =============================================================================
#  RDD project -- terminate training instance(s)   [DESTRUCTIVE]
# =============================================================================
#  Deletes the instance AND its root EBS volume. Anything not synced to S3 is
#  lost permanently. Requires typing the instance id to confirm.
#
#  Only ever touches instances tagged Project=RDD-General.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

JOB_FILTER=""
[ "${1:-}" = "--job" ] && JOB_FILTER="$2"

FILTERS=( "Name=tag:Project,Values=$PROJECT_TAG"
          "Name=instance-state-name,Values=running,pending,stopped,stopping" )
[ -n "$JOB_FILTER" ] && FILTERS+=( "Name=tag:Job,Values=$JOB_FILTER" )

aws ec2 describe-instances --region "$AWS_REGION" --filters "${FILTERS[@]}" \
  --query 'Reservations[].Instances[].{Id:InstanceId,Name:Tags[?Key==`Name`]|[0].Value,State:State.Name,Type:InstanceType}' \
  --output table

IDS=$(aws ec2 describe-instances --region "$AWS_REGION" --filters "${FILTERS[@]}" \
      --query 'Reservations[].Instances[].InstanceId' --output text)

if [ -z "$IDS" ]; then
  echo "Nothing to terminate for Project=$PROJECT_TAG${JOB_FILTER:+ Job=$JOB_FILTER}."
  exit 0
fi

COUNT=$(echo "$IDS" | wc -w)
cat <<WARN

  *** DESTRUCTIVE ***
  This terminates $COUNT instance(s) and DELETES their root volumes.
  Any data on those volumes that has not been synced to
  s3://$RDD_BUCKET is lost permanently.

  Sync first if you are unsure:
      aws ssm start-session --target <id> --region $AWS_REGION
      rdd-push

WARN

printf '  Type TERMINATE to proceed: '
read -r REPLY
[ "$REPLY" = "TERMINATE" ] || { echo "  aborted."; exit 1; }

aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids $IDS \
  --query 'TerminatingInstances[].{Id:InstanceId,State:CurrentState.Name}' --output table

echo
echo "Terminated. S3 artefacts under s3://$RDD_BUCKET are unaffected."
