#!/usr/bin/env bash
# =============================================================================
#  RDD project -- stop training instance(s)
# =============================================================================
#  Stops compute billing but KEEPS the EBS volume, so the filesystem, the
#  conda environment and any in-progress work survive. The volume continues to
#  bill at $0.08/GB-month while stopped.
#
#  Only ever touches instances tagged Project=RDD-General. Instances belonging
#  to other projects in this account are filtered out and cannot be hit.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

JOB_FILTER=""
[ "${1:-}" = "--job" ] && JOB_FILTER="$2"

FILTERS=( "Name=tag:Project,Values=$PROJECT_TAG"
          "Name=instance-state-name,Values=running,pending" )
[ -n "$JOB_FILTER" ] && FILTERS+=( "Name=tag:Job,Values=$JOB_FILTER" )

IDS=$(aws ec2 describe-instances --region "$AWS_REGION" \
      --filters "${FILTERS[@]}" \
      --query 'Reservations[].Instances[].InstanceId' --output text)

if [ -z "$IDS" ]; then
  echo "No running instances tagged Project=$PROJECT_TAG${JOB_FILTER:+ Job=$JOB_FILTER}."
  exit 0
fi

echo "Stopping (artefacts sync to S3 automatically on shutdown):"
for id in $IDS; do
  NAME=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$id" \
         --query 'Reservations[0].Instances[0].Tags[?Key==`Name`].Value|[0]' --output text)
  echo "  $id  $NAME"
done

aws ec2 stop-instances --region "$AWS_REGION" --instance-ids $IDS \
  --query 'StoppingInstances[].{Id:InstanceId,State:CurrentState.Name}' --output table

echo
echo "Compute billing ends once the state reaches 'stopped'."
echo "EBS volumes still bill. To remove those too: bash infra/terminate-instance.sh"
