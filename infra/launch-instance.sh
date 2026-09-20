#!/usr/bin/env bash
# =============================================================================
#  RDD project -- launch a training instance
# =============================================================================
#  Launches ONE GPU instance, tagged Project=RDD-General, with:
#    * the pinned Deep Learning AMI
#    * the rdd-ec2-profile instance profile (S3 access, no static keys)
#    * no inbound ports (shell access is via SSM Session Manager)
#    * an idle-GPU watchdog that stops the instance automatically
#
#  THIS COSTS MONEY. g4dn.xlarge is $0.526/hr on-demand in us-east-1.
#  The script prints the rate and asks for confirmation unless --yes is given.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

JOB_NAME="baseline"
ASSUME_YES=0
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: bash infra/launch-instance.sh [options]

  --job NAME        job tag and instance name suffix   (default: baseline)
  --type TYPE       instance type                      (default: $INSTANCE_TYPE)
  --disk GB         root volume size in GB             (default: $ROOT_VOLUME_GB)
  --idle MIN        idle minutes before auto-stop      (default: $IDLE_SHUTDOWN_MINUTES)
  --yes             skip the cost confirmation prompt
  --dry-run         print what would be launched, launch nothing
  -h, --help        this message

Cost reference (us-east-1, on-demand Linux):
  g4dn.xlarge   4 vCPU, 16 GB, 1x T4    \$0.526/hr   ~\$12.62/day if left running
  g4dn.2xlarge  8 vCPU, 32 GB, 1x T4    \$0.752/hr   ~\$18.05/day if left running
  + EBS gp3 storage \$0.08/GB-month (a ${ROOT_VOLUME_GB} GB volume is ~\$16/month
    and is billed while the instance is STOPPED as well as running)

After the run finishes:
  bash infra/stop-instance.sh --job NAME        # keeps the disk, stops compute billing
  bash infra/terminate-instance.sh --job NAME   # deletes the disk too
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --job)     JOB_NAME="$2"; shift 2 ;;
    --type)    INSTANCE_TYPE="$2"; shift 2 ;;
    --disk)    ROOT_VOLUME_GB="$2"; shift 2 ;;
    --idle)    IDLE_SHUTDOWN_MINUTES="$2"; shift 2 ;;
    --yes)     ASSUME_YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

INSTANCE_NAME="${NAME_PREFIX}-${JOB_NAME}"

# --- Guard: do not exceed the account's G/VT vCPU quota ----------------------
QUOTA=$(aws service-quotas get-service-quota --service-code ec2 \
        --quota-code L-DB2E81BA --region "$AWS_REGION" \
        --query 'Quota.Value' --output text 2>/dev/null || echo "8")
VCPUS=$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" \
        --region "$AWS_REGION" --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' \
        --output text)
RUNNING=$(aws ec2 describe-instances --region "$AWS_REGION" \
          --filters "Name=instance-state-name,Values=running,pending" \
          --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)].[InstanceType]' \
          --output text | wc -l)

cat <<PLAN

  Launch plan
  -----------
  Name              $INSTANCE_NAME
  Job tag           $JOB_NAME
  Instance type     $INSTANCE_TYPE   (${VCPUS} vCPU)
  AMI               $DLAMI_ID
  Root volume       ${ROOT_VOLUME_GB} GB gp3
  Region            $AWS_REGION
  IAM profile       $IAM_PROFILE
  Security group    $SG_NAME  (no inbound rules)
  Bucket            s3://$RDD_BUCKET
  Idle auto-stop    ${IDLE_SHUTDOWN_MINUTES} min of GPU inactivity
  G/VT vCPU quota   ${QUOTA%.*}  (${RUNNING} G-family instance(s) currently running)

PLAN

if [ "$DRY_RUN" = "1" ]; then
  echo "  --dry-run: nothing launched."
  exit 0
fi

if [ "$ASSUME_YES" != "1" ]; then
  printf '  This will start billing immediately. Type "launch" to continue: '
  read -r REPLY
  [ "$REPLY" = "launch" ] || { echo "  aborted."; exit 1; }
fi

# --- Render user-data --------------------------------------------------------
UD="$(mktemp)"
sed -e "s|__RDD_BUCKET__|$RDD_BUCKET|g" \
    -e "s|__AWS_REGION__|$AWS_REGION|g" \
    -e "s|__IDLE_SHUTDOWN_MINUTES__|$IDLE_SHUTDOWN_MINUTES|g" \
    "$HERE/userdata.sh" > "$UD"

SG_ID=$(aws ec2 describe-security-groups --region "$AWS_REGION" \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text)

echo "  launching..."
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$AWS_REGION" \
  --image-id "$DLAMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$IAM_PROFILE" \
  --user-data "$(fileurl "$UD")" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_VOLUME_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true,\"Encrypted\":true}}]" \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
  --instance-initiated-shutdown-behavior stop \
  --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=$PROJECT_TAG},{Key=Job,Value=$JOB_NAME},{Key=ManagedBy,Value=rdd-launch}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=$PROJECT_TAG}]" \
  --query 'Instances[0].InstanceId' --output text)

rm -f "$UD"

cat <<DONE

  Launched: $INSTANCE_ID  ($INSTANCE_NAME)

  First boot installs CUDA-side Python packages and takes roughly 4-6 minutes.
  Watch progress without logging in:
      aws ec2 get-console-output --instance-id $INSTANCE_ID --region $AWS_REGION --output text | tail -40

  Open a shell (no SSH key, no open ports):
      aws ssm start-session --target $INSTANCE_ID --region $AWS_REGION

  Check readiness once inside:
      cat /opt/rdd/READY

  Stop it when you are done -- compute billing continues until you do:
      bash infra/stop-instance.sh --job $JOB_NAME
DONE
