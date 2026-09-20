#!/usr/bin/env bash
# =============================================================================
#  RDD project -- shared infrastructure configuration
# =============================================================================
#  Sourced by every script in infra/. Edit here, not in the scripts.
# =============================================================================

# --- Account / region --------------------------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"

# --- Naming ------------------------------------------------------------------
# Everything this project creates is prefixed 'rdd-' and tagged Project=RDD-General.
# Assume the account hosts unrelated workloads: every script here filters strictly
# on that tag, and none of them enumerate or act on untagged resources.
# Never operate on resources you did not create.
PROJECT_TAG="RDD-General"
NAME_PREFIX="rdd"

RDD_BUCKET="${RDD_BUCKET:-${NAME_PREFIX}-general-${AWS_ACCOUNT_ID}-${AWS_REGION}}"
IAM_ROLE="${NAME_PREFIX}-ec2-role"
IAM_PROFILE="${NAME_PREFIX}-ec2-profile"
IAM_POLICY_NAME="${NAME_PREFIX}-s3-and-logs"
SG_NAME="${NAME_PREFIX}-training-sg"
BUDGET_NAME="${NAME_PREFIX}-monthly"
BUDGET_LIMIT_USD="${BUDGET_LIMIT_USD:-75}"

# --- Compute defaults --------------------------------------------------------
# g4dn.xlarge = 4 vCPU, 16 GB RAM, 1x NVIDIA T4 (16 GB VRAM). $0.526/hr on-demand.
# The account's G/VT on-demand quota is 8 vCPU => at most 2 of these at once.
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-200}"

# Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04), 2026-04-27.
# Pinned deliberately: an AMI that moves under you is a reproducibility hazard.
# Refresh with:  bash infra/find-latest-ami.sh
DLAMI_ID="${DLAMI_ID:-ami-012ba162b9cd2729c}"

# Auto-shutdown: the single most effective cost control for interactive GPU work.
# The instance halts itself after this many minutes with no GPU utilisation.
IDLE_SHUTDOWN_MINUTES="${IDLE_SHUTDOWN_MINUTES:-45}"

# --- Portability helper ------------------------------------------------------
# The AWS CLI on Windows is a native .exe, so a Git Bash path like
# /c/Users/... or /tmp/... is meaningless to it. fileurl() emits a file:// URL
# the CLI can actually open, on Git Bash, WSL, macOS and Linux alike.
fileurl() {
  local p="$1"
  if command -v cygpath >/dev/null 2>&1; then
    printf 'file://%s' "$(cygpath -m "$p")"
  else
    printf 'file://%s' "$p"
  fi
}

# Same problem for plain filesystem arguments, e.g. `aws s3 sync <dir> s3://...`.
nativepath() {
  local p="$1"
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$p"
  else
    printf '%s' "$p"
  fi
}

export -f fileurl nativepath
export AWS_REGION AWS_ACCOUNT_ID PROJECT_TAG NAME_PREFIX RDD_BUCKET \
       IAM_ROLE IAM_PROFILE IAM_POLICY_NAME SG_NAME BUDGET_NAME BUDGET_LIMIT_USD \
       INSTANCE_TYPE ROOT_VOLUME_GB DLAMI_ID IDLE_SHUTDOWN_MINUTES
