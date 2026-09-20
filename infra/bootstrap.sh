#!/usr/bin/env bash
# =============================================================================
#  RDD project -- AWS foundation bootstrap
# =============================================================================
#  Creates the durable, near-zero-cost resources the project needs:
#    * S3 bucket           (dataset + artefact store)
#    * IAM role + profile  (so EC2 reaches S3 without static keys)
#    * Security group      (egress-only; access is via SSM Session Manager)
#    * Budget              (spend guardrail)
#
#  Idempotent: safe to re-run. Creates nothing that costs money while idle
#  except S3 storage (~$0.023/GB-month).
#
#  Usage:  bash infra/bootstrap.sh
#  Env:    RDD_ALERT_EMAIL=you@example.com  bash infra/bootstrap.sh   (adds budget alerts)
# =============================================================================
set -euo pipefail

# Stop Git Bash on Windows from mangling ARNs and paths into C:/... forms.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

# ----------------------------------------------------------------------------
# Configuration -- single source of truth, also read by the launch scripts
# ----------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[0;32m[ok]\033[0m %s\n' "$*"; }
skip() { printf '    \033[0;33m[--]\033[0m %s\n' "$*"; }

say "Identity check"
CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text)
echo "    caller : $CALLER_ARN"
echo "    account: $AWS_ACCOUNT_ID"
echo "    region : $AWS_REGION"
case "$CALLER_ARN" in
  *":root") printf '    \033[0;33m[warn]\033[0m Running as ROOT. Create a scoped IAM user instead.\n' ;;
esac

# ----------------------------------------------------------------------------
# 1. S3 bucket
# ----------------------------------------------------------------------------
say "S3 bucket: $RDD_BUCKET"
if aws s3api head-bucket --bucket "$RDD_BUCKET" 2>/dev/null; then
  skip "bucket already exists"
else
  # us-east-1 is the one region that must NOT be given a LocationConstraint.
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$RDD_BUCKET" --region "$AWS_REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$RDD_BUCKET" --region "$AWS_REGION" \
      --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
  fi
  ok "bucket created"
fi

aws s3api put-public-access-block --bucket "$RDD_BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" >/dev/null
ok "public access blocked"

aws s3api put-bucket-encryption --bucket "$RDD_BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}' >/dev/null
ok "default encryption (SSE-S3) on"

# Incomplete multipart uploads are invisible in the console but are billed.
# Large dataset uploads fail often enough that this matters.
aws s3api put-bucket-lifecycle-configuration --bucket "$RDD_BUCKET" \
  --lifecycle-configuration '{
    "Rules":[
      {"ID":"abort-incomplete-multipart","Status":"Enabled","Filter":{},
       "AbortIncompleteMultipartUpload":{"DaysAfterInitiation":7}},
      {"ID":"expire-scratch-after-30d","Status":"Enabled","Filter":{"Prefix":"scratch/"},
       "Expiration":{"Days":30}}
    ]}' >/dev/null
ok "lifecycle rules set (abort stale multipart; expire scratch/ after 30d)"

aws s3api put-bucket-tagging --bucket "$RDD_BUCKET" \
  --tagging "TagSet=[{Key=Project,Value=$PROJECT_TAG},{Key=ManagedBy,Value=rdd-bootstrap}]" >/dev/null
ok "tagged Project=$PROJECT_TAG"

# ----------------------------------------------------------------------------
# 2. IAM role + instance profile
# ----------------------------------------------------------------------------
say "IAM role: $IAM_ROLE"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if aws iam get-role --role-name "$IAM_ROLE" >/dev/null 2>&1; then
  skip "role already exists"
else
  aws iam create-role --role-name "$IAM_ROLE" \
    --assume-role-policy-document "$(fileurl "$HERE/policies/ec2-trust-policy.json")" \
    --description "EC2 role for the RDD project (S3 + logs, scoped to $RDD_BUCKET)" \
    --tags "Key=Project,Value=$PROJECT_TAG" >/dev/null
  ok "role created"
fi

# Render the bucket name into the policy template, then attach inline.
POLICY_RENDERED="$(mktemp)"
sed "s/RDD_BUCKET_NAME/$RDD_BUCKET/g" "$HERE/policies/rdd-instance-policy.json" > "$POLICY_RENDERED"
aws iam put-role-policy --role-name "$IAM_ROLE" \
  --policy-name "$IAM_POLICY_NAME" \
  --policy-document "$(fileurl "$POLICY_RENDERED")" >/dev/null
rm -f "$POLICY_RENDERED"
ok "inline policy '$IAM_POLICY_NAME' attached (scoped to $RDD_BUCKET only)"

# SSM Session Manager: browser/CLI shell without opening port 22 to the world.
aws iam attach-role-policy --role-name "$IAM_ROLE" \
  --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" >/dev/null
ok "AmazonSSMManagedInstanceCore attached (shell without inbound SSH)"

if aws iam get-instance-profile --instance-profile-name "$IAM_PROFILE" >/dev/null 2>&1; then
  skip "instance profile already exists"
else
  aws iam create-instance-profile --instance-profile-name "$IAM_PROFILE" \
    --tags "Key=Project,Value=$PROJECT_TAG" >/dev/null
  ok "instance profile created"
fi

if aws iam get-instance-profile --instance-profile-name "$IAM_PROFILE" \
     --query 'InstanceProfile.Roles[0].RoleName' --output text 2>/dev/null | grep -q "^$IAM_ROLE$"; then
  skip "role already in instance profile"
else
  aws iam add-role-to-instance-profile \
    --instance-profile-name "$IAM_PROFILE" --role-name "$IAM_ROLE" >/dev/null
  ok "role added to instance profile"
fi

# ----------------------------------------------------------------------------
# 3. Security group (egress only -- no inbound rules at all)
# ----------------------------------------------------------------------------
say "Security group: $SG_NAME"
VPC_ID=$(aws ec2 describe-vpcs --region "$AWS_REGION" \
          --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
echo "    default VPC: $VPC_ID"

SG_ID=$(aws ec2 describe-security-groups --region "$AWS_REGION" \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  SG_ID=$(aws ec2 create-security-group --region "$AWS_REGION" \
          --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
          --description "RDD training instances: egress only, access via SSM" \
          --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Name,Value=$SG_NAME}]" \
          --query GroupId --output text)
  # A new SG has no inbound rules by default -- that is exactly what we want.
  ok "created $SG_ID (no inbound rules; outbound allow-all)"
else
  skip "already exists: $SG_ID"
fi

# ----------------------------------------------------------------------------
# 4. Budget guardrail
# ----------------------------------------------------------------------------
say "Budget: $BUDGET_NAME (\$$BUDGET_LIMIT_USD/month)"
if aws budgets describe-budget --account-id "$AWS_ACCOUNT_ID" --budget-name "$BUDGET_NAME" >/dev/null 2>&1; then
  skip "budget already exists"
else
  BUDGET_JSON="$(mktemp)"
  cat > "$BUDGET_JSON" <<JSON
{
  "BudgetName": "$BUDGET_NAME",
  "BudgetLimit": { "Amount": "$BUDGET_LIMIT_USD", "Unit": "USD" },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostFilters": { "TagKeyValue": ["user:Project\$$PROJECT_TAG"] }
}
JSON
  if [ -n "${RDD_ALERT_EMAIL:-}" ]; then
    NOTIF="$(mktemp)"
    cat > "$NOTIF" <<JSON
[
  {"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":50,"ThresholdType":"PERCENTAGE"},
   "Subscribers":[{"SubscriptionType":"EMAIL","Address":"$RDD_ALERT_EMAIL"}]},
  {"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80,"ThresholdType":"PERCENTAGE"},
   "Subscribers":[{"SubscriptionType":"EMAIL","Address":"$RDD_ALERT_EMAIL"}]},
  {"Notification":{"NotificationType":"FORECASTED","ComparisonOperator":"GREATER_THAN","Threshold":100,"ThresholdType":"PERCENTAGE"},
   "Subscribers":[{"SubscriptionType":"EMAIL","Address":"$RDD_ALERT_EMAIL"}]}
]
JSON
    aws budgets create-budget --account-id "$AWS_ACCOUNT_ID" \
      --budget "$(fileurl "$BUDGET_JSON")" --notifications-with-subscribers "$(fileurl "$NOTIF")" >/dev/null
    rm -f "$NOTIF"
    ok "budget created with alerts at 50% / 80% actual and 100% forecast -> $RDD_ALERT_EMAIL"
  else
    aws budgets create-budget --account-id "$AWS_ACCOUNT_ID" --budget "$(fileurl "$BUDGET_JSON")" >/dev/null
    ok "budget created (tracking only)"
    printf '    \033[0;33m[warn]\033[0m No email alerts. Re-run with RDD_ALERT_EMAIL=you@example.com to add them.\n'
  fi
  rm -f "$BUDGET_JSON"
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
say "Done"
cat <<SUMMARY
    Bucket            s3://$RDD_BUCKET
    IAM role          $IAM_ROLE
    Instance profile  $IAM_PROFILE
    Security group    $SG_ID
    Budget            $BUDGET_NAME (\$$BUDGET_LIMIT_USD/month, filtered to Project=$PROJECT_TAG)

    Idle cost of the above: S3 storage only. No compute is running.
    Next: bash infra/launch-instance.sh --help
SUMMARY
