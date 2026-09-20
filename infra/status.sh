#!/usr/bin/env bash
# =============================================================================
#  RDD project -- what is running, and what is it costing?
# =============================================================================
#  Read-only. Run this before you walk away, and again when you come back.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

hdr() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

hdr "RDD instances (Project=$PROJECT_TAG)"
aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:Project,Values=$PROJECT_TAG" \
            "Name=instance-state-name,Values=running,pending,stopping,stopped" \
  --query 'Reservations[].Instances[].{Id:InstanceId,Name:Tags[?Key==`Name`]|[0].Value,Type:InstanceType,State:State.Name,Since:LaunchTime}' \
  --output table

hdr "ALL GPU instances in the account (includes other projects)"
aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)||starts_with(InstanceType,`p`)].{Id:InstanceId,Project:Tags[?Key==`Project`]|[0].Value,Type:InstanceType,State:State.Name}' \
  --output table

hdr "EBS volumes (all projects -- these bill even when instances are stopped)"
aws ec2 describe-volumes --region "$AWS_REGION" \
  --query 'Volumes[].{Id:VolumeId,GB:Size,Type:VolumeType,State:State,Project:Tags[?Key==`Project`]|[0].Value}' \
  --output table

hdr "S3 usage: s3://$RDD_BUCKET"
aws s3 ls "s3://$RDD_BUCKET" --recursive --summarize 2>/dev/null \
  | tail -3 || echo "  (empty)"

hdr "Month-to-date spend, by service"
START=$(date -u +%Y-%m-01)
END=$(date -u +%Y-%m-%d)
if [ "$START" = "$END" ]; then END=$(date -u -d "+1 day" +%Y-%m-%d 2>/dev/null || echo "$END"); fi
aws ce get-cost-and-usage \
  --time-period "Start=$START,End=$END" \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[0].Groups[?Metrics.UnblendedCost.Amount!=`0`].{Service:Keys[0],USD:Metrics.UnblendedCost.Amount}' \
  --output table 2>/dev/null || echo "  (Cost Explorer not available)"

hdr "Budget"
aws budgets describe-budget --account-id "$AWS_ACCOUNT_ID" --budget-name "$BUDGET_NAME" \
  --query 'Budget.{Name:BudgetName,Limit:BudgetLimit.Amount,Spent:CalculatedSpend.ActualSpend.Amount,Forecast:CalculatedSpend.ForecastedSpend.Amount}' \
  --output table 2>/dev/null || echo "  (no budget)"

echo
