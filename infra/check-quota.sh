#!/usr/bin/env bash
# =============================================================================
#  RDD project -- check the status of service quota requests
# =============================================================================
#  Read-only. Shows the current G/VT quotas and any pending increase requests.
#  Quota increases are requested through the AWS console; ask the project
#  manager for the project's standing request template.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

hdr() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# quota-code : human label
QUOTAS="L-DB2E81BA:G_and_VT_OnDemand L-3819A6DF:G_and_VT_Spot L-1216C47A:Standard_OnDemand"

hdr "Current EC2 quotas ($AWS_REGION)"
printf '  %-22s %-12s %s\n' "QUOTA" "VALUE" "CODE"
for q in $QUOTAS; do
  code="${q%%:*}"; label="${q##*:}"
  val=$(aws service-quotas get-service-quota --service-code ec2 --quota-code "$code" \
        --region "$AWS_REGION" --query 'Quota.Value' --output text 2>/dev/null || echo "?")
  printf '  %-22s %-12s %s\n' "$label" "${val%.*}" "$code"
done

hdr "Pending / recent increase requests"
FOUND=0
for q in $QUOTAS; do
  code="${q%%:*}"; label="${q##*:}"
  OUT=$(aws service-quotas list-requested-service-quota-change-history-by-quota \
        --service-code ec2 --quota-code "$code" --region "$AWS_REGION" \
        --query 'RequestedQuotas[].[Status,DesiredValue,Created,CaseId]' \
        --output text 2>/dev/null || true)
  if [ -n "$OUT" ]; then
    FOUND=1
    echo "  $label:"
    echo "$OUT" | while read -r status desired created caseid; do
      printf '    %-14s desired=%-6s created=%s  case=%s\n' \
        "$status" "${desired%.*}" "${created%%T*}" "${caseid:-none}"
    done
  fi
done
[ "$FOUND" = "0" ] && echo "  (no requests on record)"

hdr "Live Spot prices (what approval would buy)"
aws ec2 describe-spot-price-history \
  --instance-types g4dn.xlarge g4dn.2xlarge \
  --product-descriptions "Linux/UNIX" --region "$AWS_REGION" --max-items 8 \
  --query 'SpotPriceHistory[].{Type:InstanceType,AZ:AvailabilityZone,Spot:SpotPrice}' \
  --output table 2>/dev/null || echo "  (unavailable)"

echo
echo "  On-demand reference: g4dn.xlarge \$0.526/hr, g4dn.2xlarge \$0.752/hr"
echo
