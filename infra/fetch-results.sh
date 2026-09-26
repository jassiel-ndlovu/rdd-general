#!/usr/bin/env bash
# =============================================================================
#  RDD project -- pull experiment results out of S3 onto this machine
# =============================================================================
#  Group members have no AWS access. This script assembles a self-contained
#  folder they can be handed directly -- weights, metrics, plots and the split
#  manifests -- with no cloud dependency.
#
#  RUN FROM YOUR LAPTOP:
#      bash infra/fetch-results.sh                      # default destination
#      bash infra/fetch-results.sh --dest /some/path
#      bash infra/fetch-results.sh --no-weights         # metrics and plots only
#
#  The per-country scratch trees are excluded. They are symlinked copies of
#  validation images that already exist under the dataset prefix, and they are
#  ~12 GB against ~0.4 GB of actual results.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

DEST="$(cd "$HERE/../.." && pwd)/Tier-A-Results"
WEIGHTS=1
RUNS=(tierA_base6_s0 tierA_base6_s1 tierA_base6_s2 tierA_ctrl4_s0)

while [ $# -gt 0 ]; do
  case "$1" in
    --dest)       DEST="$2"; shift 2 ;;
    --no-weights) WEIGHTS=0; shift ;;
    -h|--help)    sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m### %s\033[0m\n' "$*"; }
ok()  { printf '  \033[0;32mok\033[0m    %s\n' "$*"; }

aws sts get-caller-identity >/dev/null 2>&1 || {
  echo "error: AWS session expired. Re-authenticate, then run this again." >&2
  exit 1
}

mkdir -p "$DEST"/{metrics,weights,plots,splits}

# -----------------------------------------------------------------------------
say "Metrics"
# -----------------------------------------------------------------------------
aws s3 cp "s3://$RDD_BUCKET/runs/tierA_summary.json" \
          "$(nativepath "$DEST/metrics/aggregate.json")" --only-show-errors
ok "aggregate.json"

for r in "${RUNS[@]}"; do
  mkdir -p "$DEST/metrics/$r"
  for f in summary.json results.csv args.yaml; do
    aws s3 cp "s3://$RDD_BUCKET/runs/$r/$f" \
              "$(nativepath "$DEST/metrics/$r/$f")" --only-show-errors 2>/dev/null || true
  done
  # A flat copy too, so the four summaries sit side by side for easy diffing.
  cp "$DEST/metrics/$r/summary.json" "$DEST/metrics/$r.json" 2>/dev/null || true
  ok "$r"
done

# -----------------------------------------------------------------------------
say "Plots"
# -----------------------------------------------------------------------------
for r in "${RUNS[@]}"; do
  mkdir -p "$DEST/plots/$r"
  aws s3 sync "s3://$RDD_BUCKET/runs/$r/" "$(nativepath "$DEST/plots/$r")" \
    --only-show-errors \
    --exclude "*" --include "*.png" --include "*.jpg" \
    --exclude "*_per_country/*"
  ok "$r  ($(find "$DEST/plots/$r" -type f 2>/dev/null | wc -l) images)"
done

# -----------------------------------------------------------------------------
say "Split manifests and class map"
# -----------------------------------------------------------------------------
aws s3 cp "s3://$RDD_BUCKET/splits/rdd2022_core.json" \
          "$(nativepath "$DEST/splits/rdd2022_core.json")" --only-show-errors 2>/dev/null \
  && ok "split manifest" || echo "  (split manifest not found)"
aws s3 cp "s3://$RDD_BUCKET/yolo/data.yaml" \
          "$(nativepath "$DEST/splits/data.yaml")" --only-show-errors 2>/dev/null \
  && ok "data.yaml" || echo "  (data.yaml not found)"
aws s3 cp "s3://$RDD_BUCKET/coco/dedup_report.json" \
          "$(nativepath "$DEST/splits/dedup_report.json")" --only-show-errors 2>/dev/null \
  && ok "dedup report" || echo "  (dedup report not found)"
cp "$HERE/../configs/classmap.yaml" "$DEST/splits/classmap.yaml" 2>/dev/null \
  && ok "classmap.yaml"

# -----------------------------------------------------------------------------
if [ "$WEIGHTS" = "1" ]; then
say "Weights"
  for r in "${RUNS[@]}"; do
    for w in best.pt last.pt; do
      aws s3 cp "s3://$RDD_BUCKET/runs/$r/weights/$w" \
                "$(nativepath "$DEST/weights/${r}_${w}")" --only-show-errors 2>/dev/null \
        && ok "${r}_${w}  ($(du -h "$DEST/weights/${r}_${w}" 2>/dev/null | cut -f1))" || true
    done
  done
fi

# -----------------------------------------------------------------------------
say "Done"
# -----------------------------------------------------------------------------
echo "  destination : $DEST"
echo "  total size  : $(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo
echo "  This folder is self-contained. Share it directly -- it needs no AWS access."
