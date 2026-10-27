#!/usr/bin/env bash
# =============================================================================
#  Instance-side progress recorder
# =============================================================================
#  Started automatically by the run scripts. To start it by hand:
#      nohup bash infra/progress-daemon.sh > /dev/null 2>&1 &
#      bash infra/progress-daemon.sh --stop
#
#  Samples every SAMPLE_SECS and appends one JSON object per sample to
#  /opt/rdd/logs/progress.jsonl, keeping the newest in progress.json. Every
#  PUSH_EVERY samples it copies progress.json to S3.
#
#  Why a daemon rather than just polling from the laptop: infra/track.sh asks
#  the instance a question and prints the answer, so it only knows what is true
#  at the moment somebody runs it. A run is fifteen to twenty hours and nobody
#  watches all of it. This records the whole thing, so a stall at 03:00 is
#  visible afterwards instead of being inferred from a gap. Pushing to S3 also
#  means progress stays readable when SSM is slow or the local AWS session has
#  lapsed, which it has, repeatedly.
#
#  Deliberately cheap: one nvidia-smi, one df, a line count and a tail per
#  sample. The S3 listing is the only costly call and it runs once every
#  PUSH_EVERY samples, not every one.
# =============================================================================
set -uo pipefail

RUNS=/opt/rdd/runs
LOGS=/opt/rdd/logs
DATA=/opt/rdd/data
S3=s3://rdd-general-299284685700-us-east-1/runs
SAMPLE_SECS="${SAMPLE_SECS:-60}"
PUSH_EVERY="${PUSH_EVERY:-5}"
PIDFILE=/opt/rdd/logs/progress-daemon.pid
JSONL=$LOGS/progress.jsonl
LATEST=$LOGS/progress.json

if [ "${1:-}" = "--stop" ]; then
    [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && echo "stopped"
    rm -f "$PIDFILE"
    exit 0
fi

# One daemon at a time. A second would double every sample and make the
# seconds-per-epoch estimate meaningless.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running as $(cat "$PIDFILE")"
    exit 0
fi
mkdir -p "$LOGS"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

jstr() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

tick=0
while true; do
    tick=$((tick + 1))
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # --- phase -------------------------------------------------------------
    PHASE=idle
    pgrep -f "rdd.data.to_yolo_tiled" >/dev/null 2>&1 && PHASE=export
    pgrep -f "rdd.train.train_yolo"   >/dev/null 2>&1 && PHASE=training
    pgrep -f "rdd.eval.tiled_predict" >/dev/null 2>&1 && PHASE=evaluating
    pgrep -f "rdd.eval.revalidate"    >/dev/null 2>&1 && PHASE=revalidating

    # --- export progress ---------------------------------------------------
    # The exporter prints "N source images -> M tiles" as it goes; the most
    # recent line is the cheapest possible progress signal.
    EX_SRC=0; EX_TILES=0; EX_OUT=""
    if [ "$PHASE" = "export" ]; then
        EX_LOG=$(ls -1t "$LOGS"/tierb*.log 2>/dev/null | head -1)
        if [ -n "$EX_LOG" ]; then
            LINE=$(tr '\r' '\n' < "$EX_LOG" | grep -a "source images ->" | tail -1)
            EX_SRC=$(echo "$LINE"   | awk '{print $1}')
            EX_TILES=$(echo "$LINE" | awk '{print $4}')
        fi
        # Tiles actually on disk: the exporter's counter and the filesystem
        # should agree, and a divergence means something is failing to write.
        NEW=$(ls -1t -d "$DATA"/yolo/*/ 2>/dev/null | head -1)
        [ -n "$NEW" ] && EX_OUT=$(basename "${NEW%/}")
    fi

    # --- training progress -------------------------------------------------
    RUN=""; EPOCH=0; TOTAL=0; MAP50=0; MAP5095=0; SPE=0; ETA_H=0
    R=$(ls -1t "$RUNS"/*/results.csv 2>/dev/null | head -1)
    if [ -n "$R" ]; then
        D=$(dirname "$R"); RUN=$(basename "$D")
        EPOCH=$(( $(wc -l < "$R") - 1 ))
        TOTAL=$(awk '/^epochs:/ {print $2; exit}' "$D/args.yaml" 2>/dev/null)
        [ -z "$TOTAL" ] && TOTAL=0
        read -r MAP50 MAP5095 SPE ETA_H <<<"$(awk -F, -v n="$EPOCH" -v tot="$TOTAL" '
            NR>1 { t=$2; m=$8; m95=$9 }
            END {
              if (n>0) { r=t/n; printf "%.4f %.4f %.0f %.2f", m, m95, r,
                         (tot>n ? r*(tot-n)/3600 : 0) }
              else      { printf "0 0 0 0" }
            }' "$R")"
    fi

    # --- machine -----------------------------------------------------------
    read -r GPU_U GPU_M <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ',' | head -1)"
    GPU_U=${GPU_U:-0}; GPU_M=${GPU_M:-0}
    DISK_FREE=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
    LOAD=$(awk '{print $1}' /proc/loadavg)
    MEM_FREE=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)

    # --- upload ------------------------------------------------------------
    # The expensive call. Only on push ticks, and only for the current run.
    UP_OBJ=""; UP_MB=""
    if [ -n "$RUN" ] && [ $((tick % PUSH_EVERY)) -eq 0 ]; then
        SUM=$(aws s3 ls "$S3/$RUN/" --recursive --summarize 2>/dev/null | tail -2)
        UP_OBJ=$(echo "$SUM" | awk '/Total Objects/ {print $3}')
        UP_MB=$(echo "$SUM"  | awk '/Total Size/ {printf "%.1f", $3/1048576}')
    fi

    printf '{"t":"%s","phase":"%s","run":"%s","epoch":%s,"total":%s,' \
        "$NOW" "$PHASE" "$(jstr "$RUN")" "${EPOCH:-0}" "${TOTAL:-0}" > "$LATEST"
    printf '"mAP50":%s,"mAP50_95":%s,"sec_per_epoch":%s,"eta_h":%s,' \
        "${MAP50:-0}" "${MAP5095:-0}" "${SPE:-0}" "${ETA_H:-0}" >> "$LATEST"
    printf '"export_src":%s,"export_tiles":%s,"export_out":"%s",' \
        "${EX_SRC:-0}" "${EX_TILES:-0}" "$(jstr "$EX_OUT")" >> "$LATEST"
    printf '"gpu_util":%s,"gpu_mb":%s,"disk_free_gb":%s,"mem_free_mb":%s,"load":%s' \
        "${GPU_U:-0}" "${GPU_M:-0}" "${DISK_FREE:-0}" "${MEM_FREE:-0}" "${LOAD:-0}" >> "$LATEST"
    [ -n "$UP_OBJ" ] && printf ',"s3_objects":%s,"s3_mb":%s' "$UP_OBJ" "$UP_MB" >> "$LATEST"
    printf '}\n' >> "$LATEST"

    cat "$LATEST" >> "$JSONL"

    if [ $((tick % PUSH_EVERY)) -eq 0 ]; then
        aws s3 cp "$LATEST" "$S3/_progress/progress.json" --only-show-errors 2>/dev/null
        aws s3 cp "$JSONL"  "$S3/_progress/progress.jsonl" --only-show-errors 2>/dev/null
    fi

    sleep "$SAMPLE_SECS"
done
