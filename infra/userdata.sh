#!/bin/bash
# =============================================================================
#  RDD project -- EC2 first-boot provisioning (cloud-init user-data)
# =============================================================================
#  Runs once, as root, on first boot of a training instance.
#  Placeholders __LIKE_THIS__ are substituted by infra/launch-instance.sh.
#
#  Progress is written to /var/log/rdd-bootstrap.log and mirrored to the
#  console log, which you can read without logging in:
#      aws ec2 get-console-output --instance-id i-... --output text
# =============================================================================
set -x
exec > >(tee -a /var/log/rdd-bootstrap.log | logger -t rdd-bootstrap -s 2>/dev/console) 2>&1

RDD_BUCKET="__RDD_BUCKET__"
AWS_REGION="__AWS_REGION__"
IDLE_MINUTES="__IDLE_SHUTDOWN_MINUTES__"
WORKDIR=/opt/rdd

echo "=== RDD bootstrap starting $(date -Is) ==="

# ----------------------------------------------------------------------------
# 1. Base packages
# ----------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git tmux htop tree jq unzip awscli python3-pip nvtop

# ----------------------------------------------------------------------------
# 2. Working directory on the (large) root volume
# ----------------------------------------------------------------------------
mkdir -p "$WORKDIR"/{code,data,runs,logs}
chown -R ubuntu:ubuntu "$WORKDIR"

# ----------------------------------------------------------------------------
# 3. Python environment
#    The DLAMI ships a conda env with CUDA-enabled PyTorch already built.
#    Install into it rather than creating a fresh venv, so we inherit the
#    vendor's CUDA/cuDNN/driver pairing instead of re-deriving it.
# ----------------------------------------------------------------------------
PY=$(ls -d /opt/pytorch/bin/python3 /opt/conda/envs/pytorch/bin/python3 2>/dev/null | head -1)
if [ -z "$PY" ]; then
    PY=$(command -v python3)
    echo "WARNING: DLAMI python not found, falling back to $PY"
fi
echo "Using interpreter: $PY"
"$PY" -m pip install --no-cache-dir --upgrade pip
"$PY" -m pip install --no-cache-dir \
    "ultralytics==8.4.131" \
    "pycocotools>=2.0.7" \
    "imagehash>=4.3.1" \
    "boto3>=1.34" \
    "pyyaml>=6.0" \
    "pandas>=2.0" \
    "tqdm>=4.66"

# Ultralytics phones home and auto-updates by default. Both are reproducibility
# hazards in a study that reports exact version numbers, so switch them off.
sudo -u ubuntu "$PY" -c "
from ultralytics import settings
settings.update({'sync': False, 'runs_dir': '$WORKDIR/runs', 'datasets_dir': '$WORKDIR/data'})
print('ultralytics settings pinned')
" || echo "WARNING: could not pin ultralytics settings"

# A WRAPPER, not a symlink. /opt/pytorch is a virtualenv, and Python locates its
# venv by looking for pyvenv.cfg next to sys.executable. A symlink resolves
# through to the base interpreter (/usr/local/bin/python3.12), which sits outside
# the venv, so the venv's site-packages -- including torch -- never load.
# exec'ing the real path keeps sys.executable inside the venv.
cat > /usr/local/bin/rddpython <<WRAP
#!/bin/sh
exec "$PY" "\$@"
WRAP
chmod +x /usr/local/bin/rddpython
"$PY" -c "import torch" && echo "rddpython wrapper verified: torch importable"

# ----------------------------------------------------------------------------
# 4. Idle auto-shutdown -- the main cost control
#    Stops (does not terminate) the instance after IDLE_MINUTES of GPU
#    inactivity, so the EBS volume and any in-progress work survive.
#    Create /opt/rdd/KEEP_ALIVE to suspend the watchdog.
# ----------------------------------------------------------------------------
cat > /usr/local/bin/rdd-idle-check <<'IDLE'
#!/bin/bash
# GPU considered busy above this utilisation percentage.
THRESHOLD=5
STATE=/var/tmp/rdd-idle-count
LIMIT="$1"           # consecutive idle samples before shutdown

[ -f /opt/rdd/KEEP_ALIVE ] && { echo 0 > "$STATE"; exit 0; }

UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
UTIL=${UTIL:-0}

if [ "$UTIL" -gt "$THRESHOLD" ]; then
    echo 0 > "$STATE"
    exit 0
fi

COUNT=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$COUNT" > "$STATE"
logger -t rdd-idle "GPU idle (${UTIL}%), ${COUNT}/${LIMIT} samples"

if [ "$COUNT" -ge "$LIMIT" ]; then
    logger -t rdd-idle "Idle limit reached -- stopping instance"
    /usr/local/bin/rdd-sync-out || true
    shutdown -h now
fi
IDLE
chmod +x /usr/local/bin/rdd-idle-check

# Flush results to S3 before the instance goes down, so a shutdown is never
# a data-loss event.
cat > /usr/local/bin/rdd-sync-out <<SYNC
#!/bin/bash
aws s3 sync /opt/rdd/runs "s3://$RDD_BUCKET/runs/\$(hostname)/" --only-show-errors || true
aws s3 sync /opt/rdd/logs "s3://$RDD_BUCKET/logs/\$(hostname)/" --only-show-errors || true
SYNC
chmod +x /usr/local/bin/rdd-sync-out

cat > /etc/systemd/system/rdd-idle.service <<UNIT
[Unit]
Description=RDD idle GPU watchdog
[Service]
Type=oneshot
ExecStart=/usr/local/bin/rdd-idle-check ${IDLE_MINUTES}
UNIT

cat > /etc/systemd/system/rdd-idle.timer <<UNIT
[Unit]
Description=Run the RDD idle watchdog every minute
[Timer]
OnBootSec=10min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now rdd-idle.timer

# Flush to S3 on any clean shutdown, however it was triggered.
cat > /etc/systemd/system/rdd-sync-on-stop.service <<UNIT
[Unit]
Description=Sync RDD artefacts to S3 on shutdown
DefaultDependencies=no
Before=shutdown.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/rdd-sync-out
RemainAfterExit=yes
[Install]
WantedBy=shutdown.target
UNIT
systemctl daemon-reload
systemctl enable rdd-sync-on-stop.service

# ----------------------------------------------------------------------------
# 5. Convenience shell environment for the ubuntu user
# ----------------------------------------------------------------------------
cat >> /home/ubuntu/.bashrc <<BRC

# ---- RDD project ----
export RDD_BUCKET="$RDD_BUCKET"
export RDD_HOME="$WORKDIR"
export AWS_DEFAULT_REGION="$AWS_REGION"
alias rdd-keepalive='touch $WORKDIR/KEEP_ALIVE && echo "idle shutdown SUSPENDED"'
alias rdd-autostop='rm -f $WORKDIR/KEEP_ALIVE && echo "idle shutdown ARMED"'
alias rdd-push='/usr/local/bin/rdd-sync-out && echo synced'
alias rdd-gpu='watch -n2 nvidia-smi'
cd \$RDD_HOME 2>/dev/null || true
BRC
chown ubuntu:ubuntu /home/ubuntu/.bashrc

# ----------------------------------------------------------------------------
# 6. Pull project code from S3 (if it has been pushed)
# ----------------------------------------------------------------------------
aws s3 sync "s3://$RDD_BUCKET/code/" "$WORKDIR/code/" --only-show-errors || \
    echo "no code/ prefix in S3 yet -- run infra/sync-code.sh from your laptop"
chown -R ubuntu:ubuntu "$WORKDIR"

# ----------------------------------------------------------------------------
# 7. Readiness marker
# ----------------------------------------------------------------------------
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
    | tee "$WORKDIR/READY"
date -Is >> "$WORKDIR/READY"
chown ubuntu:ubuntu "$WORKDIR/READY"

echo "=== RDD bootstrap complete $(date -Is) ==="
