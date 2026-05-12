#!/usr/bin/env bash
# One-time session setup: upload code, pull data from S3, install PyTorch, start idle watchdog.
# Run ONCE after launch.sh. For subsequent experiments, use sync.sh instead.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: No instance.env found. Run launch.sh first."
    exit 1
fi
source "$ENV_FILE"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$PUBLIC_IP"
SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

echo "=== One-time setup on $PUBLIC_IP ==="

# Wait for SSH
echo "Waiting for SSH..."
for i in $(seq 1 30); do
    if $SSH "echo ok" &>/dev/null; then break; fi
    sleep 5
done

# Create remote directory
$SSH "mkdir -p ~/$REMOTE_DIR/datasets"

# Upload code files
echo "Uploading code..."
$SCP "$REPO_ROOT/prepare.py" \
     "$REPO_ROOT/train.py" \
     "$REPO_ROOT/validate.py" \
     "$REPO_ROOT/requirements.txt" \
     "ubuntu@$PUBLIC_IP:~/$REMOTE_DIR/"

# Pull datasets from S3 (fast within AWS, skips existing)
echo "Pulling datasets from s3://$S3_BUCKET/$S3_DATA_PREFIX/ ..."
$SSH "aws s3 sync s3://$S3_BUCKET/$S3_DATA_PREFIX/ ~/$REMOTE_DIR/datasets/ --region $REGION"

# Install dependencies
echo "Installing PyTorch + numpy..."
$SSH <<'REMOTE'
cd ~/sdfgan
pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip3 install --quiet numpy
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
REMOTE

# --- Idle watchdog (auto-terminates instance after IDLE_TIMEOUT_MIN of no activity) ---
echo "Installing idle watchdog (auto-shutdown after ${IDLE_TIMEOUT_MIN}m)..."
$SSH "sudo bash -c 'cat > /usr/local/bin/idle-watchdog.sh'" <<WATCHDOG
#!/bin/bash
HEARTBEAT=/tmp/sdfgan-heartbeat
MAX_IDLE=$IDLE_TIMEOUT_MIN

[ ! -f "\$HEARTBEAT" ] && { touch "\$HEARTBEAT"; exit 0; }

# Training in progress — reset heartbeat, skip
if pgrep -f "python.*(train|validate)" > /dev/null; then
    touch "\$HEARTBEAT"
    exit 0
fi

LAST=\$(stat -c %Y "\$HEARTBEAT")
NOW=\$(date +%s)
IDLE=\$(( (NOW - LAST) / 60 ))

if [ "\$IDLE" -ge "\$MAX_IDLE" ]; then
    echo "\$(date): Idle \${IDLE}m (limit: \${MAX_IDLE}m). Shutting down." >> /tmp/idle-watchdog.log
    shutdown -h now
fi
WATCHDOG

$SSH "sudo chmod +x /usr/local/bin/idle-watchdog.sh"
$SSH "sudo bash -c '(crontab -l 2>/dev/null | grep -v idle-watchdog; echo \"*/5 * * * * /usr/local/bin/idle-watchdog.sh >> /tmp/idle-watchdog.log 2>&1\") | crontab -'"
$SSH "touch /tmp/sdfgan-heartbeat"

echo ""
echo "=== Setup complete ==="
echo "  Remote dir: ~/$REMOTE_DIR"
echo "  Idle auto-shutdown: ${IDLE_TIMEOUT_MIN}m"
echo "  Next: bash sync.sh && bash run-job.sh train"
