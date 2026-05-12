#!/usr/bin/env bash
# Run a job on the remote instance.
# Usage:
#   bash run-job.sh validate          # runs python -u validate.py
#   bash run-job.sh train             # runs python -u train.py
#   bash run-job.sh "python my.py"    # runs arbitrary command
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: No instance.env found. Run launch.sh first."
    exit 1
fi
source "$ENV_FILE"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$PUBLIC_IP"

JOB="${1:-validate}"

# Map shorthand names to commands
case "$JOB" in
    validate) CMD="python3 -u validate.py" ;;
    train)    CMD="python3 -u train.py" ;;
    *)        CMD="$JOB" ;;
esac

echo "=== Running on $PUBLIC_IP ==="
echo "  Command: $CMD"
echo "  Output streams live. Also saved to ~/sdfgan/run.log on remote."
echo ""

# Reset idle watchdog, run job, reset again after completion
$SSH "touch /tmp/sdfgan-heartbeat && cd ~/$REMOTE_DIR && $CMD 2>&1 | tee run.log; touch /tmp/sdfgan-heartbeat"

echo ""
echo "=== Job finished ==="
echo "  Download results: bash download.sh"
