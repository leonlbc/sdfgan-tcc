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
shift 2>/dev/null || true
EXTRA_ARGS="$*"

# Map shorthand names to commands
case "$JOB" in
    validate) CMD="python3 -u validate.py $EXTRA_ARGS" ;;
    train)    CMD="python3 -u train.py $EXTRA_ARGS" ;;
    *)        CMD="$JOB $EXTRA_ARGS" ;;
esac

echo "=== Running on $PUBLIC_IP ==="
echo "  Command: $CMD"
echo "  Output saved to ~/sdfgan/run.log on remote (no live streaming)."
echo ""

# Reset idle watchdog, run job, reset again after completion.
# Output goes to run.log only (not stdout) to avoid flooding the caller's context.
$SSH "touch /tmp/sdfgan-heartbeat && cd ~/$REMOTE_DIR && $CMD > run.log 2>&1; touch /tmp/sdfgan-heartbeat"

echo ""
echo "=== Job finished ==="
echo "  Download results: bash download.sh"
