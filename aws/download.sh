#!/usr/bin/env bash
# Download results from the remote instance.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: No instance.env found. Run launch.sh first."
    exit 1
fi
source "$ENV_FILE"

SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=== Downloading results from $PUBLIC_IP ==="
$SCP "ubuntu@$PUBLIC_IP:~/$REMOTE_DIR/*.log" "$RESULTS_DIR/" 2>/dev/null || echo "  No .log files found."
$SCP "ubuntu@$PUBLIC_IP:~/$REMOTE_DIR/results.tsv" "$RESULTS_DIR/" 2>/dev/null || echo "  No results.tsv found."

echo "  Saved to $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"
