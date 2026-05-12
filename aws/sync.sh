#!/usr/bin/env bash
# Sync train.py to the remote instance. Sub-second — only uploads the one file you edit.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: No instance.env found. Run launch.sh first."
    exit 1
fi
source "$ENV_FILE"

SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$PUBLIC_IP"

$SCP "$REPO_ROOT/train.py" "ubuntu@$PUBLIC_IP:~/$REMOTE_DIR/"
$SSH "touch /tmp/sdfgan-heartbeat"
echo "Synced train.py -> $PUBLIC_IP"
