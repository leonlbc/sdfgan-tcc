#!/usr/bin/env bash
# Terminate the running instance.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "No instance.env found. Nothing to terminate."
    exit 0
fi
source "$ENV_FILE"

echo "=== Terminating instance $INSTANCE_ID ==="
aws ec2 terminate-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query "TerminatingInstances[0].CurrentState.Name" \
    --output text

rm -f "$ENV_FILE"
echo "  instance.env removed."
echo "  Note: Security group and key pair are kept for reuse. Run cleanup.sh to remove them."
