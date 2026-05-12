#!/usr/bin/env bash
# Upload code to the instance and pull datasets from S3.
# Run upload-data.sh first (once) to populate the S3 bucket.
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

echo "=== Setting up instance $PUBLIC_IP ==="

# Wait for SSH to be ready
echo "Waiting for SSH..."
for i in $(seq 1 30); do
    if $SSH "echo ok" &>/dev/null; then break; fi
    sleep 5
done

# Create remote directory
$SSH "mkdir -p ~/$REMOTE_DIR/datasets"

# Upload code files (small, always fresh)
echo "Uploading code..."
$SCP "$REPO_ROOT/prepare.py" \
     "$REPO_ROOT/train.py" \
     "$REPO_ROOT/validate.py" \
     "$REPO_ROOT/requirements.txt" \
     "ubuntu@$PUBLIC_IP:~/$REMOTE_DIR/"

# Pull datasets from S3 (fast, within AWS — skips existing files)
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

echo ""
echo "=== Instance ready ==="
echo "  Remote dir: ~/$REMOTE_DIR"
echo "  Next: bash run-job.sh validate"
