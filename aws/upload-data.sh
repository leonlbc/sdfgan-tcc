#!/usr/bin/env bash
# One-time: sync datasets to S3. Only needs to run once (or when data changes).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Syncing datasets to S3 ==="

# --- Create bucket if needed ---
if ! aws s3api head-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null; then
    echo "Creating bucket s3://$S3_BUCKET ..."
    aws s3api create-bucket \
        --bucket "$S3_BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" \
        > /dev/null
    echo "  Created."
else
    echo "Bucket s3://$S3_BUCKET already exists."
fi

# --- Sync data (only uploads new/changed files) ---
echo "Syncing datasets (2.3 GB)..."
aws s3 sync "$REPO_ROOT/datasets/" "s3://$S3_BUCKET/$S3_DATA_PREFIX/" --region "$REGION"

echo ""
echo "=== Data sync complete ==="
echo "  s3://$S3_BUCKET/$S3_DATA_PREFIX/"
aws s3 ls "s3://$S3_BUCKET/$S3_DATA_PREFIX/" --region "$REGION"
