#!/usr/bin/env bash
# Remove all AWS resources created by the scripts.
# Pass --keep-s3 to preserve the S3 bucket (avoid re-uploading 2.3 GB).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

KEEP_S3=false
[[ "${1:-}" == "--keep-s3" ]] && KEEP_S3=true

echo "=== Cleaning up AWS resources in $REGION ==="

# Terminate instance if still running
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
    echo "Terminating instance $INSTANCE_ID..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" > /dev/null 2>&1 || true
    echo "  Waiting for termination..."
    aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" --region "$REGION" 2>/dev/null || true
    rm -f "$ENV_FILE"
fi

# Delete security group
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null || true)

if [[ -n "$SG_ID" && "$SG_ID" != "None" ]]; then
    echo "Deleting security group $SG_ID..."
    aws ec2 delete-security-group --group-id "$SG_ID" --region "$REGION" 2>/dev/null || echo "  (may need to wait for instance termination)"
fi

# Delete key pair
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "Deleting key pair '$KEY_NAME'..."
    aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION"
fi
rm -f "$KEY_FILE"

# Delete IAM instance profile and role
if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" &>/dev/null; then
    echo "Removing instance profile '$INSTANCE_PROFILE_NAME'..."
    aws iam remove-role-from-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" \
        --role-name "$IAM_ROLE_NAME" 2>/dev/null || true
    aws iam delete-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME"
fi

if aws iam get-role --role-name "$IAM_ROLE_NAME" &>/dev/null; then
    echo "Deleting IAM role '$IAM_ROLE_NAME'..."
    aws iam delete-role-policy --role-name "$IAM_ROLE_NAME" --policy-name "sdfgan-s3-read" 2>/dev/null || true
    aws iam delete-role --role-name "$IAM_ROLE_NAME"
fi

# Delete S3 bucket
if [[ "$KEEP_S3" == true ]]; then
    echo "Keeping S3 bucket s3://$S3_BUCKET (--keep-s3)."
else
    if aws s3api head-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null; then
        echo "Deleting S3 bucket s3://$S3_BUCKET ..."
        aws s3 rm "s3://$S3_BUCKET" --recursive --region "$REGION"
        aws s3api delete-bucket --bucket "$S3_BUCKET" --region "$REGION"
    fi
fi

echo "=== Cleanup complete ==="
