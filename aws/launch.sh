#!/usr/bin/env bash
# Launch a spot g4dn.xlarge instance in sa-east-1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== SDF-GAN AWS Launcher ==="
echo "Region:   $REGION"
echo "Instance: $INSTANCE_TYPE"

# --- Key pair ---
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating key pair '$KEY_NAME'..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query "KeyMaterial" \
        --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "  Saved to $KEY_FILE"
else
    echo "Key pair '$KEY_NAME' already exists."
    if [[ ! -f "$KEY_FILE" ]]; then
        echo "ERROR: Key pair exists in AWS but $KEY_FILE not found locally."
        echo "Either delete the key pair (bash cleanup.sh) and re-run, or restore the .pem file."
        exit 1
    fi
fi

# --- Security group ---
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null || true)

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    echo "Creating security group '$SG_NAME'..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "SSH access for SDF-GAN GPU runs" \
        --region "$REGION" \
        --query "GroupId" \
        --output text)

    # Allow SSH from anywhere (short-lived spot instance)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --region "$REGION" > /dev/null
    echo "  Security group: $SG_ID"
else
    echo "Security group '$SG_NAME' already exists: $SG_ID"
fi

# --- IAM role (for S3 access) ---
if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &>/dev/null; then
    echo "Creating IAM role '$IAM_ROLE_NAME'..."
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' > /dev/null

    # Grant read-only access to our specific bucket
    aws iam put-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-name "sdfgan-s3-read" \
        --policy-document "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Action\": [\"s3:GetObject\", \"s3:ListBucket\"],
                \"Resource\": [
                    \"arn:aws:s3:::$S3_BUCKET\",
                    \"arn:aws:s3:::$S3_BUCKET/*\"
                ]
            }]
        }"
    echo "  Role created with S3 read policy."
else
    echo "IAM role '$IAM_ROLE_NAME' already exists."
fi

if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" &>/dev/null; then
    echo "Creating instance profile..."
    aws iam create-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" > /dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" \
        --role-name "$IAM_ROLE_NAME"
    # IAM is eventually consistent — wait for propagation
    echo "  Waiting for IAM propagation..."
    sleep 10
else
    echo "Instance profile '$INSTANCE_PROFILE_NAME' already exists."
fi

# --- AMI ---
echo "Finding latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners "$AMI_OWNER" \
    --filters "Name=name,Values=$AMI_QUERY_NAME" "Name=state,Values=available" \
    --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
    --output text)

if [[ "$AMI_ID" == "None" || -z "$AMI_ID" ]]; then
    echo "ERROR: Could not find Deep Learning AMI in $REGION."
    echo "Falling back to Ubuntu 22.04 base AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners "099720109477" \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
        --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
        --output text)
fi
echo "  AMI: $AMI_ID"

# --- Spot request ---
echo "Requesting spot instance..."
SPOT_JSON=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=$INSTANCE_PROFILE_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":75,"VolumeType":"gp3"}}]' \
    --tag-specifications '[{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"sdfgan-runner"}]}]' \
    --instance-initiated-shutdown-behavior terminate \
    --query "Instances[0]" \
    --output json)

INSTANCE_ID=$(echo "$SPOT_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['InstanceId'])")
echo "  Instance: $INSTANCE_ID"
echo "  Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo "  Public IP: $PUBLIC_IP"

# --- Save state ---
cat > "$ENV_FILE" <<EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
REGION=$REGION
KEY_FILE=$KEY_FILE
EOF

echo ""
echo "=== Instance ready ==="
echo "  SSH: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "  Next: bash upload.sh"
