#!/usr/bin/env bash
# Shared configuration for all AWS scripts.

REGION="sa-east-1"
INSTANCE_TYPE="g4dn.xlarge"
KEY_NAME="sdfgan-runner"
KEY_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sdfgan-runner.pem"
SG_NAME="sdfgan-runner-sg"
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/instance.env"

# S3 bucket for persistent dataset storage
S3_BUCKET="sdfgan-datasets-${REGION}"
S3_DATA_PREFIX="datasets"

# IAM role for instance S3 access
IAM_ROLE_NAME="sdfgan-runner-s3-role"
INSTANCE_PROFILE_NAME="sdfgan-runner-s3-profile"

# Deep Learning Base OSS (Nvidia Driver) AMI — Ubuntu 22.04
# Has NVIDIA drivers pre-installed; we pip-install PyTorch ourselves for simplicity.
AMI_QUERY_NAME="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
AMI_OWNER="amazon"

# Project paths (relative to repo root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="sdfgan"

# Auto-shutdown: terminate instance after this many idle minutes (no training running)
IDLE_TIMEOUT_MIN=60
