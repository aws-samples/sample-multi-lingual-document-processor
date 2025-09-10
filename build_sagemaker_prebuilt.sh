#!/bin/bash

# Build SageMaker-compatible pre-built Docker image and save to images folder
# This creates a portable Docker image that can be loaded on any system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parameters
IMAGE_NAME=${1:-"mdp-sagemaker"}
IMAGE_TAG=${2:-"latest"}
IMAGES_DIR=${3:-"images"}
DOCKERFILE=${4:-"Dockerfile.sagemaker"}

print_status "=== Building SageMaker Pre-built Docker Image ==="
print_status "Image Name: $IMAGE_NAME"
print_status "Image Tag: $IMAGE_TAG"
print_status "Images Directory: $IMAGES_DIR"
print_status "Dockerfile: $DOCKERFILE"

# Create images directory
print_status "Creating images directory..."
mkdir -p "$IMAGES_DIR"
print_success "Images directory created: $IMAGES_DIR"

# Verify required files exist
print_status "Verifying required files..."
REQUIRED_FILES=(
    "$DOCKERFILE"
    "src/"
    "processing_script_clean.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -e "$file" ]]; then
        print_error "Required file/directory not found: $file"
        exit 1
    fi
done
print_success "All required files found"

# Check Docker availability
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    print_error "Docker daemon is not running"
    exit 1
fi

# Build the Docker image for SageMaker (linux/amd64)
print_status "Building SageMaker-compatible Docker image..."
print_status "Platform: linux/amd64 (SageMaker compatible)"
print_status "Command: docker build --platform linux/amd64 -t $IMAGE_NAME:$IMAGE_TAG -f $DOCKERFILE ."

docker build --platform linux/amd64 -t "$IMAGE_NAME:$IMAGE_TAG" -f "$DOCKERFILE" .

# Verify the built image
print_status "Verifying built image..."
IMAGE_ARCH=$(docker inspect "$IMAGE_NAME:$IMAGE_TAG" --format='{{.Architecture}}')
IMAGE_OS=$(docker inspect "$IMAGE_NAME:$IMAGE_TAG" --format='{{.Os}}')
IMAGE_SIZE=$(docker images --format "table {{.Size}}" "$IMAGE_NAME:$IMAGE_TAG" | tail -n 1)

print_status "Built image details:"
echo "  Name: $IMAGE_NAME:$IMAGE_TAG"
echo "  OS: $IMAGE_OS"
echo "  Architecture: $IMAGE_ARCH"
echo "  Size: $IMAGE_SIZE"

# Verify architecture is correct for SageMaker
if [[ "$IMAGE_ARCH" == "amd64" && "$IMAGE_OS" == "linux" ]]; then
    print_success "✅ Image architecture is SageMaker compatible (linux/amd64)"
else
    print_error "❌ Image architecture ($IMAGE_OS/$IMAGE_ARCH) is not SageMaker compatible"
    print_status "SageMaker requires linux/amd64 architecture"
    exit 1
fi

# Test the image functionality
print_status "Testing image functionality..."
docker run --rm --platform linux/amd64 "$IMAGE_NAME:$IMAGE_TAG" python -c "
import sys
import platform
print(f'✅ Python version: {sys.version}')
print(f'✅ Platform: {platform.machine()}')
print(f'✅ Architecture: {platform.architecture()}')

# Test key imports
try:
    import boto3
    print('✅ boto3 imported successfully')
    import numpy
    print('✅ numpy imported successfully')
    import pandas
    print('✅ pandas imported successfully')
    import cv2
    print('✅ opencv imported successfully')
    import PIL
    print('✅ Pillow imported successfully')
    print('✅ All key dependencies verified')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

print_success "Image functionality test passed"

# Save the image to tar file
IMAGE_FILE="$IMAGES_DIR/${IMAGE_NAME}_${IMAGE_TAG}_sagemaker_amd64.tar"
print_status "Saving image to file: $IMAGE_FILE"
docker save "$IMAGE_NAME:$IMAGE_TAG" -o "$IMAGE_FILE"

# Compress the image file
print_status "Compressing image file..."
COMPRESSED_FILE="${IMAGE_FILE}.gz"
gzip "$IMAGE_FILE"
print_success "Image compressed: $COMPRESSED_FILE"

# Get file size
COMPRESSED_SIZE=$(du -h "$COMPRESSED_FILE" | cut -f1)
print_status "Compressed image size: $COMPRESSED_SIZE"

# Create image info file
INFO_FILE="$IMAGES_DIR/${IMAGE_NAME}_${IMAGE_TAG}_info.json"
print_status "Creating image info file: $INFO_FILE"

cat > "$INFO_FILE" << EOF
{
  "image_name": "$IMAGE_NAME",
  "image_tag": "$IMAGE_TAG",
  "architecture": "$IMAGE_ARCH",
  "os": "$IMAGE_OS",
  "platform": "linux/amd64",
  "sagemaker_compatible": true,
  "dockerfile": "$DOCKERFILE",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "compressed_file": "$(basename "$COMPRESSED_FILE")",
  "compressed_size": "$COMPRESSED_SIZE",
  "usage": {
    "load_command": "docker load -i $(basename "$COMPRESSED_FILE")",
    "run_command": "docker run --rm $IMAGE_NAME:$IMAGE_TAG",
    "tag_for_ecr": "docker tag $IMAGE_NAME:$IMAGE_TAG YOUR_ECR_URI:latest"
  }
}
EOF

print_success "Image info file created: $INFO_FILE"

# Create usage instructions
USAGE_FILE="$IMAGES_DIR/README.md"
print_status "Creating usage instructions: $USAGE_FILE"

cat > "$USAGE_FILE" << EOF
# SageMaker Pre-built Docker Images

This directory contains pre-built Docker images for the Multi-lingual Document Processor (MDP) that are compatible with AWS SageMaker.

## Available Images

- **${IMAGE_NAME}_${IMAGE_TAG}_sagemaker_amd64.tar.gz**: SageMaker-compatible image (linux/amd64)

## Usage Instructions

### 1. Load the Image

\`\`\`bash
# Extract and load the image
gunzip ${IMAGE_NAME}_${IMAGE_TAG}_sagemaker_amd64.tar.gz
docker load -i ${IMAGE_NAME}_${IMAGE_TAG}_sagemaker_amd64.tar
\`\`\`

### 2. Verify the Image

\`\`\`bash
# Check if image is loaded
docker images | grep $IMAGE_NAME

# Test the image
docker run --rm $IMAGE_NAME:$IMAGE_TAG python -c "import boto3, numpy, pandas; print('✅ All dependencies working')"
\`\`\`

### 3. Deploy to ECR

\`\`\`bash
# Tag for your ECR repository
docker tag $IMAGE_NAME:$IMAGE_TAG YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/YOUR_REPO:latest

# Push to ECR
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/YOUR_REPO:latest
\`\`\`

### 4. Use with Deployment Script

\`\`\`bash
# Use with the deployment script
./deploy_dynamic.sh $IMAGE_NAME:$IMAGE_TAG
\`\`\`

## Image Details

- **Architecture**: linux/amd64 (SageMaker compatible)
- **Base Image**: python:3.10-slim
- **Key Dependencies**: boto3, numpy, pandas, opencv, pillow, textract libraries
- **SageMaker Ready**: Yes
- **Build Date**: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Troubleshooting

If you encounter issues:

1. **Architecture Mismatch**: This image is built for linux/amd64 and should work on SageMaker
2. **Import Errors**: All dependencies are pre-installed and tested
3. **Size Issues**: The compressed image is optimized for deployment

For support, check the main project documentation.
EOF

print_success "Usage instructions created: $USAGE_FILE"

# Show directory contents
print_status "Images directory contents:"
ls -la "$IMAGES_DIR/"

print_success "=== SageMaker Pre-built Image Creation Completed ==="
print_status ""
print_status "Summary:"
echo "  ✅ Image built: $IMAGE_NAME:$IMAGE_TAG"
echo "  ✅ Architecture: $IMAGE_OS/$IMAGE_ARCH (SageMaker compatible)"
echo "  ✅ Compressed file: $COMPRESSED_FILE"
echo "  ✅ Size: $COMPRESSED_SIZE"
echo "  ✅ Info file: $INFO_FILE"
echo "  ✅ Usage guide: $USAGE_FILE"
print_status ""
print_status "Next steps:"
echo "  1. Load the image: gunzip $COMPRESSED_FILE && docker load -i ${COMPRESSED_FILE%.gz}"
#echo "  2. Deploy: ./deploy_dynamic.sh $IMAGE_NAME:$IMAGE_TAG"
echo "  2. Deploy: ./deploy_dynamic.sh"
echo "  3. Or push to ECR: ./build_amd64_ecr.sh YOUR_REPO latest us-east-1"
