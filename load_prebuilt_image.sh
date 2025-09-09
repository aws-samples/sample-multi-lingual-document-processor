#!/bin/bash

# Load pre-built SageMaker Docker image from images folder
# This script loads a saved Docker image and prepares it for use

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

# Parameters
IMAGES_DIR=${1:-"images"}
IMAGE_FILE=${2:-""}
VERIFY_ONLY=${3:-"false"}

print_status "=== Loading Pre-built SageMaker Docker Image ==="
print_status "Images Directory: $IMAGES_DIR"

# Check if images directory exists
if [[ ! -d "$IMAGES_DIR" ]]; then
    print_error "Images directory not found: $IMAGES_DIR"
    print_status "Please run ./build_sagemaker_prebuilt.sh first to create pre-built images"
    exit 1
fi

# List available images
print_status "Available pre-built images:"
ls -la "$IMAGES_DIR/"

# Auto-detect image file if not specified
if [[ -z "$IMAGE_FILE" ]]; then
    # Look for compressed tar files
    COMPRESSED_FILES=($(find "$IMAGES_DIR" -name "*.tar.gz" -type f))
    
    if [[ ${#COMPRESSED_FILES[@]} -eq 0 ]]; then
        print_error "No compressed image files found in $IMAGES_DIR"
        print_status "Expected files: *.tar.gz"
        exit 1
    elif [[ ${#COMPRESSED_FILES[@]} -eq 1 ]]; then
        IMAGE_FILE="${COMPRESSED_FILES[0]}"
        print_status "Auto-detected image file: $IMAGE_FILE"
    else
        print_status "Multiple image files found:"
        for i in "${!COMPRESSED_FILES[@]}"; do
            echo "  $((i+1)). ${COMPRESSED_FILES[$i]}"
        done
        read -p "Select image file (1-${#COMPRESSED_FILES[@]}): " selection
        if [[ "$selection" =~ ^[0-9]+$ ]] && [[ "$selection" -ge 1 ]] && [[ "$selection" -le ${#COMPRESSED_FILES[@]} ]]; then
            IMAGE_FILE="${COMPRESSED_FILES[$((selection-1))]}"
            print_status "Selected: $IMAGE_FILE"
        else
            print_error "Invalid selection"
            exit 1
        fi
    fi
fi

# Verify image file exists
if [[ ! -f "$IMAGE_FILE" ]]; then
    print_error "Image file not found: $IMAGE_FILE"
    exit 1
fi

# Get file info
FILE_SIZE=$(du -h "$IMAGE_FILE" | cut -f1)
print_status "Image file: $IMAGE_FILE"
print_status "File size: $FILE_SIZE"

# Extract if compressed
EXTRACTED_FILE="$IMAGE_FILE"
if [[ "$IMAGE_FILE" == *.gz ]]; then
    EXTRACTED_FILE="${IMAGE_FILE%.gz}"
    
    if [[ ! -f "$EXTRACTED_FILE" ]]; then
        print_status "Extracting compressed image..."
        gunzip -k "$IMAGE_FILE"  # Keep original compressed file
        print_success "Image extracted: $EXTRACTED_FILE"
    else
        print_status "Extracted file already exists: $EXTRACTED_FILE"
    fi
fi

# Load the Docker image
print_status "Loading Docker image..."
docker load -i "$EXTRACTED_FILE"

# Get loaded image info
print_status "Detecting loaded image..."
LOADED_IMAGES=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep -E "(mdp|sagemaker)" | head -5)

if [[ -z "$LOADED_IMAGES" ]]; then
    print_error "Could not detect loaded MDP/SageMaker images"
    print_status "All available images:"
    docker images --format "table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}"
    exit 1
fi

print_success "Loaded images:"
echo "$LOADED_IMAGES"

# Get the first loaded image for verification
FIRST_IMAGE=$(echo "$LOADED_IMAGES" | head -1 | awk '{print $1}')
print_status "Using image for verification: $FIRST_IMAGE"

# Verify image architecture and functionality
if [[ "$VERIFY_ONLY" != "true" ]]; then
    print_status "Verifying image architecture and functionality..."
    
    # Check architecture
    ARCH=$(docker inspect "$FIRST_IMAGE" --format='{{.Architecture}}')
    OS=$(docker inspect "$FIRST_IMAGE" --format='{{.Os}}')
    
    print_status "Image details:"
    echo "  OS/Architecture: $OS/$ARCH"
    
    if [[ "$ARCH" == "amd64" && "$OS" == "linux" ]]; then
        print_success "✅ SageMaker compatible architecture (linux/amd64)"
    else
        print_error "❌ Not SageMaker compatible ($OS/$ARCH)"
    fi
    
    # Test functionality
    print_status "Testing image functionality..."
    docker run --rm --platform linux/amd64 "$FIRST_IMAGE" python -c "
import sys
import platform
print(f'Python: {sys.version.split()[0]}')
print(f'Platform: {platform.machine()}')

# Test imports
try:
    import boto3, numpy, pandas, cv2, PIL
    print('✅ All key dependencies working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    print_success "Image verification completed successfully"
fi

# Show usage instructions
print_status ""
print_success "=== Pre-built Image Loaded Successfully ==="
print_status ""
print_status "Available commands:"
echo "  # Test the image:"
echo "  docker run --rm $FIRST_IMAGE python -c \"print('Hello from SageMaker image!')\""
echo ""
echo "  # Deploy with the image:"
echo "  ./deploy_dynamic.sh $FIRST_IMAGE"
echo ""
echo "  # Push to ECR:"
echo "  docker tag $FIRST_IMAGE YOUR_ECR_URI:latest"
echo "  docker push YOUR_ECR_URI:latest"
echo ""
echo "  # Use with deployment workflow:"
echo "  ./deploy_with_prebuilt_image.sh YOUR_ECR_REPO latest us-east-1 amd64"

# Check if info file exists and show it
INFO_FILES=($(find "$IMAGES_DIR" -name "*_info.json" -type f))
if [[ ${#INFO_FILES[@]} -gt 0 ]]; then
    print_status ""
    print_status "Image information (from ${INFO_FILES[0]}):"
    if command -v jq >/dev/null 2>&1; then
        jq '.' "${INFO_FILES[0]}"
    else
        cat "${INFO_FILES[0]}"
    fi
fi