#!/bin/bash

# Comprehensive CloudFormation Stack Cleanup Script
# Removes all resources created by deploy_dynamic.sh including:
# - S3 buckets (with all contents)
# - ECR repositories (with all images)
# - Bedrock Data Automation projects
# - IAM roles and policies
# - Lambda functions and layers
# - CloudFormation stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
AWS_REGION="us-east-1"
FORCE=false
STACK_NAME=""

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    echo "Usage: $0 [stack_name] [options]"
    echo ""
    echo "Options:"
    echo "  --list-stacks         List all MDP-related stacks"
    echo "  --force              Skip confirmation prompts"
    echo "  --region REGION      AWS region (default: us-east-1)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --list-stacks"
    echo "  $0 mdp-stack-user-abc123"
    echo "  $0 mdp-stack-user-abc123 --force"
    echo "  $0 mdp-stack-user-abc123 --region us-west-2"
    echo ""
    echo "What this script cleans up:"
    echo "  ✓ S3 buckets (all objects, versions, delete markers)"
    echo "  ✓ ECR repositories (all images)"
    echo "  ✓ Bedrock Data Automation projects"
    echo "  ✓ Lambda functions and layers"
    echo "  ✓ IAM roles and policies"
    echo "  ✓ API Gateway resources"
    echo "  ✓ CloudFormation stack"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --list-stacks)
            LIST_STACKS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
        *)
            if [[ -z "$STACK_NAME" ]]; then
                STACK_NAME="$1"
            else
                print_error "Multiple stack names provided: $STACK_NAME and $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check AWS CLI availability
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed or not in PATH"
    echo "Please install AWS CLI and try again"
    exit 1
fi

# Verify AWS credentials
if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1; then
    print_error "AWS credentials not configured or invalid for region $AWS_REGION"
    echo "Please run 'aws configure' and try again"
    exit 1
fi

list_mdp_stacks() {
    print_status "Listing MDP-related stacks in region: $AWS_REGION"
    
    # Get all stacks with various statuses
    STACKS=$(aws cloudformation list-stacks \
        --region "$AWS_REGION" \
        --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE CREATE_FAILED ROLLBACK_COMPLETE UPDATE_ROLLBACK_COMPLETE DELETE_FAILED \
        --query 'StackSummaries[?contains(StackName, `mdp-stack`)].{Name:StackName,Status:StackStatus,Created:CreationTime}' \
        --output table 2>/dev/null)
    
    if [[ -n "$STACKS" && "$STACKS" != *"None"* ]]; then
        echo "$STACKS"
    else
        print_warning "No MDP-related stacks found"
    fi
}

get_stack_outputs() {
    local stack_name="$1"
    
    # Get stack outputs
    aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs' \
        --output json 2>/dev/null || echo '[]'
}

get_stack_resources() {
    local stack_name="$1"
    
    # Get stack resources
    aws cloudformation describe-stack-resources \
        --stack-name "$stack_name" \
        --region "$AWS_REGION" \
        --query 'StackResources' \
        --output json 2>/dev/null || echo '[]'
}

empty_s3_bucket() {
    local bucket_name="$1"
    
    print_status "Emptying S3 bucket: $bucket_name"
    
    # Check if bucket exists
    if ! aws s3api head-bucket --bucket "$bucket_name" --region "$AWS_REGION" 2>/dev/null; then
        print_warning "Bucket $bucket_name does not exist or is not accessible"
        return 0
    fi
    
    # Remove all object versions and delete markers using AWS CLI
    print_status "Removing all object versions and delete markers..."
    
    # Get and delete all object versions
    aws s3api list-object-versions \
        --bucket "$bucket_name" \
        --region "$AWS_REGION" \
        --output text \
        --query 'Versions[].[Key,VersionId]' 2>/dev/null | \
    while read -r key version_id; do
        if [[ -n "$key" && -n "$version_id" ]]; then
            aws s3api delete-object --bucket "$bucket_name" --region "$AWS_REGION" --key "$key" --version-id "$version_id" >/dev/null 2>&1 || true
        fi
    done
    
    # Get and delete all delete markers
    aws s3api list-object-versions \
        --bucket "$bucket_name" \
        --region "$AWS_REGION" \
        --output text \
        --query 'DeleteMarkers[].[Key,VersionId]' 2>/dev/null | \
    while read -r key version_id; do
        if [[ -n "$key" && -n "$version_id" ]]; then
            aws s3api delete-object --bucket "$bucket_name" --region "$AWS_REGION" --key "$key" --version-id "$version_id" >/dev/null 2>&1 || true
        fi
    done
    
    # Remove any remaining objects (for non-versioned buckets)
    print_status "Removing remaining objects..."
    aws s3 rm "s3://$bucket_name" --recursive --region "$AWS_REGION" >/dev/null 2>&1 || true
    
    print_success "S3 bucket $bucket_name emptied successfully"
}

delete_ecr_repository() {
    local repo_name="$1"
    
    print_status "Deleting ECR repository: $repo_name"
    
    # Check if repository exists
    if ! aws ecr describe-repositories --repository-names "$repo_name" --region "$AWS_REGION" >/dev/null 2>&1; then
        print_warning "ECR repository $repo_name does not exist"
        return 0
    fi
    
    # Delete repository with all images
    if aws ecr delete-repository --repository-name "$repo_name" --region "$AWS_REGION" --force >/dev/null 2>&1; then
        print_success "ECR repository $repo_name deleted successfully"
    else
        print_error "Failed to delete ECR repository $repo_name"
        return 1
    fi
}

cleanup_lambda_layers() {
    local stack_name="$1"
    
    print_status "Cleaning up Lambda layers for stack: $stack_name"
    
    # Get all layers that match the stack pattern
    local layers
    layers=$(aws lambda list-layers --region "$AWS_REGION" --query "Layers[?contains(LayerName, '$stack_name')].LayerName" --output text 2>/dev/null || true)
    
    if [[ -n "$layers" ]]; then
        for layer_name in $layers; do
            print_status "Deleting layer: $layer_name"
            
            # Get all versions of the layer
            local versions
            versions=$(aws lambda list-layer-versions --layer-name "$layer_name" --region "$AWS_REGION" --query 'LayerVersions[].Version' --output text 2>/dev/null || true)
            
            for version in $versions; do
                aws lambda delete-layer-version --layer-name "$layer_name" --version-number "$version" --region "$AWS_REGION" >/dev/null 2>&1 || true
            done
            
            print_success "Layer $layer_name deleted"
        done
    else
        print_status "No Lambda layers found for cleanup"
    fi
}

cleanup_bedrock_projects() {
    local project_name="$1"
    
    if [[ -z "$project_name" ]]; then
        return 0
    fi
    
    print_status "Checking Bedrock Data Automation project: $project_name"
    
    # List projects to see if it exists
    local project_exists
    project_exists=$(aws bedrock list-data-automation-projects --region "$AWS_REGION" --query "projects[?projectName=='$project_name'].projectName" --output text 2>/dev/null || true)
    
    if [[ -n "$project_exists" ]]; then
        print_warning "Bedrock project $project_name exists - it will be deleted with the CloudFormation stack"
    else
        print_status "Bedrock project $project_name not found or already deleted"
    fi
}

delete_stack() {
    local stack_name="$1"
    
    print_status "Deleting CloudFormation stack: $stack_name"
    
    # Check if stack exists
    if ! aws cloudformation describe-stacks --stack-name "$stack_name" --region "$AWS_REGION" >/dev/null 2>&1; then
        print_warning "Stack $stack_name does not exist"
        return 0
    fi
    
    # Delete the stack
    if aws cloudformation delete-stack --stack-name "$stack_name" --region "$AWS_REGION" 2>/dev/null; then
        print_status "Stack deletion initiated. Waiting for completion..."
        
        # Wait for stack deletion with timeout
        local max_attempts=120  # 60 minutes (30 seconds * 120)
        local attempt=0
        
        while [[ $attempt -lt $max_attempts ]]; do
            local stack_status
            stack_status=$(aws cloudformation describe-stacks --stack-name "$stack_name" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "DELETE_COMPLETE")
            
            if [[ "$stack_status" == "DELETE_COMPLETE" ]]; then
                print_success "CloudFormation stack $stack_name deleted successfully"
                return 0
            elif [[ "$stack_status" == "DELETE_FAILED" ]]; then
                print_error "Stack deletion failed. Check AWS console for details."
                return 1
            fi
            
            echo -n "."
            sleep 30
            ((attempt++))
        done
        
        print_error "Stack deletion timed out after 60 minutes"
        return 1
    else
        print_error "Failed to initiate stack deletion"
        return 1
    fi
}

extract_json_value() {
    local json="$1"
    local key="$2"
    
    # Simple JSON extraction without jq dependency
    echo "$json" | grep -o "\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | sed "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/" | head -1
}

cleanup_stack() {
    local stack_name="$1"
    
    print_status "=== Starting cleanup for stack: $stack_name ==="
    print_status "Region: $AWS_REGION"
    echo ""
    
    # Check if stack exists
    if ! aws cloudformation describe-stacks --stack-name "$stack_name" --region "$AWS_REGION" >/dev/null 2>&1; then
        print_error "Stack $stack_name does not exist in region $AWS_REGION"
        return 1
    fi
    
    # Get stack status
    local stack_status
    stack_status=$(aws cloudformation describe-stacks --stack-name "$stack_name" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null)
    print_status "Stack status: $stack_status"
    
    # Get stack outputs and resources
    local outputs resources
    outputs=$(get_stack_outputs "$stack_name")
    resources=$(get_stack_resources "$stack_name")
    
    # Extract resource information using simple text processing
    local bucket_name ecr_repo_name bedrock_project_name
    
    # From outputs - extract BucketName
    bucket_name=$(echo "$outputs" | grep -A 3 '"OutputKey": "BucketName"' | grep '"OutputValue"' | sed 's/.*"OutputValue": "\([^"]*\)".*/\1/' | head -1)
    
    # Extract ECR repo from ECRImageUri output
    local ecr_uri
    ecr_uri=$(echo "$outputs" | grep -A 3 '"OutputKey": "ECRImageUri"' | grep '"OutputValue"' | sed 's/.*"OutputValue": "\([^"]*\)".*/\1/' | head -1)
    if [[ -n "$ecr_uri" && "$ecr_uri" != "null" ]]; then
        # Extract repo name from URI like: 123456789.dkr.ecr.us-east-1.amazonaws.com/repo-name:tag
        ecr_repo_name=$(echo "$ecr_uri" | sed 's|.*/||' | sed 's|:.*||')
    fi
    
    # From resources - get S3 bucket if not found in outputs
    if [[ -z "$bucket_name" || "$bucket_name" == "null" ]]; then
        bucket_name=$(echo "$resources" | grep -A 5 '"ResourceType": "AWS::S3::Bucket"' | grep '"PhysicalResourceId"' | sed 's/.*"PhysicalResourceId": "\([^"]*\)".*/\1/' | head -1)
    fi
    
    # From resources - get ECR repo if not found in outputs
    if [[ -z "$ecr_repo_name" || "$ecr_repo_name" == "null" ]]; then
        ecr_repo_name=$(echo "$resources" | grep -A 5 '"ResourceType": "AWS::ECR::Repository"' | grep '"PhysicalResourceId"' | sed 's/.*"PhysicalResourceId": "\([^"]*\)".*/\1/' | head -1)
    fi
    
    # Get Bedrock project name
    bedrock_project_name=$(echo "$resources" | grep -A 5 '"ResourceType": "AWS::Bedrock::DataAutomationProject"' | grep '"PhysicalResourceId"' | sed 's/.*"PhysicalResourceId": "\([^"]*\)".*/\1/' | head -1)
    
    # Show what will be cleaned up
    echo "📋 Resources to clean up:"
    echo "   S3 Bucket: ${bucket_name:-'None found'}"
    echo "   ECR Repository: ${ecr_repo_name:-'None found'}"
    echo "   Bedrock Project: ${bedrock_project_name:-'None found'}"
    echo "   Lambda Layers: Will scan for stack-related layers"
    echo "   CloudFormation Stack: $stack_name"
    echo ""
    
    # Confirmation
    if [[ "$FORCE" != "true" ]]; then
        echo -e "${YELLOW}⚠️  WARNING: This will permanently delete all resources and data!${NC}"
        read -p "Continue with cleanup? (yes/no): " confirm
        if [[ "$confirm" != "yes" && "$confirm" != "y" ]]; then
            print_status "Cleanup cancelled"
            return 0
        fi
    fi
    
    echo ""
    print_status "🚀 Starting resource cleanup..."
    
    local cleanup_success=true
    
    # 1. Empty S3 bucket
    if [[ -n "$bucket_name" && "$bucket_name" != "null" ]]; then
        if ! empty_s3_bucket "$bucket_name"; then
            cleanup_success=false
        fi
    fi
    
    # 2. Delete ECR repository
    if [[ -n "$ecr_repo_name" && "$ecr_repo_name" != "null" ]]; then
        if ! delete_ecr_repository "$ecr_repo_name"; then
            cleanup_success=false
        fi
    fi
    
    # 3. Clean up Lambda layers
    cleanup_lambda_layers "$stack_name"
    
    # 4. Check Bedrock projects
    if [[ -n "$bedrock_project_name" && "$bedrock_project_name" != "null" ]]; then
        cleanup_bedrock_projects "$bedrock_project_name"
    fi
    
    # 5. Delete CloudFormation stack
    if ! delete_stack "$stack_name"; then
        cleanup_success=false
    fi
    
    echo ""
    echo "============================================================"
    if [[ "$cleanup_success" == "true" ]]; then
        print_success "🎉 Cleanup completed successfully!"
        echo "All resources for stack '$stack_name' have been removed."
    else
        print_warning "⚠️  Cleanup completed with some errors"
        echo "Please check the output above and manually clean up any remaining resources."
    fi
    
    return 0
}

# Main execution
if [[ "$LIST_STACKS" == "true" ]]; then
    list_mdp_stacks
    exit 0
fi

if [[ -z "$STACK_NAME" ]]; then
    print_error "Stack name is required"
    print_usage
    exit 1
fi

cleanup_stack "$STACK_NAME"