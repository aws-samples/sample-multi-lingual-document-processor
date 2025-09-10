#!/bin/bash

# Dynamic Multi-lingual Document Processor Deployment Script
# Automatically generates unique names for all resources
# 
# Usage:
#   ./deploy_dynamic.sh [prebuilt_image_name] [force_rebuild]
#
# Examples:
#   ./deploy_dynamic.sh                                    # Auto-detect pre-built image
#   ./deploy_dynamic.sh mdp-processor:latest              # Use specific image
#   ./deploy_dynamic.sh mdp-processor:amd64               # Use AMD64 image
#   ./deploy_dynamic.sh "" true                           # Force rebuild (legacy mode)
#
# Prerequisites:
#   - Docker image must be pre-built using build_amd64_ecr.sh or build_multiarch_ecr.sh
#   - AWS CLI configured with appropriate permissions
#   - Docker daemon running

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

# Platform compatibility check function
check_platform_compatibility() {
    local arch=$(uname -m)
    local os=$(uname -s)
    
    print_status "Checking platform compatibility..."
    
    # Check Docker availability
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        echo "Please install Docker and try again"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        echo "Please start Docker and try again"
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed or not in PATH"
        echo "Please install AWS CLI and try again"
        exit 1
    fi
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        print_error "AWS credentials not configured or invalid"
        echo "Please run 'aws configure' and try again"
        exit 1
    fi
    
    print_success "Platform compatibility check passed"
    echo "  OS: $os"
    echo "  Architecture: $arch"
    echo "  Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
    echo "  AWS CLI: $(aws --version | cut -d' ' -f1 | cut -d'/' -f2)"
}

# Run compatibility check
check_platform_compatibility

# Parameters
PREBUILT_IMAGE_NAME=${1:-""}
FORCE_REBUILD=${2:-"false"}

# Generate unique identifiers
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RANDOM_ID=$(openssl rand -hex 4)
USER_ID=$(whoami | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

# Generate unique resource names
STACK_NAME="mdp-stack-${USER_ID}-${RANDOM_ID}"
BUCKET_NAME="mdp-bucket-${USER_ID}-${RANDOM_ID}"
ECR_REPO_NAME="mdp-processor-${USER_ID}-${RANDOM_ID}"
BEDROCK_PROJECT_NAME="mdp-bedrock-${USER_ID}-${RANDOM_ID}"

# For now use a region where BDA is available (for english), namely:
# US East (N. Virginia) (us-east-1)
# US West (Oregon) (us-west-2)
# AWS GovCloud (US-West) (us-gov-west-1)
# Europe Regions:
# Europe (Frankfurt) (eu-central-1)
# Europe (Ireland) (eu-west-1)
# Europe (London) (eu-west-2)
# Asia Pacific Regions:
# Asia Pacific (Mumbai) (ap-south-1)
# Asia Pacific (Sydney) (ap-southeast-2)
AWS_REGION="us-east-1"

print_status "=== Dynamic MDP Deployment ==="
print_status "Generated Configuration:"
echo "  Stack Name: $STACK_NAME"
echo "  Bucket Name: $BUCKET_NAME"
echo "  ECR Repository: $ECR_REPO_NAME"
echo "  Bedrock Project: $BEDROCK_PROJECT_NAME"
echo "  AWS Region: $AWS_REGION"
if [[ -n "$PREBUILT_IMAGE_NAME" ]]; then
    echo "  Pre-built Image: $PREBUILT_IMAGE_NAME"
fi
echo

read -p "Proceed with deployment? (Y/n): " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    print_status "Deployment cancelled"
    exit 0
fi

# Create S3 bucket
print_status "Creating S3 bucket: $BUCKET_NAME"
if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3 mb "s3://$BUCKET_NAME"
else
    aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION"
fi
print_success "S3 bucket created: $BUCKET_NAME"

# Create ECR repository
print_status "Creating ECR repository: $ECR_REPO_NAME"
aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION"
ECR_URI=$(aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" --query 'repositories[0].repositoryUri' --output text)
print_success "ECR repository created: $ECR_URI"

# Use pre-built Docker image from images folder and push to ECR
print_status "Using pre-built Docker image from images folder..."

IMAGES_DIR="images"
SOURCE_IMAGE=""

# Check if images directory exists
if [[ ! -d "$IMAGES_DIR" ]]; then
    print_error "Images directory not found: $IMAGES_DIR"
    print_status "Please run ./build_sagemaker_prebuilt.sh first to create pre-built images"
    exit 1
fi

# If a specific image name was provided, use it
if [[ -n "$PREBUILT_IMAGE_NAME" ]]; then
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "$PREBUILT_IMAGE_NAME"; then
        SOURCE_IMAGE="$PREBUILT_IMAGE_NAME"
        print_status "Using specified pre-built image: $SOURCE_IMAGE"
    else
        print_error "Specified image '$PREBUILT_IMAGE_NAME' not found!"
        print_status "Loading from images directory..."
        # Try to load from images directory
        if [[ -f "$IMAGES_DIR/mdp-sagemaker_latest_sagemaker_amd64.tar.gz" ]]; then
            print_status "Loading pre-built image from $IMAGES_DIR..."
            ./load_prebuilt_image.sh "$IMAGES_DIR" "" true
            SOURCE_IMAGE="mdp-sagemaker:latest"
        else
            print_error "No pre-built image found in $IMAGES_DIR"
            exit 1
        fi
    fi
else
    # Check if SageMaker image is already loaded
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "mdp-sagemaker:latest"; then
        SOURCE_IMAGE="mdp-sagemaker:latest"
        print_status "Found loaded SageMaker image: $SOURCE_IMAGE"
    else
        # Load from images directory
        print_status "Loading pre-built SageMaker image from $IMAGES_DIR..."
        if [[ -f "$IMAGES_DIR/mdp-sagemaker_latest_sagemaker_amd64.tar.gz" ]]; then
            ./load_prebuilt_image.sh "$IMAGES_DIR" "" true
            SOURCE_IMAGE="mdp-sagemaker:latest"
            print_success "Pre-built SageMaker image loaded: $SOURCE_IMAGE"
        else
            print_error "No pre-built SageMaker image found in $IMAGES_DIR"
            print_status "Please run ./build_sagemaker_prebuilt.sh first"
            exit 1
        fi
    fi
fi

# Verify the source image architecture
print_status "Verifying source image architecture..."
SOURCE_ARCH=$(docker inspect "$SOURCE_IMAGE" --format='{{.Architecture}}')
SOURCE_OS=$(docker inspect "$SOURCE_IMAGE" --format='{{.Os}}')
print_status "Source image: $SOURCE_IMAGE ($SOURCE_OS/$SOURCE_ARCH)"

# Verify it's SageMaker compatible
if [[ "$SOURCE_ARCH" == "amd64" && "$SOURCE_OS" == "linux" ]]; then
    print_success "✅ Image is SageMaker compatible (linux/amd64)"
else
    print_error "❌ Image is not SageMaker compatible ($SOURCE_OS/$SOURCE_ARCH)"
    print_status "SageMaker requires linux/amd64 architecture"
    exit 1
fi

# Tag the pre-built image for ECR
print_status "Tagging image for ECR..."
docker tag "$SOURCE_IMAGE" "$ECR_URI:latest"

print_status "Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URI"

print_status "Pushing Docker image to ECR..."
docker push "$ECR_URI:latest"
print_success "Docker image pushed: $ECR_URI:latest"

# Upload Lambda layers
print_status "Uploading Lambda layers to S3..."
if [ -d "Layers" ]; then
    for layer_file in Layers/*.zip; do
        if [ -f "$layer_file" ]; then
            layer_name=$(basename "$layer_file")
            print_status "Uploading layer: $layer_name"
            aws s3 cp "$layer_file" "s3://$BUCKET_NAME/Layers/$layer_name"
        fi
    done
    print_success "Lambda layers uploaded"
else
    print_error "Layers directory not found"
    exit 1
fi

# Create Lambda deployment package
print_status "Creating Lambda deployment package..."
TEMP_DIR=$(mktemp -d)
LAMBDA_ZIP="lambda-deployment-package.zip"

cp lambda_function.py "$TEMP_DIR/"
cp processing_script_clean.py "$TEMP_DIR/"
cp -r src/ "$TEMP_DIR/"

cd "$TEMP_DIR"
zip -r "../$LAMBDA_ZIP" .
cd - > /dev/null

mv "$TEMP_DIR/../$LAMBDA_ZIP" .
rm -rf "$TEMP_DIR"

# Upload Lambda code
print_status "Uploading Lambda code to S3..."
aws s3 cp "$LAMBDA_ZIP" "s3://$BUCKET_NAME/code/multi-lingual-document-processor.zip"
rm -f "$LAMBDA_ZIP"
print_success "Lambda code uploaded"

# Create dynamic CloudFormation template
print_status "Creating CloudFormation template..."
cat > "CFT_template_dynamic.yml" << EOF
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Dynamic Multi-lingual Document Processor with auto-generated resource names (uksb-0bszcc4za6) (tag: mdp)'

Parameters:
  BucketName:
    Type: String
    Default: $BUCKET_NAME
  
  ECRImageUri:
    Type: String
    Default: $ECR_URI:latest
  
  BedrockProjectName:
    Type: String
    Default: $BEDROCK_PROJECT_NAME

Resources:
  # Lambda Layers
  SMLayer:
    Type: AWS::Lambda::LayerVersion
    Properties:
      LayerName: !Sub '\${AWS::StackName}-SM-layer'
      Description: SageMaker layer for document processing
      Content:
        S3Bucket: !Ref BucketName
        S3Key: Layers/SM_layer_new-a5ffd1d6-c8ea-46da-a6a2-5fde3f56bfea.zip
      CompatibleRuntimes:
        - python3.10

  RPDSLayer:
    Type: AWS::Lambda::LayerVersion
    Properties:
      LayerName: !Sub '\${AWS::StackName}-rpds-layer'
      Description: RPDS layer for document processing
      Content:
        S3Bucket: !Ref BucketName
        S3Key: Layers/rpds_layer_new-77170122-82b4-4d4c-a641-21c22d1958cc.zip
      CompatibleRuntimes:
        - python3.10

  PandasLayer:
    Type: AWS::Lambda::LayerVersion
    Properties:
      LayerName: !Sub '\${AWS::StackName}-pandas-layer'
      Description: Pandas layer for document processing
      Content:
        S3Bucket: !Ref BucketName
        S3Key: Layers/pandas_layer_new-aeb1aa13-9fd0-42e3-92eb-3a88421a15b0.zip
      CompatibleRuntimes:
        - python3.10

  PydanticLayer:
    Type: AWS::Lambda::LayerVersion
    Properties:
      LayerName: !Sub '\${AWS::StackName}-pydantic-layer'
      Description: Pydantic layer for document processing
      Content:
        S3Bucket: !Ref BucketName
        S3Key: Layers/pydantic_layer-5de6968a-c7c4-4ff1-adc4-2bbbdc7e8a59.zip
      CompatibleRuntimes:
        - python3.10

  NumpyLayer:
    Type: AWS::Lambda::LayerVersion
    Properties:
      LayerName: !Sub '\${AWS::StackName}-numpy-layer'
      Description: Numpy layer for document processing
      Content:
        S3Bucket: !Ref BucketName
        S3Key: Layers/numpy_layer_new-2f638ff0-7c24-4697-a9bb-44cc0eef08a7.zip
      CompatibleRuntimes:
        - python3.10

  # IAM Roles
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '\${AWS::StackName}-SageMakerExecutionRole'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        - arn:aws:iam::aws:policy/AmazonTextractFullAccess
        - arn:aws:iam::aws:policy/AmazonBedrockFullAccess
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
      Policies:
        - PolicyName: BedrockDataAutomationPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - bedrock-data-automation-runtime:*
                  - bedrock-data-automation:*
                  - bedrock:*
                  - bedrock:InvokeDataAutomationAsync
                  - sts:AssumeRole
                Resource: "*"
              - Effect: Allow
                Action: sts:AssumeRole
                Resource: !Sub "arn:aws:iam::\${AWS::AccountId}:role/\${AWS::StackName}-BedrockDataAutomationRole"

  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '\${AWS::StackName}-LambdaExecutionRole'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        - arn:aws:iam::aws:policy/AmazonTextractFullAccess
        - arn:aws:iam::aws:policy/AmazonBedrockFullAccess
        - arn:aws:iam::aws:policy/AmazonS3FullAccess

  BedrockDataAutomationRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '\${AWS::StackName}-BedrockDataAutomationRole'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: bedrock.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonBedrockFullAccess
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
      Policies:
        - PolicyName: BedrockDataAutomationPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - bedrock-data-automation-runtime:*
                  - bedrock-data-automation:*
                  - bedrock:*
                  - sts:AssumeRole
                Resource: "*"

  # Lambda Function
  LambdaFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub '\${AWS::StackName}-MDPDocumentProcessor'
      Handler: lambda_function.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Code:
        S3Bucket: !Ref BucketName
        S3Key: code/multi-lingual-document-processor.zip
      Runtime: python3.10
      Timeout: 900
      MemorySize: 1024
      Layers:
        - !Ref SMLayer
        - !Ref RPDSLayer
        - !Ref PandasLayer
        - !Ref PydanticLayer
        - !Ref NumpyLayer
      EphemeralStorage:
        Size: 10240
      Environment:
        Variables:
          SAGEMAKER_ROLE_ARN: !GetAtt SageMakerExecutionRole.Arn
          STACK_NAME: !Ref AWS::StackName
          CODE_BUCKET: !Ref BucketName
          BEDROCK_PROJECT_NAME: !Ref BedrockProjectName
          BEDROCK_PROJECT_ARN: !GetAtt BedrockDataAutomationProject.ProjectArn


          BEDROCK_DATA_AUTOMATION_ROLE_ARN: !GetAtt BedrockDataAutomationRole.Arn
          ECR_IMAGE_URI: !Ref ECRImageUri

  # Bedrock Data Automation Project
  BedrockDataAutomationProject:
    Type: AWS::Bedrock::DataAutomationProject
    Properties:
      ProjectName: !Ref BedrockProjectName
      ProjectDescription: !Sub 'Automated data processing project for \${AWS::StackName}'
      ProjectStage: DEVELOPMENT
      StandardOutputConfiguration:
        Document:
          Extraction:
            Granularity:
              Types:
                - PAGE
                - ELEMENT
                - WORD
            BoundingBox:
              State: ENABLED
          GenerativeField:
            State: ENABLED
        Image:
          Extraction:
            Category:
              State: ENABLED
              Types:
                - CONTENT_MODERATION
                - TEXT_DETECTION
            BoundingBox:
              State: ENABLED
          GenerativeField:
            State: ENABLED
            Types:
              - IMAGE_SUMMARY

  # API Gateway
  ApiGateway:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: !Sub '\${AWS::StackName}-DocumentProcessorAPI'
      Description: API for Multi-lingual Document Processor

  LambdaResource:
    Type: AWS::ApiGateway::Resource
    Properties:
      RestApiId: !Ref ApiGateway
      ParentId: !GetAtt ApiGateway.RootResourceId
      PathPart: process

  LambdaMethod:
    Type: AWS::ApiGateway::Method
    Properties:
      RestApiId: !Ref ApiGateway
      ResourceId: !Ref LambdaResource
      HttpMethod: POST
      AuthorizationType: NONE
      ApiKeyRequired: true
      Integration:
        Type: AWS
        IntegrationHttpMethod: POST
        Uri: !Sub 'arn:aws:apigateway:\${AWS::Region}:lambda:path/2015-03-31/functions/\${LambdaFunction.Arn}/invocations'
        RequestParameters:
          integration.request.header.X-Amz-Invocation-Type: "'Event'"
        IntegrationResponses:
          - StatusCode: 202
            ResponseTemplates:
              application/json: |
                {
                  "message": "Document processing request accepted",
                  "requestId": "\$context.requestId"
                }
      MethodResponses:
        - StatusCode: 202

  ApiDeployment:
    Type: AWS::ApiGateway::Deployment
    DependsOn: LambdaMethod
    Properties:
      RestApiId: !Ref ApiGateway
      StageName: prod

  ApiKey:
    Type: AWS::ApiGateway::ApiKey
    Properties:
      Name: !Sub '\${AWS::StackName}-ApiKey'
      Description: API Key for Document Processor
      Enabled: true

  UsagePlan:
    Type: AWS::ApiGateway::UsagePlan
    DependsOn: ApiDeployment
    Properties:
      UsagePlanName: !Sub '\${AWS::StackName}-UsagePlan'
      Description: Usage plan for Document Processor API
      ApiStages:
        - ApiId: !Ref ApiGateway
          Stage: prod
      Throttle:
        RateLimit: 100
        BurstLimit: 200
      Quota:
        Limit: 10000
        Period: DAY

  UsagePlanKey:
    Type: AWS::ApiGateway::UsagePlanKey
    Properties:
      KeyId: !Ref ApiKey
      KeyType: API_KEY
      UsagePlanId: !Ref UsagePlan

  LambdaInvokePermission:
    Type: AWS::Lambda::Permission
    Properties:
      Action: lambda:InvokeFunction
      FunctionName: !Ref LambdaFunction
      Principal: apigateway.amazonaws.com
      SourceArn: !Sub arn:aws:execute-api:\${AWS::Region}:\${AWS::AccountId}:\${ApiGateway}/*/*

Outputs:
  StackName:
    Description: Name of the CloudFormation stack
    Value: !Ref AWS::StackName
    
  BucketName:
    Description: Name of the S3 bucket
    Value: !Ref BucketName
    
  ECRImageUri:
    Description: ECR Image URI
    Value: !Ref ECRImageUri
    
  ApiEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub 'https://\${ApiGateway}.execute-api.\${AWS::Region}.amazonaws.com/prod/process'
    
  ApiKeyId:
    Description: API Key ID
    Value: !Ref ApiKey
    
  LambdaFunctionName:
    Description: Lambda function name
    Value: !Ref LambdaFunction
EOF

# Validate CloudFormation template syntax
print_status "Validating CloudFormation template syntax..."

# Use AWS CloudFormation validation (skip Python validation to avoid constructor issues)
if aws cloudformation validate-template --template-body file://CFT_template_dynamic.yml >/dev/null 2>&1; then
    print_success "CloudFormation template syntax is valid"
else
    print_error "CloudFormation template has syntax errors"
    echo "Detailed error information:"
    aws cloudformation validate-template --template-body file://CFT_template_dynamic.yml
    echo ""
    echo "Generated template content (first 20 lines):"
    head -20 CFT_template_dynamic.yml
    exit 1
fi

# Deploy CloudFormation stack
print_status "Deploying CloudFormation stack: $STACK_NAME"
aws cloudformation create-stack \
    --stack-name "$STACK_NAME" \
    --template-body file://CFT_template_dynamic.yml \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters \
        ParameterKey=BucketName,ParameterValue="$BUCKET_NAME" \
        ParameterKey=ECRImageUri,ParameterValue="$ECR_URI:latest" \
        ParameterKey=BedrockProjectName,ParameterValue="$BEDROCK_PROJECT_NAME" \
    --region "$AWS_REGION"

print_status "Waiting for stack creation to complete..."
aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"

# Apply Bedrock permissions fix after stack creation
print_status "Applying Bedrock Data Automation permissions fix..."

# Create inline permissions fix for this deployment
SAGEMAKER_ROLE_NAME="$STACK_NAME-SageMakerExecutionRole"
BEDROCK_ROLE_NAME="$STACK_NAME-BedrockDataAutomationRole"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
BEDROCK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${BEDROCK_ROLE_NAME}"

print_status "Updating Bedrock Data Automation role trust policy..."

# Create trust policy that includes SageMaker role
cat > /tmp/bedrock_trust_policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "$SAGEMAKER_ROLE_ARN"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Update the trust policy
if aws iam update-assume-role-policy \
    --role-name "$BEDROCK_ROLE_NAME" \
    --policy-document file:///tmp/bedrock_trust_policy.json \
    --region "$AWS_REGION"; then
    print_success "Bedrock Data Automation permissions configured successfully"
else
    print_error "Failed to configure Bedrock permissions, but stack deployment succeeded"
    print_status "You can manually run the fix_bedrock_permissions.sh script"
fi

# Clean up temp file
rm -f /tmp/bedrock_trust_policy.json

# Get outputs
print_success "=== DEPLOYMENT COMPLETED ==="
aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}' --output table

print_success "Stack deployed successfully: $STACK_NAME"
print_status "Save these details for future reference:"
echo "  Stack Name: $STACK_NAME"
echo "  Bucket Name: $BUCKET_NAME"
echo "  ECR Repository: $ECR_REPO_NAME"
echo "  Region: $AWS_REGION"

