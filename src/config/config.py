
"""
Configuration settings for the document processing system.
"""
import os

# Check if running in SageMaker environment BEFORE accessing any paths
def is_in_sagemaker():
    """Determine if we're running in a SageMaker environment"""
    return (os.environ.get('SM_FRAMEWORK_PARAMS') is not None or
            os.environ.get('SM_CHANNEL_TRAINING') is not None or
            os.path.exists('/opt/ml/processing') or
            os.path.exists('/opt/ml/input'))

# AWS Configuration
AWS_REGION = "us-east-1"
MAX_RETRIES = 10
S3_BUCKET = "mdp-newtest-3"
#OUTPUT_DIR = "output_report"
IN_SAGEMAKER = is_in_sagemaker()
if IN_SAGEMAKER:
    INPUT_DIR = '/opt/ml/processing/input/data'
    CODE_DIR = '/opt/ml/processing/input/code'
    OUTPUT_DIR = '/opt/ml/processing/output'
else:
    # Local paths for testing
    INPUT_DIR = os.path.join(os.getcwd(), 'input', 'data')
    CODE_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(os.getcwd(), 'output')

# Bedrock Model Configuration
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Update as needed
CONFIDENCE_MODEL_ID = 'claude-3-7-sonnet-20250219-v1:0'
#BDA_arn = os.environ.get('BEDROCK_PROJECT_ARN')
# BDA_arn = "arn:aws:bedrock:us-east-1:203918850931:data-automation-project/3baebf6b0a1c"
#BDA_arn = "dummy arn"
tabula_model_ID= "us.anthropic.claude-sonnet-4-20250514-v1:0"
table_validation_model_ID= "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Timeout and Retry Settings
BOTO3_CONFIG = {
    "connect_timeout": 30,
    "read_timeout": 300,  # 5 minutes timeout
    "retries": {"max_attempts": MAX_RETRIES},
}

# Directory Structure
TEMP_DIRECTORIES = [
    "/tmp/data_in_pages",
    "/tmp/final_output",
    "/tmp/final_output_text",
    "/tmp/data_in_images",
    "/tmp/page",
    "/tmp/image_output",
    "/tmp/table_output",
    "/tmp/table_output_final",
    "/tmp/text_output",
    "/tmp/table_narration",
    "/tmp/image_narration",
    "/tmp/pages_in_images",
    '/tmp/confidence_in_images'
]

# File paths
COMBINED_OUTPUT_FILE = "combined_output.txt"

