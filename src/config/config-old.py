"""
Configuration settings for the document processing system.
"""
import os

# AWS Configuration
AWS_REGION = "us-east-1"
MAX_RETRIES = 10
S3_BUCKET = "mdp-newtest-3"
OUTPUT_DIR = "output_report"

# Bedrock Model Configuration
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Update as needed
CONFIDENCE_MODEL_ID = 'claude-3-7-sonnet-20250219-v1:0'
BDA_arn = os.environ.get('BEDROCK_PROJECT_ARN')
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
