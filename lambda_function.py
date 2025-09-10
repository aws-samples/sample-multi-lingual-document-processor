
import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
import logging
import os
import json

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Simplified Lambda handler for MDP Document Processor
    Reads bucket name, document type, and document language from payload
    Starts SageMaker Processing job with Bedrock Data Automation project ARN
    """

    try:
        logger.info("Starting MDP Document Processor")

        # Extract parameters from payload
        bucket_name = event.get('bucket')
        file_key = event.get('file_key')  # S3 file key path
        doc_category = event.get('doc_category', 'report')  # Changed from doc_type to doc_category
        doc_language = event.get('doc_language', 'english')
        output_language = event.get('output_language', 'english')  # Added output_language

        # Validate required parameters
        if not bucket_name:
            raise ValueError("Missing required parameter: bucket")
        if not file_key:
            raise ValueError("Missing required parameter: file_key")

        logger.info(f"Bucket: {bucket_name}, File Key: {file_key}, Doc Category: {doc_category}, Language: {doc_language}, Output Language: {output_language}")

        # Get environment variables
        sagemaker_role = os.environ['SAGEMAKER_ROLE_ARN']
        ecr_image_uri = os.environ['ECR_IMAGE_URI']
        bedrock_project_arn = os.environ['BEDROCK_PROJECT_ARN']
        bedrock_data_automation_role_arn = os.environ['BEDROCK_DATA_AUTOMATION_ROLE_ARN']

        # Create SageMaker ScriptProcessor
        processor = ScriptProcessor(
            command=['python3'],
            image_uri=ecr_image_uri,
            role=sagemaker_role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            env={
                "BEDROCK_PROJECT_ARN": bedrock_project_arn,
                "BEDROCK_DATA_AUTOMATION_ROLE_ARN": bedrock_data_automation_role_arn,
                "FILE_KEY": file_key
            }
        )

        # Define processing inputs and outputs
        # Extract directory path from file_key for input
        file_directory = '/'.join(file_key.split('/')[:-1]) if '/' in file_key else ''
        #source=f"s3://{bucket_name}/{file_directory}/" if file_directory else f"s3://{bucket_name}/"
        inputs = [
            ProcessingInput(
                source=f"s3://{bucket_name}/{file_key}/",
                destination="/opt/ml/processing/input/data/"
            )
        ]

        outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/output/",
                destination=f"s3://{bucket_name}/output/"
            )
        ]

        # Processing job arguments - matching processing_script_clean.py expectations
        arguments = [
            "--bucket", bucket_name,
            "--file_key", file_key,
            "--doc_category", doc_category,
            "--doc_language", doc_language,
            "--output_language", output_language,
            "--bedrock_project_arn", bedrock_project_arn
        ]

        # Start SageMaker processing job
        processor.run(
            inputs=inputs,
            outputs=outputs,
            code="processing_script_clean.py",
            wait=False,
            arguments=arguments
        )

        logger.info("SageMaker processing job started successfully")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing job started successfully',
                'bucket': bucket_name,
                'file_key': file_key,
                'doc_category': doc_category,
                'doc_language': doc_language,
                'output_language': output_language,
                'bedrock_project_arn': bedrock_project_arn
            })
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

