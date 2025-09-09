
"""
SageMaker Processing Script for document processing.
This script is executed inside the SageMaker Processing container or locally.
"""
import sys
sys.path.append('/opt/ml')
sys.path.append('/opt/ml/src')
sys.path.append('/opt/ml/processing/input/code')
print(sys.path)

import os
# print("Files in /opt/ml/processing/input/code:")
# print(os.listdir('/opt/ml/processing/input/code'))

import logging
import glob
import shutil
import traceback
import argparse
#from src.config.config import BDA_arn

from src.processors.document_processor import DocumentProcessor
from src.utils.cleanup_utility import cleanup_for_sagemaker
import boto3
import gc
import threading
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if running in SageMaker environment BEFORE accessing any paths
def is_in_sagemaker():
    """Determine if we're running in a SageMaker environment"""
    return (os.environ.get('SM_FRAMEWORK_PARAMS') is not None or
            os.environ.get('SM_CHANNEL_TRAINING') is not None or
            os.path.exists('/opt/ml/processing') or
            os.path.exists('/opt/ml/input'))

# Set paths based on environment - don't access /opt/ml directly yet
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

# Add current directory to path for imports
sys.path.append(os.getcwd())

def process_documents(doc_category, doc_language, output_language, bedrock_project_arn=None):
    """
    Process documents in the input directory.

    Args:
        doc_category: Document category
        doc_language: Document language
        output_language: Document output language
        input_filename: Input filename to process
        bedrock_project_arn: ARN of the Bedrock data automation project (optional)
    """
    ## Check if input directory exists
    #if not os.path.exists(INPUT_DIR):
    #    logger.warning(f"Input directory does not exist: {INPUT_DIR}")
    #    if not IN_SAGEMAKER:
    #        logger.info("Creating sample input directory for local testing")
    #        os.makedirs(INPUT_DIR, exist_ok=True)
    #    return

    # Create output directory if it doesn't exist
    #os.makedirs(OUTPUT_DIR, exist_ok=True)
    #print(input_filename)

    # Get all PDF files in the input directory
    pdf_files = glob.glob(f"{INPUT_DIR}/*.pdf", recursive=True)

    if not pdf_files:
        logger.warning(f"No PDF files found in the input directory: {INPUT_DIR}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Initialize Bedrock client if project ARN is provided
    bedrock_client = None
    if bedrock_project_arn:

        try:

            bedrock_client = boto3.client('bedrock-data-automation', region_name='us-east-1')
            logger.info(f"Bedrock client initialized with project ARN: {bedrock_project_arn}")

            # Validate Bedrock project
            try:

                response = bedrock_client.get_data_automation_project(
                    projectArn=bedrock_project_arn
                )
                logger.info("Bedrock project validated successfully")
                logger.info(f"Project details: {response.get('project', {}).get('projectName', 'Unknown')}")
            except Exception as bedrock_error:
                logger.warning(f"Bedrock project validation failed: {str(bedrock_error)}")
                bedrock_client = None

        except Exception as e:
            logger.warning(f"Failed to initialize Bedrock client: {str(e)}")
            bedrock_client = None
    else:
        logger.info("No Bedrock project ARN provided - proceeding with standard processing")

    # Process each PDF file
    for pdf_file in pdf_files:
        processor = None

        try:
            # relative_path = os.path.relpath(pdf_file, INPUT_DIR)

            # Initialize and run the document processor
            if pdf_file:
                logger.info(f"Processing file: {pdf_file}")

                # Initialize DocumentProcessor with Bedrock project ARN
                # Note: input_bucket should be passed from main() or made global
                processor = DocumentProcessor(
                    pdf_file,
                    os.environ.get('INPUT_BUCKET', 'default-bucket'),  # Get from environment or use default
                    os.environ.get('FILE_KEY', 'uploads'),
                    doc_category,
                    doc_language,
                    output_language,
                    pdf_file,
                    bedrock_project_arn=bedrock_project_arn
                )

                # Process with Bedrock if available
                if bedrock_client and bedrock_project_arn:
                    try:
                        logger.info("Attempting Bedrock-enhanced processing")
                        # Add Bedrock processing logic here
                        # This is where you would integrate specific Bedrock data automation calls
                        processor.process_with_bedrock = True
                        processor.bedrock_client = bedrock_client
                        processor.bedrock_project_arn = bedrock_project_arn
                    except Exception as bedrock_error:
                        logger.warning(f"Bedrock processing setup failed: {str(bedrock_error)}")
                        logger.info("Falling back to standard processing")

                # Run the standard processing
                processor.process()

                # Copy the combined output file to the output directory
                # combined_output = '/tmp/combined_output.txt'
                # if os.path.exists(combined_output):
                #     # Create a directory structure in the output that mirrors the input
                #     output_subdir = os.path.join(OUTPUT_DIR, os.path.dirname(relative_path))
                #     os.makedirs(output_subdir, exist_ok=True)

                #     # Create the output filename based on the input filename
                #     base_name = os.path.splitext(os.path.basename(pdf_file))[0]

                #     # Create an additional directory level with base_name
                #     base_name_dir = os.path.join(output_subdir, base_name)
                #     os.makedirs(base_name_dir, exist_ok=True)

                #     # Save the processed text file in the base_name directory
                #     output_file = os.path.join(base_name_dir, f"{base_name}_processed.txt")
                #     shutil.copy(combined_output, output_file)
                #     logger.info(f"Copied processed results to {output_file}")


                #     # Copy the original PDF file to the same directory
                #     pdf_destination = os.path.join(base_name_dir, os.path.basename(pdf_file))
                #     shutil.copy(pdf_file, pdf_destination)
                #     logger.info(f"Copied original PDF to {pdf_destination}")

                #     # Copy all files from /tmp/table_output to the same directory
                #     table_output_dir = "/tmp/table_output_final"
                #     table_process_dir= os.path.join(base_name_dir ,'tables')
                #     os.makedirs(table_process_dir, exist_ok=True)


                #     if os.path.exists('/tmp/processed_images'):
                #         # Save the processed image file in the base_name directory
                #         image_folder = os.path.join(base_name_dir, "image_processed")
                #         image_files = glob.glob(os.path.join('/tmp/processed_images', "*"))
                #         os.makedirs(image_folder, exist_ok=True)
                #         for image_file in image_files:
                #             shutil.copy(image_file, image_folder)
                #             logger.info(f"Copied images to {image_folder}")

                #     if os.path.exists(table_process_dir):
                #         table_files = glob.glob(os.path.join(table_output_dir, "*"))
                #         if table_files:
                #             for table_file in table_files:
                #                 if os.path.isfile(table_file):
                #                     if(table_file.endswith('.csv')):
                #                         table_destination = os.path.join(table_process_dir, os.path.basename(table_file))
                #                         shutil.copy(table_file, table_destination)
                #                         logger.info(f"Copied table file to {table_destination}")
                #         else:
                #             logger.info("No table output files found to copy")
                #     else:
                #         logger.info("Table output directory does not exist")

                #     confidence_score_dir= '/tmp/confidence_in_images'
                #     confidence_process_dir= os.path.join(base_name_dir ,'confidence_scores')
                #     os.makedirs(confidence_process_dir, exist_ok=True)
                #     print(confidence_process_dir)
                #     if os.path.exists(confidence_process_dir):
                #         confidence_score_files = glob.glob(os.path.join(confidence_score_dir, "*"))
                #         if confidence_score_files:
                #             for confidence_file in confidence_score_files:
                #                 if os.path.isfile(confidence_file):
                #                     confidence_destination = os.path.join(confidence_process_dir, os.path.basename(confidence_file))
                #                     shutil.copy(confidence_file, confidence_destination)
                #                     logger.info(f"Copied table file to {confidence_destination}")
                #         else:
                #             logger.info("No Confidence output files found to copy")
                #     else:
                #         logger.info("Confidence output directory does not exist")

                # else:
                #     logger.warning(f"No output file found for {pdf_file}")

                # Clean up temporary files using the processor's cleanup method
                # if processor:
                #     processor.cleanup()

            else:
                continue

        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            logger.error(traceback.format_exc())

            # Ensure cleanup happens even if processing fails
            if processor:
                try:
                    processor.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Error during processor cleanup: {str(cleanup_error)}")

            # Continue with the next file

def main():
    """
    Main function for the processing script.
    """
    try:
        logger.info("Starting document processing job")
        logger.info(f"Running in SageMaker environment: {IN_SAGEMAKER}")
        parser = argparse.ArgumentParser()
        parser.add_argument("--bucket", help="S3 bucket name")
        parser.add_argument("--file_key", help="S3 file key path")
        parser.add_argument("--doc_language", default="english", help="Document language")
        parser.add_argument("--output_language", default="english", help="Document language")
        parser.add_argument("--doc_category", default="general", help="Document category")
        parser.add_argument("--bedrock_project_arn", help="ARN of the Bedrock data automation project")
        args, _ = parser.parse_known_args()

        # Get parameters from command line args or environment variables
        input_bucket = args.bucket or os.environ.get('INPUT_BUCKET') or os.environ.get('BUCKET_NAME')
        file_key = args.file_key or os.environ.get('FILE_KEY') or os.environ.get('INPUT_FILE_KEY')
        doc_language = args.doc_language or os.environ.get('DOC_LANGUAGE', 'english')
        output_language = args.output_language or os.environ.get('OUTPUT_LANGUAGE', 'english')
        doc_category = args.doc_category or os.environ.get('DOC_CATEGORY', 'general')
        bedrock_project_arn = args.bedrock_project_arn or os.environ.get('BEDROCK_PROJECT_ARN')

        # For testing/demo purposes, use default values if not provided
        if not input_bucket:
            logger.warning("No bucket specified, using test mode")
            input_bucket = "test-bucket"
        #if not file_key:
        #    logger.warning("No file key specified, using test mode")
        #    file_key = "test-document.pdf"

        logger.info(f"Processing parameters: bucket={input_bucket}, language={doc_language}, category={doc_category}")
        file_list = os.listdir('/opt/ml/processing/input/data/')
        logger.info(f"files: {file_list}")

        # Extract filename from file key
        #input_filename = os.path.basename(file_key)

        logger.info(f"Processing parameters:")
        logger.info(f"  Bucket: {input_bucket}")
        logger.info(f"  File key: {file_key}")
        #logger.info(f"  Filename: {input_filename}")
        logger.info(f"  Category: {doc_category}")
        logger.info(f"  Language: {output_language}")
        logger.info(f"  Output Language: {doc_language}")
        logger.info(f"  Bedrock ARN: {bedrock_project_arn}")

        # Set environment variable for bucket access in processing
        os.environ['INPUT_BUCKET'] = input_bucket

        # Process the document
        process_documents(doc_category, doc_language, output_language, bedrock_project_arn)

        logger.info("Document processing job completed successfully")

    except Exception as e:
        logger.error(f"Error in processing job: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # finally:
    #     # Always perform cleanup at the end of processing
    #     try:
    #         logger.info("Performing SageMaker cleanup")

    #         # First, perform the standard cleanup
    #         cleanup_for_sagemaker()
    #         active_threads = threading.enumerate()
    #         logger.info(f"Active threads before exit: {len(active_threads)}")
    #         for thread in active_threads:
    #             if thread != threading.current_thread():
    #                 logger.info(f"Thread still active: {thread.name}, Daemon: {thread.daemon}")
    #                 # Try to force threads to be daemon so they don't prevent exit
    #                 if not thread.daemon:
    #                     try:
    #                         thread._Thread__daemonic = True
    #                     except:
    #                         pass

    #         # Close any open resources from ThreadPoolExecutor
    #         from concurrent.futures import thread
    #         try:
    #             thread._threads_queues.clear()
    #         except:
    #             pass

    #         logger.info("SageMaker cleanup completed")

    #         # Force immediate exit after logging is complete
    #         logger.info("Forcing process exit")

    #         # Flush any remaining log output
    #         # import sys
    #         # sys.stdout.flush()
    #         # sys.stderr.flush()

    #         # # Terminate the process immediately
    #         # os._exit(0)

    #     except Exception as cleanup_error:
    #         logger.warning(f"Error during SageMaker cleanup: {str(cleanup_error)}")
    #         # Even if cleanup fails, ensure we exit
    #         os._exit(1)

    finally:
        # Always perform cleanup at the end of processing
        try:
            logger.info("Performing SageMaker cleanup")
            cleanup_for_sagemaker()
            logger.info("SageMaker cleanup completed")

            gc.collect()
        except Exception as cleanup_error:
            logger.warning(f"Error during SageMaker cleanup: {str(cleanup_error)}")

            gc.collect()



if __name__ == "__main__":
    main()

