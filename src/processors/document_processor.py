
"""
Main Document Processor class that orchestrates the document processing workflow.
"""
import sys
import os

import logging

import boto3
from botocore.config import Config
from pathlib import Path
from src.config.config import AWS_REGION, BOTO3_CONFIG, COMBINED_OUTPUT_FILE
from src.processors.file_processor import FileProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.text_processor import TextProcessor, BedrockDataAutomationClient
from src.processors.table_processor import TableProcessor
from src.processors.localization_processor import LocalizationProcessor
from src.utils.directory_manager import DirectoryManager
from src.processors.confidence_score_processor import ConfidenceScoreProcessor
import copy
from src.processors.utils import save_processed_content, process_report


# Process pages in parallel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

print(os.getcwd())

def is_in_sagemaker():
    """Determine if we're running in a SageMaker environment"""
    return os.environ.get('SM_FRAMEWORK_PARAMS') is not None or os.environ.get('SM_CHANNEL_TRAINING') is not None

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


class DocumentProcessor:
    """
    Main class that orchestrates the document processing workflow.
    """
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

    def __init__(self, input_file_path, input_bucket, input_folder, doc_category, doc_language, output_language, input_filename, bedrock_project_arn=None):
        """
        Initialize the document processor.

        Args:
            input_file_path (str): Path to the input PDF file
            input_bucket (str): Name of the S3 bucket containing the input file
            doc_category (str): Category of document
            doc_language (str): Language of the document
            input_filename (str): Original filename
            bedrock_project_arn (str): ARN of the Bedrock data automation project (optional)
        """

        self.input_file_path = input_file_path
        self.input_bucket = input_bucket
        self.input_folder = input_folder
        self.doc_language = doc_language
        self.output_language = output_language
        self.doc_category = doc_category
        filename = input_filename.split('/')[-1]
        self.original_filename = filename
        self.bedrock_project_arn = bedrock_project_arn

        # Bedrock integration flags
        self.process_with_bedrock = False
        self.bedrock_client = None

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Configure logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Log Bedrock availability
        if self.bedrock_project_arn:
            self.logger.info(f"DocumentProcessor initialized with Bedrock project ARN: {self.bedrock_project_arn}")
        else:
            self.logger.info("DocumentProcessor initialized without Bedrock integration")

        # Configure boto3 with longer timeout and retry settings
        self.boto3_config = Config(**BOTO3_CONFIG)

        # Initialize component processors
        self.directory_manager = DirectoryManager()
        self.file_processor = FileProcessor()
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor(bedrock_project_arn=self.bedrock_project_arn)
        filename = self.original_filename
        filename = filename.split('.')[0]
        self.bda_processor = BedrockDataAutomationClient(
                                    region=AWS_REGION,
                                    s3_bucket=self.input_bucket,
                                    project_arn=self.bedrock_project_arn,
                                    input_path=self.input_folder,
                                    output_path=f"output/{filename}"
                                )
        self.table_processor = TableProcessor()
        self.localization_processor = LocalizationProcessor(self.boto3_config)
        self.confidence_processor = ConfidenceScoreProcessor()


    def process(self):
        """
        Process the document through the entire pipeline.
        """
        print("doc_lang:", self.doc_language)
        print("doc_cate:", self.doc_category)
        try:

            # Check for Bedrock integration
            if self.process_with_bedrock and self.bedrock_client and self.bedrock_project_arn:
                self.logger.info("Processing with Bedrock data automation integration")
                try:
                    self._process_with_bedrock_integration()
                except Exception as bedrock_error:
                    self.logger.warning(f"Bedrock processing failed: {str(bedrock_error)}")
                    self.logger.info("Continuing with standard processing")

            try:
                start_time = time.time()
                # Split PDF into pages
                pdf_pages, page_images = self.file_processor.split_pdf_from_s3(self.input_file_path)
                isBDA = False

                if self.doc_language != 'english':
                    self.logger.info(f"Starting PDF processing for file: {self.input_file_path}")

                    # Thread-safe token tracking
                    token_lock = threading.Lock()

                    def process_page(page_data):
                        i, pdf_page, page_image = page_data
                        page_start_time = time.time()
                        page_tokens = {'input': 0, 'output': 0}

                        try:
                            # Process PDF files for text extraction
                            text_content = self.text_processor.process_pdf_files(pdf_page, page_image, self.image_processor)
                            page_tokens['input'] += text_content.get('input_tokens', 0)
                            page_tokens['output'] += text_content.get('output_tokens', 0)

                            # Process images from PDF and track tokens
                            image_content = self.image_processor.process_images_from_pdf(page_image, self.output_language)
                            page_tokens['input'] += image_content.get('input_tokens', 0)
                            page_tokens['output'] += image_content.get('output_tokens', 0)

                            page_image_copy = copy.deepcopy(page_image)
                            pdf_page_copy = copy.deepcopy(pdf_page)
                            # Process tables from PDF and track tokens
                            table_content = self.table_processor.process_single_page_tables(pdf_page=pdf_page_copy, pdf_image=page_image_copy, output_language=self.output_language)
                            page_tokens['input'] += table_content.get('input_tokens', 0)
                            page_tokens['output'] += table_content.get('output_tokens', 0)

                            # Generate final report
                            final_report = self.file_processor.final_report(image_content=image_content, table_content=table_content, text_content=text_content)

                            # Localize and arrange content, track tokens
                            localization_content = self.localization_processor.main_Bedrock_arrange(final_report=final_report, page_num=i, output_language=self.output_language)
                            page_tokens['input'] += localization_content.get('input_tokens', 0)
                            page_tokens['output'] += localization_content.get('output_tokens', 0)

                            page_end_time = time.time()
                            page_duration = page_end_time - page_start_time

                            return i, table_content, image_content, text_content, localization_content, page_tokens, page_duration

                        except Exception as e:
                            page_end_time = time.time()
                            page_duration = page_end_time - page_start_time
                            self.logger.error(f"Error processing page {i}: {str(e)}")
                            # Return empty results for failed page
                            return i, {}, {}, {}, {}, page_tokens, page_duration

                    # Prepare page data for parallel processing
                    page_data_list = [(i, pdf_pages[i], page_images[i]) for i in range(len(pdf_pages))]

                    # Process pages in parallel with 2 workers to avoid overwhelming resources
                    page_results = {}
                    with ThreadPoolExecutor(max_workers=min(2, len(pdf_pages))) as executor:
                        future_to_page = {executor.submit(process_page, page_data): page_data[0] for page_data in page_data_list}

                        for future in as_completed(future_to_page):
                            try:
                                i, table_content, image_content, text_content, localization_content, page_tokens, page_duration = future.result()
                                page_results[i] = (table_content, image_content, text_content, localization_content)

                                # Thread-safe token accumulation
                                with token_lock:
                                    self.total_input_tokens += page_tokens['input']
                                    self.total_output_tokens += page_tokens['output']
                                    self.logger.info(f"Page {i} processing completed in {page_duration:.2f}s - used {page_tokens['input']} input tokens and {page_tokens['output']} output tokens")
                            except Exception as e:
                                self.logger.error(f"Error processing page {future_to_page[future]}: {str(e)}")

                    final_results = []
                    # nosemgrep: arbitrary-sleep
                    #time.sleep(120)
                    # Combine results in page order
                    for i in range(len(pdf_pages)):
                        if i in page_results:
                            table_content, image_content, text_content, localization_content = page_results[i]
                            combine_text = self.file_processor.combine_text_files(table_content=table_content,
                                                                image_content=image_content,
                                                                text_content=text_content,
                                                                localization_content=localization_content)
                            final_results.append(combine_text)

                        # # UNCOMMENT THIS FOR WRITING THE FILES
                        # saved_file_paths = save_processed_content(final_report=final_report,
                        #                                           localization_content=localization_content,
                        #                                           page_num=i,
                        #                                           logger=self.logger,
                        #                                           original_filename=self.original_filename,
                        #                                           output_dir='./results'
                        #                                           )

                elif self.doc_language == 'english' and self.doc_category == 'report':
                    isBDA = True
                    self.logger.info("Processed document using Bedrock Data Automation")
                    final_results, input_tokens, output_tokens =\
                          self.bda_processor.run_bda(self.input_file_path, page_images, self.image_processor, self.table_processor)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens

                elif(self.doc_language == 'english' and self.doc_category == 'invoice'):
                    self.logger.info(f"Starting PDF processing for file: {self.input_file_path}")

                    token_lock = threading.Lock()

                    def process_english_page(page_data):
                        i, pdf_page, page_image = page_data
                        page_tokens = {'input': 0, 'output': 0}
                        page_start_time = time.time()
                        
                        try:
                            # Process tables for this page
                            table_content = self.table_processor.process_single_page_tables(pdf_page=pdf_page, pdf_image=page_image, output_language=self.output_language)
                            
                            # Get confidence scores for this page
                            text_content = self.confidence_processor.extract_from_pdf(page_image)
                            
                            # Generate final report for this page
                            final_report = self.file_processor.final_report(table_content=table_content, image_content=None, text_content=text_content)

                            # Localize and arrange content, track tokens
                            localization_content = self.localization_processor.main_Bedrock_arrange(
                                final_report=final_report, page_num=i, output_language=self.output_language)
                            page_tokens['input'] += localization_content.get('input_tokens', 0)
                            page_tokens['output'] += localization_content.get('output_tokens', 0)

                            page_end_time = time.time()
                            page_duration = page_end_time - page_start_time

                            return i, table_content, None, text_content, localization_content, page_tokens, page_duration
                        
                        except Exception as e:
                            page_duration = time.time() - page_start_time
                            self.logger.error(f"Error processing English page {i}: {str(e)}")
                            return i, {}, {}, {}, {}, page_tokens, page_duration
                    
                    # Process pages in parallel using existing pdf_pages and page_images
                    page_data_list = [(i, pdf_pages[i], page_images[i]) for i in range(len(pdf_pages))]
                    page_results = {}
                    
                    with ThreadPoolExecutor(max_workers=min(1, len(pdf_pages))) as executor:
                        future_to_page = {executor.submit(process_english_page, page_data): page_data[0] for page_data in page_data_list}

                        for future in as_completed(future_to_page):
                            try:
                                i, table_content, image_content, text_content, localization_content, page_tokens, page_duration = future.result()
                                page_results[i] = (table_content, image_content, text_content, localization_content)

                                # Thread-safe token accumulation
                                with token_lock:
                                    self.total_input_tokens += page_tokens['input']
                                    self.total_output_tokens += page_tokens['output']
                                    self.logger.info(
                                        f"Page {i} processing completed in {page_duration:.2f}s - used {page_tokens['input']} input tokens and {page_tokens['output']} output tokens")
                            except Exception as e:
                                self.logger.error(f"Error processing page {future_to_page[future]}: {str(e)}")

                    final_results = []
                    # Combine results in page order
                    for i in range(len(pdf_pages)):
                        if i in page_results:
                            table_content, image_content, text_content, localization_content = page_results[i]
                            combine_text = self.file_processor.combine_text_files(table_content=table_content,
                                                                                  image_content=image_content,
                                                                                  text_content=text_content,
                                                                                  localization_content=localization_content)
                            # print('================================',combine_text,'================================')
                            final_results.append(combine_text)


                else:
                    self.logger.info("document category not found")
                processing_reports = {}
                processing_reports['total_input_tokens'] = self.total_input_tokens
                processing_reports['total_output_tokens'] = self.total_output_tokens
                processing_reports['total_time'] = time.time() - start_time

                filename = self.input_file_path.split('/')[-1]
                filename = filename.split('.')[0]
                output_path = os.path.join(OUTPUT_DIR, filename)
                process_report(final_results, processing_reports, self.input_bucket, output_path, isBDA)
                #process_report(final_results, self.input_bucket, output_path)

            except Exception as e:
                self.logger.error(f"Error in document processing: {str(e)}", exc_info=True)
                raise

        finally:
            # Force cleanup of any remaining thread resources
            import gc
            gc.collect()

    def _process_with_bedrock_integration(self):
        """
        Process document with Bedrock data automation integration.
        This method handles Bedrock-specific processing logic.
        """
        try:
            self.logger.info("Starting Bedrock data automation processing")

            # Validate Bedrock project
            response = self.bedrock_client.get_data_automation_project(
                projectArn=self.bedrock_project_arn
            )

            project_name = response.get('project', {}).get('projectName', 'Unknown')
            self.logger.info(f"Using Bedrock project: {project_name}")

            self.logger.info("Bedrock processing completed - integrating with standard pipeline")


        except Exception as e:
            self.logger.error(f"Bedrock integration processing failed: {str(e)}")
            raise

    def cleanup(self):
        """
        Clean up temporary files and directories.
        """
        self.directory_manager.cleanup_temp_files()


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python document_processor.py <input_file_path> <input_bucket> <doc_category> <doc_language> <input_filename> [bedrock_project_arn]")
        sys.exit(1)

    input_file_path = sys.argv[1]
    input_bucket = sys.argv[2]
    doc_category = sys.argv[3]
    doc_language = sys.argv[4]
    input_filename = sys.argv[5]
    bedrock_project_arn = sys.argv[6] if len(sys.argv) > 6 else None

    processor = DocumentProcessor(
        input_file_path,
        input_bucket,
        doc_category,
        doc_language,
        input_filename,
        bedrock_project_arn=bedrock_project_arn
    )
    processor.process()

