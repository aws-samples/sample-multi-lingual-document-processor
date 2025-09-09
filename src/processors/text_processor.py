"""
Text Processor class for handling text extraction operations.
"""

import os
import glob
import json
import time
import logging
import boto3
import base64
import sys
from PIL import Image
from io import BytesIO
from botocore.config import Config
from langdetect import detect, DetectorFactory
from src.config.config import MODEL_ID, AWS_REGION
from src.config.prompts import text_narration_prompt
from src.processors.image_processor import ImageProcessor
from src.utils.postprocessing import extract_paragraphs
#from src.config.config import BDA_arn
from src.config.config import MAX_RETRIES
from src.processors.utils import download_s3_file_to_memory, generate_full_report_from_BDA
from src.config.config import OUTPUT_DIR, S3_BUCKET
from pathlib import Path
from src.processors.utils import upload_to_s3

# Set seed for deterministic language detection
DetectorFactory.seed = 0


def check_text_extraction(pdf_page, text_threshold):
    """
    Check if text can be extracted directly from PDF.
    
    Args:
        pdf_path: Path to the PDF file
        sample_pages: Number of pages to analyze
        text_threshold: Minimum character count threshold
        
    Returns:
        bool: True if scanned, False if native, None if inconclusive
    """
    try:
        # with open(pdf_path, 'rb') as file:
        # reader = PdfReader(file)
        # total_pages = len(reader.pages)
        # pages_to_check = min(sample_pages, total_pages)

        # # Choose pages from beginning, middle and end
        # page_indices = [0]
        # if total_pages > 2:
        #     page_indices.append(total_pages // 2)
        # if total_pages > 1:
        #     page_indices.append(total_pages - 1)
        #
        # # Limit to the requested sample size
        # page_indices = page_indices[:pages_to_check]

        # text_content = []
        # for page_idx in page_indices:
        text = pdf_page.extract_text().strip()
        # text_content.append(text.strip())

        # If we found sufficient text in all sampled pages, it's likely not scanned
        if len(text) > text_threshold:
            return False

        # If we found no text at all, it's likely scanned
        if len(text) < text_threshold//2:
            return True

        # Otherwise, result is inconclusive, proceed to image analysis
        return None
                
    except Exception as e:
        return None
def IsScanned(pdf_page, text_threshold=50):
    """
    Check if a PDF is likely scanned or contains native text.
    
    Args:
        pdf_path: Path to the PDF file
        sample_pages: Number of pages to analyze (for large PDFs)
        text_threshold: Minimum character count to consider a page as containing text
        image_analysis_threshold: Threshold for image-based analysis
        
    Returns:
        bool: True if PDF is likely scanned, False if likely digital/native
    """
    # Method 1: Check for text extraction
    text_extraction_result = check_text_extraction(pdf_page, text_threshold)
    
    # if text_extraction_result is True or text_extraction_result is False:
    return text_extraction_result
def isImage(page):
    """
    Count the number of images in a PDF document.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        int: Total number of images found in the PDF
    """
    image_count = 0
    
    try:
        # Open the PDF file
        # pdf = PdfReader(pdf_path)
        
        # Iterate through all pages
        # for page_num in range(len(pdf.pages)):
        # page = pdf.pages[page_num]

        # Get the page content as a dictionary
        if '/Resources' in page and '/XObject' in page['/Resources']:
            x_objects = page['/Resources']['/XObject']

            # Loop through each object to find images
            for obj_name in x_objects:
                x_object = x_objects[obj_name]
                if x_object['/Subtype'] == '/Image':
                    image_count += 1
        
        return image_count
    
    except Exception as e:
        return 0
class TextProcessor:
    """
    Handles text extraction operations from PDF files.
    """
    
    def __init__(self, bedrock_project_arn=None):
        """
        Initialize the text processor.
        
        Args:
            bedrock_project_arn: ARN of the Bedrock data automation project (optional)
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Configure boto3 with longer timeout and retry settings
        self.config = Config(
            read_timeout=200,
            retries=dict(max_attempts=MAX_RETRIES)
        )
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # BDA configuration
        self.aws_region = AWS_REGION
        if not self.aws_region:
            self.aws_region = "us-east-1"  # Default fallback
            self.logger.warning("AWS_REGION not set, using default: us-east-1")
        
        self.logger.info(f"TextProcessor initialized with region: {self.aws_region}")
        
        # Use Bedrock project ARN if provided, otherwise fall back to config
        #if bedrock_project_arn:
        self.project_arn = bedrock_project_arn
        self.logger.info(f"TextProcessor initialized with Bedrock project ARN: {bedrock_project_arn}")
        #else:
        #    self.project_arn = BDA_arn
        #    self.logger.info("TextProcessor initialized with default BDA ARN from config")
    
    def process_pdf_files(self, pdf_page, page_image, image_cons):
        """
        Process PDF files to extract text content.
        
        Returns:
            dict: Token usage statistics
        """
        # Reset token counters for this processing run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        self.logger.info("Starting PDF text processing")
        #if self.project_arn and self.project_arn != BDA_arn:
        #    self.logger.info(f"Using custom Bedrock project ARN for text processing: {self.project_arn}")
        #else:
        #    self.logger.info("Using default BDA configuration for text processing")

        img_cnt = isImage(pdf_page)
        self.logger.info(f"Image count for {img_cnt}")

        if img_cnt == 1:
            self.logger.info("Processing single image PDF")

            if not IsScanned(pdf_page):
                self.logger.info(f"Processing scanned file: {pdf_page}")

                # Process image file with narration and track tokens
                result = self.narration_gen(page_image)
                print("Text narration is:", result)
                self.total_input_tokens += result.get('input_tokens', 0)
                self.total_output_tokens += result.get('output_tokens', 0)

            else:
                try:
                    image_content = pdf_page.extract_text()

                except Exception as e:
                    self.logger.error(f"Error processing file {str(e)}")
                    image_content = ''

                narration = (
                    f"Starting Text Extraction...\n"
                    f"{image_content}\n"
                    f"Text Extraction Completed"
                )
                result = {
                    'narration':narration,
                    'content': image_content,  # Store the raw content from the model
                    'input_tokens': self.total_input_tokens,
                    'output_tokens': self.total_output_tokens,
                    'total_tokens': self.total_input_tokens + self.total_output_tokens
                }
                

        else:

            try:
                image_content = pdf_page.extract_text()

            except Exception as e:
                self.logger.error(f"Error processing file {str(e)}")
                image_content = ''

            narration = (
                f"Starting Text Extraction...\n"
                f"{image_content}\n"
                f"Text Extraction Completed"
            )
            result = {
                'narration': narration,
                'content': image_content,  # Store the raw content from the model
                'input_tokens': self.total_input_tokens,
                'output_tokens': self.total_output_tokens,
                'total_tokens': self.total_input_tokens + self.total_output_tokens
            }
        
        # Return token usage statistics
        self.logger.info(f"Total text processing token usage - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
        
        return result
    
    def narration_gen(self, img):
        """
        Generate text narration for an image.
        
        Args:
            doc: Image filename
            output_path: Output directory for narration
            
        Returns:
            dict: Token usage information
        """
        # Ensure output directory exists
        # os.makedirs(output_path, exist_ok=True)
        
        token_usage = {
            'input_tokens': 0,
            'output_tokens': 0
        }

        result = {
            'narration': "",
            'content': "",
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
        
        try:
            # img = Image.open(file_path + doc)
            
            # Get width and height
            width = img.width
            height = img.height
            
            if width > 30 and height > 30:
                # with open(file_path + doc, "rb") as f:
                #     base64_encoded_image = base64.b64encode(f.read())
                from src.processors.utils import image_to_base64
                image_data = image_to_base64(img)

                # image_data = base64_encoded_image.decode("utf-8")
                
                # Add delay to avoid rate limiting
                # nosemgrep: arbitrary-sleep
                # time.sleep(60)
                
                # Get narration from model
                res = self._invoke_claude_3_with_text_and_image(text_narration_prompt, image_data)
                
                # Extract token usage
                if 'usage' in res:
                    token_usage['input_tokens'] = res['usage']['input_tokens']
                    token_usage['output_tokens'] = res['usage']['output_tokens']
                
                # Prepare output file
                # output_file = os.path.join(output_path, f"text_results_{doc.split('.')[0]}.txt")
                
                narration = (
                    f"Starting Text Extraction...\n"
                    f"{res['content'][0]['text']}\n"
                    f"Text Extraction Completed"
                )

                # Build result in the same format as table content
                result = {
                    'narration': narration,
                    'content': res['content'],  # Store the raw content from the model
                    'input_tokens': token_usage['input_tokens'],
                    'output_tokens': token_usage['output_tokens'],
                    'total_tokens': token_usage['input_tokens'] + token_usage['output_tokens']
                }
                print("Text extracted from the page:", result)
            
            else:
                self.logger.info(f"Skipping image too small")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing {str(e)}")
            return result
    
    def _invoke_claude_3_with_text_and_image(self, prompt, image_data):
        """
        Invoke Claude 3 with text and image inputs, with exponential backoff retries.
        
        Args:
            prompt: Text prompt
            image_data: Base64 encoded image data
            
        Returns:
            dict: Model response with token usage
        """
        # Initialize the Amazon Bedrock runtime client
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            config=self.config
        )
        
        # Set up retry parameters
        base_delay = 2  # base delay in seconds
        
        # Try the API call with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                # Invoke Claude 3 with the text prompt and image
                response = client.invoke_model(
                    modelId=MODEL_ID,
                    body=json.dumps(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 4000,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": image_data,
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": prompt
                                        }
                                    ],
                                }
                            ],
                        }
                    ),
                )
                
                # Process and return the response
                result = json.loads(response.get("body").read())
                input_tokens = result["usage"]["input_tokens"]
                output_tokens = result["usage"]["output_tokens"]
                
                # Add to running totals
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                
                self.logger.info(f"- The input length is {input_tokens} tokens.")
                self.logger.info(f"- The output length is {output_tokens} tokens.")
                
                return result
                
            except (client.exceptions.ThrottlingException, 
                    client.exceptions.ServiceUnavailableException,
                    client.exceptions.ModelTimeoutException,
                    client.exceptions.ModelStreamErrorException,
                    client.exceptions.ValidationException) as e:
                
                # Calculate exponential backoff with jitter
                if attempt < MAX_RETRIES - 1:  # Don't sleep on the last attempt
                    # Calculate delay: base_delay * 2^attempt + random jitter
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"API call failed with {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
                    # nosemgrep: arbitrary-sleep
                    time.sleep(delay)
                else:
                    self.logger.error(f"API call failed after {MAX_RETRIES} attempts: {str(e)}")
                    raise
                    
            except Exception as e:
                # For other exceptions, don't retry
                self.logger.error(f"Unexpected error calling Claude API: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise Exception(f"Failed to invoke Claude model after {MAX_RETRIES} attempts")



class BedrockDataAutomationClient:
    """
    Client for interacting with Bedrock Data Automation.
    """
    
    def __init__(self, region=AWS_REGION, s3_bucket=None, 
                 project_arn="",
                 input_path="bda-input", output_path="bda-input"):
        """
        Initialize the Bedrock Data Automation client.
        
        Args:
            region: AWS region
            s3_bucket: S3 bucket name
            project_arn: Bedrock Data Automation project ARN
            input_path: S3 input path
            output_path: S3 output path
        """
        self.logger = logging.getLogger()
        self.region = region
        self.s3_bucket = s3_bucket
        self.project_arn = project_arn
        self.s3_bucket_input_prefix = input_path
        self.s3_bucket_output_prefix = output_path
        
        # Log project ARN information
        #if project_arn != BDA_arn:
        #    self.logger.info(f"BedrockDataAutomationClient initialized with custom project ARN: {project_arn}")
        #else:
        #    self.logger.info("BedrockDataAutomationClient initialized with default BDA ARN")
        
        self.bda_client = boto3.client(
            "bedrock-data-automation-runtime", region_name=self.region
        )
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.sts_client = boto3.client("sts")
        self.aws_account_id = self.sts_client.get_caller_identity().get("Account")

    
    def upload_file_to_s3(self, input_file_path):
        """
        Upload a file to S3 bucket.
        
        Args:
            input_file_path: Path to the input file
            
        Returns:
            str: S3 URI of the uploaded file
        """
        file_name = os.path.basename(input_file_path)
        s3_key = f"{self.s3_bucket_input_prefix}/{file_name}"
        
        try:
            self.s3_client.upload_file(input_file_path, self.s3_bucket, s3_key)
            return f"s3://{self.s3_bucket}/{s3_key}"
        except Exception as e:
            print(f"Error uploading file to S3: {str(e)}")
            raise
    
    def invoke_data_automation(self, input_file_path):
        """
        Invoke Bedrock Data Automation on a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            dict: Invocation response
        """

        file_name = os.path.basename(input_file_path)
        input_file = f"s3://{self.s3_bucket}/{self.s3_bucket_input_prefix}/{file_name}" #self.upload_file_to_s3(file_path)
        print("text_processor input_file: ", input_file)
        output_file = f"s3://{self.s3_bucket}/{self.s3_bucket_output_prefix}"
        print("text_processor output_file: ", output_file)
        
        # Get the Bedrock Data Automation role ARN from environment
        bedrock_role_arn = os.environ.get('BEDROCK_DATA_AUTOMATION_ROLE_ARN')
        print("bedrock_role_arn: ", bedrock_role_arn)
        
        params = {
            "inputConfiguration": {"s3Uri": input_file},
            "outputConfiguration": {"s3Uri": output_file},
            "dataAutomationConfiguration": {
                "dataAutomationProjectArn": self.project_arn
            },
            "dataAutomationProfileArn": f"arn:aws:bedrock:{self.region}:{self.aws_account_id}:data-automation-profile/us.data-automation-v1",
        }
        
        # Note: executionRoleArn is not a valid parameter for invoke_data_automation_async
        # The BDA service will use the role that calls the API (SageMaker role)
        if bedrock_role_arn:
            self.logger.info(f"Bedrock role available but not used in API call: {bedrock_role_arn}")
        
        # Log the parameters for debugging
        self.logger.info(f"BDA Parameters: {params}")
        self.logger.info(f"Input S3 URI: {input_file}")
        self.logger.info(f"Output S3 URI: {output_file}")
        self.logger.info(f"Project ARN: {self.project_arn}")

        response = self.bda_client.invoke_data_automation_async(**params)
        return response
        

    def wait_for_data_automation_to_complete(self, invocation_arn, loop_time_in_seconds=1):
        """
        Wait for a Bedrock Data Automation job to complete.
        
        Args:
            invocation_arn: Invocation ARN
            loop_time_in_seconds: Time to wait between status checks
            
        Returns:
            dict: Final status response
        """

        # Set up retry parameters
        base_delay = 2  # base delay in seconds

        while True:  # Continue polling until job completes
            # Try the API call with exponential backoff
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.bda_client.get_data_automation_status(invocationArn=invocation_arn)
                    status = response["status"]
                    if status not in ["Created", "InProgress"]:
                        print(f" {status}")
                        return response
                    else:
                        # Job still in progress, wait and poll again
                        # nosemgrep: arbitrary-sleep
                        time.sleep(loop_time_in_seconds)
                        break  # Break the retry loop, but continue the outer polling loop
                        
                except (self.bda_client.exceptions.ThrottlingException,
                        self.bda_client.exceptions.ServiceUnavailableException,
                        self.bda_client.exceptions.ModelTimeoutException,
                        self.bda_client.exceptions.ModelStreamErrorException,
                        self.bda_client.exceptions.ValidationException) as e:

                    if attempt < MAX_RETRIES - 1:  # Don't sleep on the last attempt
                        # Calculate delay: base_delay * 2^attempt
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"API call failed with {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
                        # nosemgrep: arbitrary-sleep
                        time.sleep(delay)
                    else:
                        self.logger.error(f"API call failed after {MAX_RETRIES} attempts: {str(e)}")
                        raise

                except Exception as e:
                    # For other exceptions, don't retry
                    self.logger.error(f"Unexpected error calling BDA: {str(e)}")
                    raise

            # If we exhausted all retries due to throttling, but need to continue polling
            if attempt == MAX_RETRIES - 1:
                raise Exception(f"Failed to poll BDA status after {MAX_RETRIES} attempts")

    def run_bda(self, input_file_path, pdf_images, image_processor, table_processor):
        """
        Run Bedrock Document Analysis on a PDF file.

        Args:
            file_name: Path to the PDF file
            pdf_images: PDF as images
        """
        print("input_file_path: ", input_file_path)
        invocation_arn = self.invoke_data_automation(input_file_path)
        self.logger.info(f"BDA Processing....{invocation_arn}")

        response = self.wait_for_data_automation_to_complete(
            invocation_arn["invocationArn"]
        )

        s3_uri = response['outputConfiguration']['s3Uri']
        print("s3_uri: ", s3_uri)
        meta_data = download_s3_file_to_memory(s3_uri)
        print("meta_data: ", meta_data)
        for i in range(len(meta_data['output_metadata'])):
            for j in range(len(meta_data['output_metadata'][i]['segment_metadata'])):
                doc_bda_content = download_s3_file_to_memory(meta_data['output_metadata'][i]['segment_metadata'][j]['standard_output_path'])
                print("doc_bda_content: ", doc_bda_content)
                final_report, input_tokens, output_tokens = generate_full_report_from_BDA(doc_bda_content, pdf_images, image_processor, table_processor)


        self.logger.info(f"Job completed with status: {response['status']}")
        return final_report, input_tokens, output_tokens

