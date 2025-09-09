import os
import json
import base64
import boto3
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw
from botocore.client import Config
from botocore.exceptions import ClientError
from pdf2image import convert_from_path
import multiprocessing
from functools import partial
import time
import re
import random
from datetime import datetime
import io
from src.config.config import AWS_REGION



# Standalone helper functions for multiprocessing
def extract_table_section_content_worker(section_info):
    """Worker function to extract content from a table section (used by multiprocessing)"""
    # page_num = section_info['page_num']
    table_id = section_info['table_id']
    section_id = section_info['section_id']
    total_sections = section_info['total_sections']
    # section_path = section_info['image_path']
    model_id = section_info['model_id']
    # caption = section_info.get('caption', f"Table {table_id} on page {page_num}")
    
    # List of available regions for bedrock
    regions = [
        "us-east-1",
        "us-west-2",
        "us-west-2",
        "us-east-2",
        "eu-west-3",
        "eu-central-1",
        "eu-west-1"
    ]
    
    # Use a different region for each section based on section_id
    region = regions[section_id % len(regions)]
    
    
    prompt = """
    You are a table extraction specialist. Your task is to precisely extract the table content from the provided image.
    Guidelines:
    1. Extract the table structure and content as accurately as possible.
    2. Maintain the exact structure of the table (rows, columns, headers).
    3. Preserve all data in the cells exactly as they appear, including:
    - Numbers (integers, decimals, percentages, scientific notation, currencies)
    - Text and descriptions (preserve formatting, capitalization, and abbreviations)
    - Symbols and special characters (mathematical symbols, units, etc.)
    - For images within cells, describe them as [IMAGE: brief description] or [ICON: brief description]
    - For complex visualizations in cells (like charts/graphs), note as [CHART: brief description]
    4. Format the extracted table in markdown format.
    5. Be precise with numerical data:
    - Maintain all decimal places as shown
    - Preserve negative signs, plus signs, and percentage symbols
    - Keep scientific notation intact (e.g., 1.2E-6)
    - Maintain currency symbols and formatting
    6. Preserve the language accurately, including non-English text.
    7. For tables split across multiple images:
    - Be aware that rows might be cut off at the top or bottom
    - Note where content appears to be truncated with [...]
    8. Focus on extracting the visible portion accurately, clearly identifying:
    - Headers and subheaders
    - Row labels and nested hierarchies
    - Data cells and their alignment
    9. If the table has caption, title, footnotes, or source information, extract these and clearly label them (e.g., "Caption:", "Footnotes:").

    Return only the extracted table content in markdown format without any additional analysis, explanation or notes.
    """
    model_tokens = {}
    model_tokens['input_tokens'] = 0
    model_tokens['output_tokens'] = 0
    try:
        # image = Image.open(section_path)
        image = section_info['image_section_table']
        
        # Prepare the image for API
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        # Structure the query
        section_desc = f"section {section_id} of {total_sections}" if total_sections > 1 else "entire table"
        query_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Extract the table content from this image ({section_desc}):"},
                {"type": "image", 
                 "source": {
                     "type": "base64",
                     "media_type": "image/jpeg",
                     "data": img_str.decode('utf-8')
                 }
                }
            ]
        }]
        # Create bedrock config and client for this specific section
        bedrock_config = Config(
            connect_timeout=300, 
            read_timeout=300, 
            retries={'max_attempts': 5}, 
            region_name=region
        )
        bedrock_client = boto3.client('bedrock-runtime', config=bedrock_config)
        # Call the Bedrock Claude API
        body = {
            "system": prompt,
            "messages": query_messages,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20000,
            "temperature": 0,
            "top_k": 1,
            "stop_sequences": ["Human:"]
        }
        print(f"  Extracting {section_desc} from Table {table_id} in region {region}...")
        
        # Use the local client we just created
        attempt = 0
        max_retries = 10
        base_backoff = 1
        if 'eu' in region and 'us' in model_id:
            model_id = model_id.replace('us', 'eu')
        if 'us' in region and 'eu' in model_id:
            model_id = model_id.replace('eu', 'us')
        
        while attempt <= max_retries:
            try:
                response = bedrock_client.invoke_model(
                    body=json.dumps(body).encode(),
                    accept="application/json",
                    contentType="application/json",
                    modelId=model_id
                )
                
                input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
                output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
                model_tokens['input_tokens'] = input_tokens
                model_tokens['output_tokens'] = output_tokens
                break  # Success, break out of the loop
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException' and attempt < max_retries:
                    attempt += 1
                    # Calculate backoff with jitter
                    max_backoff = base_backoff * (2 ** attempt)
                    jitter = random.uniform(0, 0.1 * max_backoff)
                    backoff_time = max_backoff + jitter
                    print(f"  Throttling error occurred. Retrying in {backoff_time:.2f} seconds (attempt {attempt}/{max_retries})...")
                    # nosemgrep: arbitrary-sleep
                    time.sleep(backoff_time)
                else:
                    # Either it's not a ThrottlingException or we've exceeded max retries
                    print(f"  Error invoking model: {str(e)}", model_id, region)
                    raise  # Re-raise the exception to be caught by the outer try-except

        # If we get here and attempt > max_retries, we've failed all retries
        if attempt > max_retries:
            raise Exception(f"Failed to invoke model after {max_retries} attempts")

        
        response_body = json.loads(response["body"].read())
        section_content = response_body['content'][0]['text']
        
        # Save the extracted section content
        # sections_dir = f'{output_dir}/table_sections'
        # os.makedirs(sections_dir, exist_ok=True)
        # section_file = f'{sections_dir}/page_{page_num}_table_{table_id}_section_{section_id}.md'
        
        # with open(section_file, 'w', encoding='utf-8') as f:
        #     if section_id == 1 and caption:  # Only add caption to first section
        #         f.write(f"# {caption}\n\n")
        #     f.write(section_content)
        
        return {
            'table_id': table_id,
            'section_id': section_id,
            'content': section_content,
            # 'file_path': section_file,
            'has_content': True, 
            'input_tokens': model_tokens['input_tokens'],
            'output_tokens': model_tokens['output_tokens']
        }
    except Exception as e:
        print(f"Error extracting section {section_id} of table {table_id} on page: {str(e)}")
        # Create a placeholder for error case
        # sections_dir = f'{output_dir}/table_sections'
        # os.makedirs(sections_dir, exist_ok=True)
        # section_file = f'{sections_dir}/page_{page_num}_table_{table_id}_section_{section_id}_error.md'
        
        # with open(section_file, 'w', encoding='utf-8') as f:
        #     f.write(f"# Error extracting section {section_id} of table {table_id} on page {page_num}\n\n{str(e)}")
        
        return {
            'table_id': table_id,
            'section_id': section_id,
            'content': f"Error extracting section: {str(e)}",
            # 'file_path': section_file,
            'has_content': False,
            'error': str(e), 
            'input_tokens': model_tokens['input_tokens'],
            'output_tokens': model_tokens['output_tokens']}


class PDFTableExtractor:
    """ A class for extracting tables from PDF documents using AWS Bedrock and Claude models.

    This class handles:
    - PDF to image conversion
    - Table detection using computer vision
    - Table extraction and formatting
    - Handling large tables by splitting and reassembling
    """

    def __init__(self, model_id="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1", 
                 retry_attempts=10, base_backoff=1):
        """
        Initialize the PDF Table Extractor.
        
        Args:
            model_id (str): The AWS Bedrock model ID to use
            region_name (str): AWS region to use
            output_dir (str): Directory to store all output files
            retry_attempts (int): Maximum number of retry attempts for API calls
            base_backoff (int): Base value for exponential backoff (in seconds)
        """
        self.model_id = model_id
        self.region_name = region_name
        self.MAX_RETRIES = retry_attempts
        self.BASE_BACKOFF_SECONDS = base_backoff
        
        
        # Create all directories
        # for dir_path in self.dirs.values():
        #     os.makedirs(dir_path, exist_ok=True)
        
        # Initialize AWS Bedrock Client
        self.bedrock_client = self._create_bedrock_client()
        
        # Set up logger
        import logging
        self.logger = logging.getLogger('PDFTableExtractor')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _create_bedrock_client(self):
        """Create a bedrock client that uses a different region each time it's called"""
        # List of AWS regions where Bedrock is available
        bedrock_regions = [
        "us-east-1",
        "us-west-2",
        "us-west-2",
        "us-east-2",
        "eu-west-3",
        "eu-central-1",
        "eu-west-1"
        ]
        
        
        # Get or initialize the region index
        if not hasattr(self, '_region_index'):
            self._region_index = 0
        
        # Select the current region and increment the index
        current_region = bedrock_regions[self._region_index]
        self._region_index = (self._region_index + 1) % len(bedrock_regions)
        
        # Create and return bedrock client with the selected region
        bedrock_config = Config(
            connect_timeout=1200, 
            read_timeout=1200, 
            retries={'max_attempts': 2}, 
            region_name=current_region
        )
        
        return boto3.client('bedrock-runtime', config=bedrock_config)


    def _exponential_backoff(self, attempt):
        """Calculate backoff time with jitter for retries"""
        max_backoff = self.BASE_BACKOFF_SECONDS * (2 ** attempt)
        jitter = random.uniform(0, 0.1 * max_backoff)  # 10% jitter
        return max_backoff + jitter

    def _invoke_textract_with_retry(self, img_bytes, max_retries=3):
        """Call Textract API with retry logic"""
        textract = boto3.client('textract', region_name = 'us-east-1')
        
        retries = 0
        while retries < max_retries:
            try:
                response = textract.analyze_document(
                    Document={'Bytes': img_bytes},
                    FeatureTypes=['TABLES']
                )
                return response
            
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"Failed to call Textract API after {max_retries} attempts: {str(e)}")
                    raise
                
                print(f"Textract API call failed, retrying ({retries}/{max_retries})...")
                # nosemgrep: arbitrary-sleep
                time.sleep(2 ** retries)  # Exponential backoff


    def _invoke_bedrock_with_retry(self, body, region=None):
        """Invoke bedrock model with retry logic for throttling exceptions"""
        attempt = 0
        model_tokens = {}
        model_tokens['input_tokens'] = 0
        model_tokens['output_tokens'] = 0
        # print("Model NAME:", self.model_id, region)
        
        if 'eu' in region and 'us' in self.model_id:
            self.model_id = self.model_id.replace('us', 'eu')
        if 'us' in region and 'eu' in self.model_id:
            self.model_id = self.model_id.replace('eu', 'us')
        # Use a specific region if provided, otherwise use the instance's client
        if region:
            # Create a bedrock client for the specified region
            bedrock_config = Config(
                connect_timeout=1200, 
                read_timeout=1200, 
                retries={'max_attempts': 2}, 
                region_name=region
            )
            client = boto3.client('bedrock-runtime', config=bedrock_config)
        else:
            client = self.bedrock_client
        
        while True:
            print("MODEL", self.model_id, region)
            try:
                response = client.invoke_model(
                    body=json.dumps(body).encode(),
                    accept="application/json",
                    contentType="application/json",
                    modelId=self.model_id
                )
                input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
                output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
                model_tokens['input_tokens'] = input_tokens
                model_tokens['output_tokens'] = output_tokens
                
                self.logger.info(f"- The input length is {input_tokens} tokens.")
                self.logger.info(f"- The output length is {output_tokens} tokens.")
                return response, model_tokens
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException' and attempt < self.MAX_RETRIES:
                    attempt += 1
                    backoff_time = self._exponential_backoff(attempt)
                    print(f"  Throttling error occurred. Retrying in {backoff_time:.2f} seconds (attempt {attempt}/{self.MAX_RETRIES})...")
                    # nosemgrep: arbitrary-sleep
                    time.sleep(backoff_time)
                else:
                    raise

    # def _clear_directory(self, dir_path):
    #     """Clear all files in a directory"""
    #     if os.path.isdir(dir_path):
    #         for filename in os.listdir(dir_path):
    #             file_path = os.path.join(dir_path, filename)
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)

    def _table_evaluator(self, table_content):
        prompt = """
        You are specialized in evaluating the quality of a table extracted from a document. Your task is to provide a single numerical score (0-10) for the accuracy of the given table.
        The score should reflect how well the data is structured, how accurate the columns and rows are captured, and how much of the important information is preserved.
        Important: Your response must contain only a single integer score between 0 and 10.
        """
        model_tokens = {}
        model_tokens['input_tokens'] = 0
        model_tokens['output_tokens'] = 0
        try:
            # Structure the query
            query_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Evaluate this table for quality (respond with only a single score 0-10):\n\n{table_content}"}
                ]
            }]
            
            # Call the Bedrock Claude API
            body = {
                "system": prompt,
                "messages": query_messages,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,  # Reduced since we only need a small response
                "temperature": 0,
                "top_k": 1,
                "stop_sequences": ["Human:"]
            }

            response, model_tokens = self._invoke_bedrock_with_retry(body, region=AWS_REGION)
            response_body = json.loads(response["body"].read())
            
            # Extract just the integer from the response
            response_text = response_body.get('content', [{}])[0].get('text', '0')
            # Find the first integer in the response
            import re
            match = re.search(r'\b([0-9]|10)\b', response_text)
            if match:
                score = int(match.group(0))
            else:
                score = 0
                
            return {'score': score, 'model_tokens': model_tokens}
        except Exception as e:
            print(f"  Error evaluating table: {str(e)}")
            return {
                'score': 0,
                'error': str(e),
                'model_tokens': model_tokens}

    

    def _detect_tables_using_textract(self, pdf_image):
        """Detect tables in a page using AWS Textract"""
        # page_num, image_path = page_info
        
        try:
            print(f"  Detecting tables on with AWS Textract...")
            
            # Load and get the image dimensions for later conversion
            
            img_width, img_height = pdf_image.size
            
            # Initialize Textract client
            # textract = boto3.client('textract', region_name = 'us-east-1')
            
            # Read the image file
            img_byte_arr = io.BytesIO()
            pdf_image.save(img_byte_arr, format=pdf_image.format if pdf_image.format else 'PNG')
            img_bytes = img_byte_arr.getvalue()

            
            # Call Textract's analyze_document API with TABLES feature
            response = self._invoke_textract_with_retry(img_bytes)
            
            # Process the Textract response to extract table bounding boxes
            tables = []
            table_id = 1
            
            # Find all table blocks in the response
            for block in response.get('Blocks', []):
                if block['BlockType'] == 'TABLE':
                    # Extract geometry information
                    if 'Geometry' in block:
                        bbox = block['Geometry']['BoundingBox']
                        
                        # Convert to [x1, y1, x2, y2] format in percentages (0-100)
                        x1 = bbox['Left'] * 100
                        y1 = bbox['Top'] * 100 
                        x2 = (bbox['Left'] + bbox['Width']) * 100
                        y2 = (bbox['Top'] + bbox['Height']) * 100
                        
                        # Add some margin around the table (2%)
                        margin = 2.0
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(100, x2 + margin)
                        y2 = min(100, y2 + margin)
                        
                        # Create table entry
                        table_entry = {
                            "table_id": table_id,
                            "bbox": [x1, y1, x2, y2],
                            "caption": f"Table {table_id}"
                        }
                        
                        tables.append(table_entry)
                        table_id += 1
            
            print(f"  Found {len(tables)} tables on this page.")
            for table in tables:
                x1, y1, x2, y2 = table['bbox']
                table['bbox'] = [
                    int(x1 * img_width / 100),
                    int(y1 * img_height / 100),
                    int(x2 * img_width / 100),
                    int(y2 * img_height / 100)
                ]

            # Create a debug image with bounding boxes in memory
            debug_img = pdf_image.copy()  # Using pdf_image instead of image
            draw = ImageDraw.Draw(debug_img)
            for table in tables:
                x1, y1, x2, y2 = table['bbox']
                draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
                draw.text((x1 + 10, y1 + 10), f"Table {table['table_id']}", fill="red")

            # Convert debug image to bytes instead of saving to disk
            debug_img_byte_arr = io.BytesIO()
            debug_img.save(debug_img_byte_arr, format='JPEG', quality=95)
            debug_img_bytes = debug_img_byte_arr.getvalue()

            # Convert original image to bytes
            img_byte_arr = io.BytesIO()
            pdf_image.save(img_byte_arr, format=pdf_image.format if pdf_image.format else 'PNG')
            img_bytes = img_byte_arr.getvalue()

            return {
                'image_bytes': img_bytes,  # Return bytes instead of file path
                'tables': tables,
                'debug_bytes': debug_img_bytes  # Return bytes instead of file path
            }

        
        except Exception as e:
            print(f"  Error detecting tables on page: {str(e)}")
            return {
                # 'image_path': image_path,
                'tables': [],
                'error': str(e)
            }


    def _split_table_into_sections(self, table_info, num_sections=4, overlap_percentage=20):
        """Split a table image into multiple sections with overlap"""
        try:
            table_id = table_info['table_id']
            table_img = table_info['table_img']
            
            
            img_width, img_height = table_img.size
            
            # If the image is very small or narrow, don't split
            min_dimension = 500  # Minimum pixel dimension to consider splitting
            if img_height < min_dimension or img_width < min_dimension:
                print(f"  Table {table_id} too small to split, processing as single section")
                # Create a copy with needed information for the multiprocessing worker
                section_info = table_info.copy()
                section_info.update({
                    'section_id': 1,
                    'total_sections': 1,
                    'original_table': table_info
                })
                return [section_info]
                
            # Calculate section height with overlap
            
            base_section_height = img_height // num_sections
            overlap_px = int(base_section_height * overlap_percentage / 100)
            
            sections = []
            for i in range(num_sections):
                # Calculate top position with overlap
                top = max(0, i * base_section_height - overlap_px if i > 0 else 0)
                
                # Calculate bottom position with overlap
                if i < num_sections - 1:
                    bottom = min(img_height, (i + 1) * base_section_height + overlap_px)
                else:
                    bottom = img_height
                
                # Create section image
                section = table_img.crop((0, top, img_width, bottom))
                # section_path = f"{self.dirs['tmp']}/page_{page_num}_table_{table_id}_section_{i+1}.jpeg"
                # section.save(section_path, 'JPEG', quality=95)
                
                # Create section info
                section_info = table_info.copy()  # Copy original table info
                section_info.update({
                    'section_id': i + 1,
                    'image_section_table': section,
                    'total_sections': num_sections,
                    'original_table': table_info,
                    'model_id': self.model_id,  # Needed for worker function
                })
                sections.append(section_info)
            
            return sections
            
        except Exception as e:
            print(f"Error splitting table image: {str(e)}")
            # Return the original table as a single section
            section_info = table_info.copy()
            section_info.update({
                'section_id': 1,
                'image_section_table': None,
                'total_sections': 1,
                'original_table': table_info,
                'model_id': self.model_id,  # Needed for worker function
                'error': str(e)
            })
            return [section_info]

    def _parallel_extract_section_content(self, sections):
        """Extract content from multiple table sections in parallel"""
        # Create a pool with multiple processes
        print('Multiprocessing for extracting Table from the sections')
        num_processes = min(len(sections), multiprocessing.cpu_count())
        
        # For small number of sections, avoid multiprocessing overhead
        if len(sections) == 1:
            return [extract_table_section_content_worker(sections[0])]
        
        # Ensure each section has all required information
        for section in sections:
            if 'model_id' not in section:
                section['model_id'] = self.model_id
        
        # Use multiprocessing safely with proper cleanup
        try:
            print(f"Extracting {len(sections)} sections in parallel using {num_processes} processes")
            
            # Try to terminate any existing zombie processes
            try:
                multiprocessing.active_children()  # This helps clean up lingering processes
            except:
                pass
            
            # Use 'spawn' method for multiprocessing which creates fresh processes
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                # If the method is already set, continue
                pass
            
            # Create a new pool with timeout handling
            pool = multiprocessing.Pool(processes=num_processes)
            
            try:
                # Use apply_async with timeout to prevent hanging processes
                results = []
                tasks = []
                
                # Submit all tasks
                for section in sections:
                    task = pool.apply_async(extract_table_section_content_worker, (section,))
                    tasks.append((task, section))
                
                # Collect results with timeout
                for task, section in tasks:
                    try:
                        # Set a generous timeout (5 minutes per section)
                        result = task.get(timeout=300)  
                        results.append(result)
                    except multiprocessing.TimeoutError:
                        print(f"  Timeout processing section {section['section_id']} of table {section['table_id']} on page")
                        # Create an error result for the timed-out task
                        results.append({
                            'table_id': section['table_id'],
                            'section_id': section['section_id'],
                            'content': f"Error: Processing timeout",
                            # 'file_path': f"{section['output_dir']}/table_sections/timeout_error.md",
                            'has_content': False,
                            'error': "Task exceeded timeout limit",
                            'input_tokens': 0,
                            'output_tokens': 0
                        })
                    except Exception as e:
                        print(f"  Error processing section: {str(e)}")
                        results.append({
                            'table_id': section['table_id'],
                            'section_id': section['section_id'],
                            'content': f"Error: {str(e)}",
                            # 'file_path': f"{section['output_dir']}/table_sections/error.md",
                            'has_content': False,
                            'error': str(e),
                            'input_tokens': 0,
                            'output_tokens': 0
                        })
                
                return results
            finally:
                # Make sure to close and terminate the pool
                pool.close()
                pool.terminate()
                pool.join()
                
        except Exception as e:
            print(f"Error in multiprocessing: {str(e)}")
            # Fall back to sequential processing
            print("Falling back to sequential processing")
            results = []
            for section in sections:
                results.append(extract_table_section_content_worker(section))
            return results



    def _extract_table_section_content(self, section_info):
        """Extract content from a table section (non-parallel version)"""
        # Just delegate to the worker function
        return extract_table_section_content_worker(section_info)

    def _consolidate_table_sections(self, table_info, section_results):
        """Consolidate content from multiple table sections into one coherent table"""
        table_id = table_info['table_id']
        caption = table_info.get('caption', f"Table {table_id} on page")
        
        # Sum up input and output tokens from each section
        total_input_tokens = sum(result.get('input_tokens', 0) for result in section_results)
        total_output_tokens = sum(result.get('output_tokens', 0) for result in section_results)
        
        # If there's only one section, no need to consolidate
        if len(section_results) == 1 and section_results[0].get('has_content', False):
            table_content = section_results[0]['content']
            
            # # Save the consolidated table content
            # table_file = f'{self.dirs["table_content"]}/page_{page_num}_table_{table_id}.md'
            
            # with open(table_file, 'w', encoding='utf-8') as f:
            #     if caption:
            #         f.write(f"# {caption}\n\n")
            #     f.write(table_content)
            
            # Convert markdown to pandas DataFrame
            df = self._markdown_to_dataframe(table_content)
            
            return {
                'table_id': table_id,
                'content': table_content,
                'caption': caption,
                # 'file_path': table_file,
                'has_table': True,
                'dataframe': df,
                'shape': df.shape if df is not None else None,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }
        
        # Combine section content in order
        sorted_sections = sorted(section_results, key=lambda x: x['section_id'])
        sections_content = ""
        for section in sorted_sections:
            if section.get('has_content', False):
                sections_content += f"### Section {section['section_id']}\n\n"
                sections_content += section['content'] + "\n\n"
        
        # If no valid sections were found, return error
        if not sections_content:
            error_msg = "All table sections failed extraction"
            # print(f"  {error_msg} for table {table_id} on page {page_num}")
            
            # table_file = f'{self.dirs["table_content"]}/page_{page_num}_table_{table_id}_error.md'
            
            # with open(table_file, 'w', encoding='utf-8') as f:
            #     f.write(f"# Error consolidating table {table_id} on page {page_num}\n\n{error_msg}")
            
            return {
                'table_id': table_id,
                'content': error_msg,
                'caption': caption,
                # 'file_path': table_file,
                'has_table': False,
                'dataframe': None,
                'shape': None,
                'error': error_msg,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }
        
        # If we have multiple sections, use Claude to consolidate them
        prompt = """
        You are a table consolidation specialist. I've provided you with sections of a table that was split due to its size.
        Your task is to combine these sections into a complete, coherent table without the caption in markdown format.
        
        Guidelines:
        1. Recognize where sections of the table overlap and remove duplicate rows.
        2. Maintain the table structure (headers, columns, rows) consistent across all parts.
        3. Preserve all data exactly as they appear, including numbers, text, and formatting.
        4. Format the final output as a single clean markdown table.
        5. If the sections appear to be from different tables, keep them as separate tables in your output.
        6. Preserve table headers correctly even if they appear in multiple sections.
        7. Be precise with numerical data, preserving decimal places, signs, and special characters.
        
        IMPORTANT: Return only the consolidated table in markdown format without additional explanation and captions.
        """
        
        try:
            # Structure the query
            query_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Consolidate these table sections into a complete table:\n\n{sections_content}"}
                ]
            }]
            
            # Call the Bedrock Claude API
            body = {
                "system": prompt,
                "messages": query_messages,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 20000,
                "temperature": 0,
                "top_k": 1,
                "stop_sequences": ["Human:"]
            }
            
            print(f"  Merging sections for table {table_id} on ONE page...")
            
            response, model_tokens = self._invoke_bedrock_with_retry(body, region = 'eu-north-1')
            response_body = json.loads(response["body"].read())
            table_content = response_body['content'][0]['text']
            
            # Add token counts from consolidation step
            total_input_tokens += model_tokens['input_tokens']
            total_output_tokens += model_tokens['output_tokens']
            
            # Save the consolidated table content
            # table_file = f'{self.dirs["table_content"]}/page_{page_num}_table_{table_id}.md'
            
            # with open(table_file, 'w', encoding='utf-8') as f:
            #     if caption:
            #         f.write(f"# {caption}\n\n")
            #     f.write(table_content)
            
            # Convert markdown to pandas DataFrame
            df = self._markdown_to_dataframe(table_content)
            
            return {
                # 'page_num': page_num,
                'table_id': table_id,
                'content': table_content,
                'caption': caption,
                # 'file_path': table_file,
                'has_table': True,
                'dataframe': df,
                'shape': df.shape if df is not None else None,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }
        except Exception as e:
            print(f"  Error consolidating table {table_id} on page : {str(e)}")
            
            # Save the raw sections for debugging
            # table_file = f'{self.dirs["table_content"]}/page_{page_num}_table_{table_id}_consolidation_error.md'
            
            # with open(table_file, 'w', encoding='utf-8') as f:
            #     f.write(f"# Error consolidating table {table_id} on page {page_num}\n\n{str(e)}\n\n")
            #     f.write("## Raw Section Content\n\n")
            #     f.write(sections_content)
            
            return {
                # 'page_num': page_num,
                'table_id': table_id,
                'content': f"Error consolidating table: {str(e)}",
                'caption': caption,
                'section_content': sections_content, 
                # 'file_path': table_file,
                'has_table': False,
                'dataframe': None,
                'shape': None,
                'error': str(e),
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }

    def _markdown_to_dataframe(self, markdown_text):
        """Convert markdown table to pandas DataFrame using buffer instead of file I/O"""
    
        # Create a StringIO buffer with the markdown text
        buffer = io.StringIO(markdown_text)
        
        rows = []
        for row in buffer.readlines():
            # Skip non-table rows (doesn't start with |)
            if not row.strip().startswith('|'):
                continue
                
            # Get rid of leading and trailing '|'
            # The [1:-2] assumes the last character is a newline followed by '|'
            # If the last character is just '|', we should use [1:-1]
            if row.strip().endswith('|'):
                if row.endswith('|\n'):
                    tmp = row.strip()[1:-1]  # Remove | and newline
                else:
                    tmp = row[1:-1]  # Just remove |
            else:
                continue  # Skip malformed rows
            
            # Split line and ignore column whitespace
            clean_line = [col.strip() for col in tmp.split('|')]
            
            # Append clean row data to rows variable
            rows.append(clean_line)
        
        # Get rid of syntactical sugar to indicate header (2nd row)
        # Make sure we have at least 3 rows (header, separator, and data)
        if len(rows) >= 3:
            # Check if second row looks like a separator (contains - or : characters)
            if all(('-' in cell or ':' in cell) for cell in rows[1]):
                rows = rows[:1] + rows[2:]
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        return df



    def _extract_full_page(self, pdf_image):
        """Extract tables from a full page when no tables were detected"""
        
        prompt = """
        You are a table extraction specialist. Your task is to extract all tables from the provided image.
        
        Guidelines:
        1. Identify and extract all tables present in the image.
        2. Maintain the exact structure of each table (rows, columns, headers).
        3. Preserve all data in the cells exactly as they appear.
        4. Format the extracted tables in markdown format.
        5. If there are multiple tables, separate them clearly with headers indicating "Table 1", "Table 2", etc.
        6. Include any table captions or titles that are visible.
        7. If no tables are found in the image, clearly state "No tables found in this image."
        
        Return only the extracted tables in markdown format without any additional analysis or explanation.
        """
        
        try:
            # Prepare the image for API
            buffered = BytesIO()
            pdf_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            
            # Structure the query
            query_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all tables from this page:"},
                    {"type": "image", 
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_str.decode('utf-8')
                    }
                    }
                ]
            }]
            
            # Call the Bedrock Claude API
            body = {
                "system": prompt,
                "messages": query_messages,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 20000,
                "temperature": 0,
                "top_k": 1,
                "stop_sequences": ["Human:"]
            }
            
            print(f"  Extracting tables from full page...")
            
            response, model_tokens = self._invoke_bedrock_with_retry(body, region=AWS_REGION)
            response_body = json.loads(response["body"].read())
            page_content = response_body['content'][0]['text']
            
            has_tables = "No tables found" not in page_content
            
            # Save the extracted content
            # page_file = f'{self.dirs["page_tables"]}/page_{page_num}_full.md'
            
            # with open(page_file, 'w', encoding='utf-8') as f:
            #     f.write(f"# Page {page_num}\n\n")
            #     f.write(page_content)
            
            # Extract DataFrames if tables found
            dataframes = []
            
            if has_tables:
                # Find markdown tables
                # table_pattern = r'\|.+\|[\r\n]+\|(?:\s*[-:]+\s*\|)+[\r\n]+(?:\|.+\|[\r\n]+)+'
                # tables = re.findall(table_pattern, page_content)
                
                # for i, md_table in enumerate(tables):
                df = self._markdown_to_dataframe(page_content)
                if df is not None:
                    dataframes.append({
                        'table_id': i + 1,
                        'data': df,
                        'shape': df.shape
                    })
        
            return {
                # 'page_num': page_num,
                'content': page_content,
                # 'file_path': page_file,
                'has_tables': has_tables,
                'dataframes': dataframes, 
                'input_tokens': model_tokens['input_tokens'],
                'output_tokens': model_tokens['output_tokens']
            }
        except Exception as e:
            print(f"  Error extracting from full page: {str(e)}")
            
            # Create error file
            # page_file = f'{self.dirs["page_tables"]}/page_{page_num}_error.md'
            
            # with open(page_file, 'w', encoding='utf-8') as f:
            #     f.write(f"# Error extracting from page {page_num}\n\n{str(e)}")
            
            return {
                # 'page_num': page_num,
                'content': f"Error extracting page: {str(e)}",
                # 'file_path': page_file,
                'has_tables': False,
                'dataframes': [],
                'error': str(e), 
                'input_tokens': 0,
                'output_tokens': 0
            }



    def _process_page(self, pdf_image, num_sections=4):
        """Process a page by detecting tables, splitting them, and extracting content"""
        
        print(f"Processing page")
        
        try:
            # Step 1: Detect tables on the page
            detection_result = self._detect_tables_using_textract(pdf_image)
            tables = detection_result.get('tables', [])
            
            # Step 2: If no tables detected, process the entire page
            if not tables:
                print(f"  No tables detected, processing full page...")
                full_page_result = self._extract_full_page(pdf_image)
                
                extracted_tables = []
                if full_page_result.get('has_tables', False) and full_page_result.get('dataframes', []):
                    for df_info in full_page_result['dataframes']:
                        table_id = df_info['table_id']
                        extracted_tables.append({
                            'table_id': table_id,
                            'title': f"Table {table_id}",
                            'method': "full_page",
                            'shape': df_info['shape'],
                            'confidence': 0.7,  # Moderate confidence for page-level extraction
                            'data': df_info['data'],
                            'claude_analysis': full_page_result,
                            'extraction_timestamp': datetime.now().isoformat()
                        })
                
                return {
                    # 'page_num': page_num,
                    'tables': extracted_tables,
                    # 'output_path': full_page_result['file_path'],
                    'has_tables': full_page_result.get('has_tables', False),
                    'content': full_page_result.get('content', ''),
                    'input_tokens': full_page_result.get('input_tokens', 0), 
                    'output_tokens': full_page_result.get('output_tokens', 0)
                }
            
            # Step 3: Process each detected table
            all_table_results = []
            page_content = f"# Tables from ONE Page\n\n"
            page_has_tables = False
            extracted_tables = []
            total_input_tokens = 0
            total_output_tokens = 0
            
            for table in tables:
                try:
                    table_id = table['table_id']
                    x1, y1, x2, y2 = table['bbox']
                    caption = table.get('caption', f"Table {table_id}")
                    
                    # Load original page image
                    # page_image = Image.open(pdf_image)
                    
                    # Crop the table from the page image
                    table_img = pdf_image.crop((x1, y1, x2, y2))
                    
                    # Save the cropped table image
                    # table_img_path = f'{self.dirs["table_images"]}/page_{page_num}_table_{table_id}.jpeg'
                    # table_img.save(table_img_path, 'JPEG', quality=95)
                    
                    # Create table info
                    table_info = {
                        # 'page_num': page_num,
                        'table_id': table_id,
                        'table_img': table_img,
                        'caption': caption,
                        'model_id': self.model_id,
                        'region_name': self.region_name,
                        'bbox': [x1, y1, x2, y2]
                    }
                    
                    # Step 4: Split the table into sections
                    table_sections = self._split_table_into_sections(table_info, num_sections)
                    
                    # Step 5: Extract content from each section in parallel
                    section_results = self._parallel_extract_section_content(table_sections)
                    
                    # Step 6: Consolidate sections into one table
                    table_result = self._consolidate_table_sections(table_info, section_results)
                    
                    # Step 7: Add to page results and accumulate token counts
                    total_input_tokens += table_result.get('input_tokens', 0)
                    total_output_tokens += table_result.get('output_tokens', 0)
                    
                    if table_result.get('has_table', False):
                        page_has_tables = True
                        page_content += f"## {caption}\n\n"
                        page_content += table_result['content'] + "\n\n"
                        
                        # Add to extracted tables list with pandas DataFrame
                        extracted_tables.append({
                            'table_id': table_id,
                            'title': caption,
                            'method': "detected_table",
                            'shape': table_result.get('shape'),
                            'confidence': 0.9,  # Higher confidence for table detection
                            'data': table_result.get('dataframe'),
                            'claude_analysis': table_result,
                            'extraction_timestamp': datetime.now().isoformat()
                        })
                    
                    all_table_results.append(table_result)
                    
                except Exception as e:
                    print(f"  Error processing table {table['table_id']}: {str(e)}")
            
            # Step 8: Save page content
            if not page_has_tables:
                page_content += "No tables found on this page.\n"
            
            # page_file = f'{self.dirs["page_tables"]}/page_{page_num}_tables.md'
            # with open(page_file, 'w', encoding='utf-8') as f:
            #     f.write(page_content)
            
            return {
                # 'page_num': page_num,
                'tables': extracted_tables,
                # 'output_path': page_file,
                'has_tables': page_has_tables,
                'content': page_content,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }
        except Exception as e:
            print(f"  Error processing page : {str(e)}")
            
            # Create error page
            # error_path = f'{self.dirs["page_tables"]}/page_{page_num}_error.md'
            # with open(error_path, 'w', encoding='utf-8') as f:
            #     f.write(f"# Error processing page {page_num}\n\n{str(e)}")
            
            return {
                # 'page_num': page_num,
                'tables': [],
                # 'output_path': error_path,
                'has_tables': False,
                'error': str(e),
                'input_tokens': 0,
                'output_tokens': 0
            }

    def extract_tables(self, pdf_image, num_sections=6):
        """
        Extract tables from a PDF file.
        
        Args:
            pdf_file (str): Path to the PDF file
            output_path (str, optional): Path to save final output, if None uses default location
            num_sections (int): Number of sections to split large tables into
            clear_output (bool): Whether to clear existing output directories
            
        Returns:
            list: List of extracted tables with their content including pandas DataFrames
        """
        start_time = time.time()
        # print(f"Processing PDF: {pdf_image}")
        
        
        # Convert PDF to images
        # image_paths = self._convert_pdf_to_images(pdf_file)
        
        # Process each page
        page_results = []
        extracted_tables = []
        input_tokens = 0
        output_tokens = 0
        # for page_num, img_path in image_paths:
        # page_info = (page_num, img_path)
        print("Processing Pages............")
        page_result = self._process_page(pdf_image, num_sections=num_sections)
        input_tokens += page_result.get('input_tokens', 0)
        output_tokens += page_result.get('output_tokens', 0)
        # Collect tables from this page
        page_results.append(page_result)
        if 'tables' in page_result and page_result['tables']:
            extracted_tables.extend(page_result['tables'])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Table extraction complete. Processed {1} page in {elapsed_time:.2f} seconds.")
        print(f"Extracted {len(extracted_tables)} tables.")
        model_tokens = {}
        model_tokens['input_tokens'] = input_tokens
        model_tokens['output_tokens'] = output_tokens
        return extracted_tables, model_tokens
