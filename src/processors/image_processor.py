"""
Image Processor class for handling image operations.
"""

import os
import logging
import glob
import json
import time
import boto3
import base64
import numpy as np
import cv2
import pypdfium2 as pdfium
from PIL import Image
from io import BytesIO
from pypdf import PdfReader
from botocore.config import Config
from src.config.config import MODEL_ID, MAX_RETRIES
from src.config.prompts import image_narration_prompt, table_definition, graph_definitions, histogram, bar_chart, pie_chart, line_graph, scatter_plot, infographics, natural_image, percentage_stacked_bar_chart


class ImageProcessor:
    """
    Handles image operations like converting PDF pages to images and extracting images.
    """
    
    def __init__(self):
        """
        Initialize the image processor.
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Configure boto3 with longer timeout and retry settings
        self.config = Config(
            read_timeout=200,
            retries=dict(max_attempts=5)
        )
        
        # Initialize token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def page_to_image(self, pdf_path):
        """
        Convert all PDF pages into images.
        
        Args:
            pdf_path: Path to the PDF file
        """
        # Create output directory if it doesn't exist
        output_dir = "/tmp/pages_in_images"
        os.makedirs(output_dir, exist_ok=True)
        
        pdf = pdfium.PdfDocument(pdf_path)
        
        for i in range(len(pdf)):
            # Save pages as images in the pdf
            page = pdf[i]
            image = page.render(scale=4).to_pil()
            output_file = os.path.join(
                output_dir,
                f"page{i + 1}.png"
            )
            image.save(output_file)
            self.logger.info(f"Saved image: {output_file}")
    
    def process_images_from_pdf(self, page_image, output_language):
        """
        Extract images from PDFs and generate narrations for the extracted images.
        
        Args:
            input_path: Directory containing PDF files
            file_path: Directory for storing and processing images
            
        Returns:
            dict: Token usage statistics
        """
        # Reset token counters for this processing run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # for items in im_dir:
        results = self.image_narration_gen(page_image, output_language)
        
        if results:
            self.total_input_tokens += results.get('input_tokens', 0)
            self.total_output_tokens += results.get('output_tokens', 0)

        # Return token usage statistics
        self.logger.info(f"Total image processing token usage - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
        return results
    
    def image_extraction(self, pdf_page):
        """
        Extract images from PDF files that aren't scanned documents.
        
        Args:
            pdf_url: List of PDF file paths
        """
        # Create output directory if it doesn't exist
        # output_dir = "/tmp/data_in_images"
        # os.makedirs(output_dir, exist_ok=True)

        text_pages = []
        image_count = []
        
        try:
            # for files in pdf_url:
            try:
                # Check if file is scanned
                if not self.IsScanned(pdf_page):

                    # Get image count
                    img_cnt = self.isImage(pdf_page)
                    self.logger.info(f"Image count for {img_cnt}")

                    text_pages.append(pdf_page)
                    image_count.append(img_cnt)

            except Exception as e:
                self.logger.error(f"Error processing file {str(e)}")
            
            # Process files with images
            if text_pages is not None and image_count is not None:
                min_count = min(image_count)
                min_value_ind = image_count.count(min_count)
                
                # for i in range(len(text_pages)):
                # Determine which files to process based on image count
                should_process = (
                    (min_value_ind == 1 and image_count[0] > 0) or
                    (min_value_ind > 1 and image_count[0] > min_count)
                )

                if should_process:
                    pdf = pdfium.PdfDocument(f"{text_pages[i]}")

                    try:
                        for j in range(len(pdf)):
                            try:
                                page = pdf[j]
                                image = page.render(scale=4).to_pil()

                                # Create output filename
                                base_name = os.path.splitext(os.path.basename(text_pages[i]))[0]
                                output_file = os.path.join(
                                    output_dir,
                                    f"{base_name}_page{j+1}.png"
                                )

                                # Save the image
                                image.save(output_file)
                                self.logger.info(f"Saved image: {output_file}")

                            except Exception as e:
                                self.logger.error(f"Error processing page {j} of {text_pages[i]}: {str(e)}")
                                continue

                    finally:
                        pass
                    
        except Exception as e:
            self.logger.error(f"Error in image extraction: {str(e)}")
            raise

    def isImage(self, page):
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
            self.logger.error(f"Error processing PDF: {e}")
            return 0
    
    def get_image_bytes(self, image):
        """
        Get base64 encoded image bytes.
        
        Args:
            image: PIL Image
            
        Returns:
            str: Base64 encoded image data
        """
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def IsScanned(self, pdf_page, text_threshold=50):
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
        text_extraction_result = self.check_text_extraction(pdf_page, text_threshold)
        
        # if text_extraction_result is True or text_extraction_result is False:
        return text_extraction_result
        
        # # Method 2: Analyze with image properties (not implemented here)
        # else:
        #     return None
    
    def check_text_extraction(self, pdf_page, text_threshold):
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
            self.logger.error(f"Error during text extraction check: {e}")
            return None
    
    def invoke_claude_3_with_text_and_image(self, prompt, image_data, max_retries=MAX_RETRIES, base_delay=1):
        """
        Invokes Anthropic Claude 3 with text and image inputs with retry logic.
        
        Args:
            prompt: Text prompt for the model
            image_data: Base64 encoded image data
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            dict: Model response with token usage
        """
        # Initialize the Amazon Bedrock runtime client
        client = boto3.client(
            service_name="bedrock-runtime", 
            region_name="us-east-1", 
            config=self.config
        )
        
        retries = 0
        while retries < max_retries:
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
                
                self.logger.info(f"- The input length is {input_tokens} tokens.")
                self.logger.info(f"- The output length is {output_tokens} tokens.")
                
                # Add token usage to result
                result['token_usage'] = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
                
                return result
                
            except Exception as e:
                retries += 1
                
                # Check if we've reached max retries
                if retries >= max_retries:
                    self.logger.error(f"Failed to call Claude API after {max_retries} attempts: {str(e)}")
                    raise

                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** retries)
                self.logger.warning(f"Claude API call failed: {str(e)}. Retrying ({retries}/{max_retries}) in {delay:.2f} seconds...")
                # nosemgrep: arbitrary-sleep
                time.sleep(delay)
        
        # This should not be reached due to the raise in the exception handler
        # But adding as a fallback
        raise Exception(f"Failed to invoke Claude model after {max_retries} attempts")

    
    def image_narration_gen(self, img, output_language):
        """
        Generate narration for an image using Bedrock model.
        
        Args:
            img: Image object to analyze
            output_language: Language for the output narration
            
        Returns:
            dict: Contains narration, content, and token usage information
        """
        token_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
        
        result = {
            'narration': "",
            'content': None,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
        
        try:
            # Get width and height 
            width = img.width 
            height = img.height 
            
            if width > 30 and height > 30:
                from src.processors.utils import image_to_base64
                image_data = image_to_base64(img)
                
                prompt = image_narration_prompt.format(
                    table_definition=table_definition,
                    graph_definitions=graph_definitions,
                    histogram=histogram,
                    bar_chart=bar_chart,
                    pie_chart=pie_chart,
                    line_graph=line_graph,
                    scatter_plot=scatter_plot,
                    infographics=infographics,
                    natural_image=natural_image,
                    percentage_stacked_bar_chart=percentage_stacked_bar_chart,
                    output_language=output_language
                )
                
                # Get narration from model
                res = self.invoke_claude_3_with_text_and_image(prompt, image_data)
                
                # Extract token usage
                if 'token_usage' in res:
                    token_usage['input_tokens'] = res['token_usage']['input_tokens']
                    token_usage['output_tokens'] = res['token_usage']['output_tokens']
                    token_usage['total_tokens'] = token_usage['input_tokens'] + token_usage['output_tokens']
                
                # Format the narration content
                narration = f"Starting Image Extraction...\nFigure name: {res['content'][0]['text']}\nImage Extraction Completed\n"
                
                # Build result in the same format as table content
                result = {
                    'narration': narration,
                    'content': res['content'],  # Store the raw content from the model
                    'input_tokens': token_usage['input_tokens'],
                    'output_tokens': token_usage['output_tokens'],
                    'total_tokens': token_usage['input_tokens'] + token_usage['output_tokens']
                }
                result['image'] = img
            return result
        
        except Exception as e:
            self.logger.error(f"Error generating narration for image: {str(e)}")
            return result
