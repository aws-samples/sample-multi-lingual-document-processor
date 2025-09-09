"""
Localization Processor class for handling content localization operations.
"""

import os
import re
import json
import time
import logging
import boto3
import boto3.exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from src.config.config import MODEL_ID
from src.config.prompts import localization_prompt, table_definition, graph_definitions, histogram, bar_chart, pie_chart, line_graph, scatter_plot
from src.processors.image_processor import ImageProcessor
from src.utils.postprocessing import process_file
import shutil
from src.config.config import AWS_REGION, MAX_RETRIES
class LocalizationProcessor:
    """
    Handles content localization operations using Bedrock models.
    """
    
    def __init__(self, boto3_config):
        """
        Initialize the localization processor.
        
        Args:
            boto3_config: Boto3 configuration for AWS clients
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.boto3_config = boto3_config
        
        # Initialize token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def natural_sort_key(self, s):
        """
        Natural sort key for sorting filenames with numbers.
        
        Args:
            s: String to sort
            
        Returns:
            list: List of components for natural sorting
        """
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]
    
    @retry(
        retry=retry_if_exception_type((
            boto3.exceptions.Boto3Error,
            json.JSONDecodeError,
            Exception
        )),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def invoke_bedrock_model(self, bedrock_runtime, request_body):
        """
        Invoke Bedrock model with retry logic.
        
        Args:
            bedrock_runtime: Bedrock runtime client
            request_body: Request body for the model
            
        Returns:
            dict: Model response
        """
        try:
            response = bedrock_runtime.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())
            
            # Extract and track token usage
            input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            self.logger.info(f"- The input length is {input_tokens} tokens.")
            self.logger.info(f"- The output length is {output_tokens} tokens.")
            
            # Add token usage to the response
            response_body['token_usage'] = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
            
            return response_body
        except Exception as e:
            if 'ThrottlingException' in str(e):
                self.logger.info("Rate limit hit. Waiting before retry...")
                # nosemgrep: arbitrary-sleep
                time.sleep(30)
            raise
    
    def process_document_pair(self, image, text, bedrock_runtime, prompt, output_language):
        """
        Process a document pair (image and text) using Bedrock model.
        
        Args:
            image: PIL Image object or None
            text: Text content to process
            bedrock_runtime: Bedrock runtime client
            prompt: Prompt for the model
            output_language: Language for output
            
        Returns:
            tuple: Model response text, token usage dictionary
        """
        try:
            formatted_prompt = prompt.format(
                text_content=text,
                table_definition=table_definition,
                graph_definitions=graph_definitions,
                histogram=histogram,
                bar_chart=bar_chart,
                pie_chart=pie_chart,
                line_graph=line_graph,
                scatter_plot=scatter_plot,
                output_language=output_language
            )
            
            # Prepare request body based on whether we have an image
            if image is not None:
                try:
                    # Convert image to base64
                    from src.processors.utils import image_to_base64
                    image_bytes = image_to_base64(image)
                    
                    # Request body with image
                    request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 5000,
                        "temperature": 0,
                        "top_p": 1,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": image_bytes
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": formatted_prompt
                                    }
                                ]
                            }
                        ]
                    }
                except Exception as img_err:
                    self.logger.error(f"Error processing image: {str(img_err)}")
                    # Fall back to text-only if image processing fails
                    request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 5000,
                        "temperature": 0,
                        "top_p": 1,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": formatted_prompt
                                    }
                                ]
                            }
                        ]
                    }
            else:
                # Text-only request body
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 5000,
                    "temperature": 0,
                    "top_p": 1,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": formatted_prompt
                                }
                            ]
                        }
                    ]
                }
            
            # Use the retry-enabled function
            response_body = self.invoke_bedrock_model(bedrock_runtime, request_body)
            
            # Extract token usage
            token_usage = response_body.get('token_usage', {
                'input_tokens': 0,
                'output_tokens': 0
            })
            
            response_text = response_body.get('content')[0].get('text') if response_body.get('content') else None
            return response_text, token_usage
        
        except Exception as e:
            self.logger.error(f"Error processing document pair: {str(e)}")
            return None, {'input_tokens': 0, 'output_tokens': 0}

    
    
    def main_Bedrock_arrange(self, final_report, page_num, output_language):
        """
        Main function to arrange content using Bedrock model.
        
        Returns:
            dict: Token usage statistics
        """
        # Reset token counters for this processing run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Initialize Bedrock client with configured timeout and retry settings
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            config=self.boto3_config
        )
        
        
        # Get and sort files
        image = final_report.get('image')
        text = final_report.get('report_text', '')
        
        
        # Process each pair
        self.logger.info(f"Processing {image} with {text}...")
        # Extract page number from image filename
        # page_num = page_num
            
            
        # Process the pair with retry logic
        prompt = localization_prompt
            
        max_attempts = MAX_RETRIES
            
        for attempt in range(max_attempts):
            try:
                result, token_usage = self.process_document_pair(image, text, bedrock_runtime, prompt, output_language)
                
                if result:
                    self.logger.info("Document processed successfully")
                    # Track token usage
                    self.total_input_tokens += token_usage.get('input_tokens', 0)
                    self.total_output_tokens += token_usage.get('output_tokens', 0)
                    break  # Success! Exit the retry loop
                else:
                    self.logger.warning(f"Attempt {attempt + 1} returned empty result")
                    raise ValueError("Empty result received from process_document_pair")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                
                # Check if we've reached the maximum attempts
                if attempt >= max_attempts - 1:
                    self.logger.error(f"All {max_attempts} attempts failed, giving up")
                    break
                    
                wait_time = (2 ** attempt)  
                self.logger.info(f"Waiting {wait_time} seconds before retry #{attempt + 2}...")
                # nosemgrep: arbitrary-sleep
                time.sleep(wait_time)

        # Return token usage statistics (even if all attempts failed)
        self.logger.info(f"Total localization token usage - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "result": result if 'result' in locals() else None
        }