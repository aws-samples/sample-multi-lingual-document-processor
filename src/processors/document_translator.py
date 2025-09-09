"""
Document Translator class for translating text content to different languages.
"""

import json
import time
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import boto3.exceptions
from src.config.prompts import translation_prompt
from src.config.config import MODEL_ID

class DocumentTranslator:
    """
    Handles translation of text content to different languages using Bedrock models.
    """
    
    def __init__(self, model_id=MODEL_ID):
        """
        Initialize the DocumentTranslator with the specified model.
        
        Args:
            model_id: The Bedrock model ID to use for translation
        """
        self.model_id = model_id
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
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
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            return json.loads(response['body'].read())
        except Exception as e:
            if 'ThrottlingException' in str(e):
                self.logger.info("Rate limit hit. Waiting before retry...")
                # nosemgrep: arbitrary-sleep
                time.sleep(30)
            raise
    
    def translate_text(self, text_content: str, output_language: str, bedrock_runtime) -> str:
        """
        Translate text content to the specified output language.
        
        Args:
            text_content: The text to translate
            output_language: Target language for translation
            bedrock_runtime: The Bedrock runtime client
            
        Returns:
            Translated text
        """
        try:
            # Skip empty or whitespace-only content
            if not text_content or not text_content.strip():
                return text_content
                
            # Format the translation prompt with the content and target language
            formatted_prompt = translation_prompt.format(
                text_content=text_content,
                output_language=output_language
            )
            
            # Prepare the request body
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
            
            # Extract the translated text from the response
            translated_text = ""
            for content_item in response_body.get('content', []):
                if content_item.get('type') == 'text':
                    translated_text += content_item.get('text', '')
                    
            return translated_text if translated_text else text_content
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            # Return the original text if translation fails
            return text_content
