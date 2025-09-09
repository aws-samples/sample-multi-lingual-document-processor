import boto3
import json
from botocore.config import Config
from textractor.parsers import response_parser
import natsort
from botocore.exceptions import ClientError
from src.config.prompts import confidence_example
import os
from litellm import Router
from src.config.config import CONFIDENCE_MODEL_ID
from io import BytesIO

class ConfidenceScoreProcessor:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.router = self.llm_model()
        self.keys_to_extract = ["Text", "Confidence"]
        adaptive_config = Config(
            retries={"max_attempts": 100, "mode": "adaptive"}, max_pool_connections=20 * 3
        )
        self.textract_client = boto3.client(
            "textract", region_name="us-east-1", config=adaptive_config
        )

    def llm_model(self):
        """
        Creates and configures a Router instance for load balancing LLM requests across multiple AWS regions.

        This function sets up a Router with multiple model endpoints across different AWS regions for
        redundancy and load distribution. Each endpoint is configured with the same model but in different
        geographical locations to provide failover capabilities and improved latency.

        Returns:
            Router: A configured Router instance that handles request distribution across the defined endpoints.
                The Router is configured with:
                - 100 retry attempts
                - 100 second cooldown between retries
                - 100,000 tokens per minute (tpm) rate limit per endpoint

        Example:
            >>> router = llm_model('claude-3-7-sonnet-20250219-v1:0')
            >>> response = router.completion(...)
        """
        model_list = [
            {
                "model_name": CONFIDENCE_MODEL_ID,
                "litellm_params": {
                    "model": f"bedrock/us.anthropic.{CONFIDENCE_MODEL_ID}",
                    "aws_region_name": "us-east-1",  # US East (N. Virginia)
                },
                "tpm": 100000,  
            },
            {
                "model_name": CONFIDENCE_MODEL_ID,
                "litellm_params": {
                    "model": f"bedrock/us.anthropic.{CONFIDENCE_MODEL_ID}",
                    "aws_region_name": "us-west-2",  # US West
                },
                "tpm": 100000,  
            },
            {
                "model_name": CONFIDENCE_MODEL_ID,
                "litellm_params": {
                    "model": f"bedrock/us.anthropic.{CONFIDENCE_MODEL_ID}",
                    "aws_region_name": "us-east-2",  # US East
                },
                "tpm": 100000,  
            }
        ]
        router = Router(model_list=model_list, num_retries=100, cooldown_time=100)
        return router


    def run_model(self, input_texts, llm_router=None):
        """
        Processes OCR text from PDF documents using an LLM model to extract structured information.

        Args:
            input_texts (str): OCR text extracted from PDF pages
            llm_router (Router, optional): Router instance for LLM model calls. Defaults to None

        Returns:
            dict: Parsed JSON containing extracted information with confidence scores, or None if parsing fails
                  Format example:
                  {
                    "carrier": {
                        "text": "SOUTHLAKE SPECIALTY INSURANCE COMPANY",
                        "confidence": 97.61912536621094
                    },
                    "run_date": {
                        "text": "12/18/2024",
                        "confidence": 97.49634552001953
                    }
                  }

        Raises:
            json.JSONDecodeError: If the model output cannot be parsed as valid JSON
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""You are an insurance-claims extraction specialist.  
                                    Given OCR text {input_texts} from a PDF, convert everything you see into structured JSON.
                                    Follow the instructions below to complete the task: 
                                    <instructions>
                                    - **Capture every detail** that appears in the text, omit nothing and invent nothing.
                                    - For each extracted value, MUST include a corresponding confidence score field immediately after the value.
                                    - Claim Number consists only of digits; never mistake letters for numbers.
                                    - Ensure the JSON is properly formatted with no syntax errors.
                                    - For example, the output should look like {confidence_example}.
                                    </instructions>
                                    """,
                }
            ],
        }
        messages = [message]

        response = llm_router.completion(
            model=CONFIDENCE_MODEL_ID,
            messages=messages,
            temperature=0.1,
            max_tokens=128_000,
        )
        json_str = response.choices[0].message.content

        json_str = json_str.split("```json")[1]
        json_str = json_str.split("```")[0]
        json_str = json_str.strip()

        try:
            return json.loads(json_str), response
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None, None

    def extract_from_pdf(self, pdf_image):
        """
        Extracts text and structured information from PDF files using AWS Textract and LLM processing.

        This function processes PDF files from an input folder, extracts text using AWS Textract,
        and uses an LLM model to parse the extracted text into structured JSON format. It can process
        PDFs either page-by-page or as complete documents.

        The function creates the following outputs:
        - JPEG images of each PDF page
        - JSON files containing the structured information extracted by the LLM

        The output JSON files contain extracted information with confidence scores.

        Raises:
            Various exceptions may be raised during PDF processing, file I/O, or API calls.
            These are caught and logged, allowing processing to continue with the next file.
        """
        result = {
            'narration': '',
            'content': '',  # Store the raw content from the model
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
        try:
            img_buffer = BytesIO()
            pdf_image.save(img_buffer, format='PNG')  # or 'JPEG'
            textract_result = self.textract_client.detect_document_text(
                Document={"Bytes": img_buffer.getvalue()}
            )
            parsed_result = response_parser.parse(textract_result)
            document_texts = parsed_result.text

            blocks = textract_result["Blocks"]
            new_dict = [
                {k: block[k] for k in self.keys_to_extract if k in block}
                for block in blocks
            ]
            new_dict = [
                str(d).replace("{", "\n").replace("}", "\n").replace("'", "")
                for d in new_dict
                if d
            ]
            document_texts += "\n".join(new_dict)

            res, response = self.run_model(
                document_texts,
                llm_router=self.router,
            )
            result['narration'] = f"Starting PDF item and corresponding confidence score Extraction...\nExtracted value and confidence score: {res}\nPDF item and corresponding confidence score Extraction Completed\n"
            result['content'] = res
            result['input_tokens'] = response.usage.get('prompt_tokens', 0),
            result['output_tokens'] = response.usage.get('completion_tokens', 0)
            result['total_tokens'] = response.usage.get('total_tokens', 0)

        except Exception as e:
            print(f"Error processing PDF {str(e)}")

        return result