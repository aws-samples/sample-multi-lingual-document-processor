
"""
Table Processor class for handling table extraction operations.
"""

import os
import glob
import logging
import json
import time
import boto3
from botocore.config import Config
from tabula import read_pdf
from tabulate import tabulate
from src.config.prompts import table_narration_prompt
from src.config.config import MODEL_ID
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures, Direction, DirectionalFinderType
from textractor.data.constants import TextractFeatures
from textractor.data.text_linearization_config import TextLinearizationConfig
from pathlib import Path
import pandas as pd
from src.processors.table_extraction_validator_new import validate_extracted_tables
import shutil
from src.config.config import tabula_model_ID, AWS_REGION
import os
import PyPDF2
import io

from src.processors.table_claude_ocr import PDFTableExtractor

extractor = PDFTableExtractor(
    model_id=tabula_model_ID,
    region_name=AWS_REGION)


class TableProcessor:
    """
    Handles table extraction operations from PDF files.
    """

    def __init__(self):
        """
        Initialize the table processor.
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
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            config=self.config
        )

    def process_single_page_tables(self, pdf_page, pdf_image, output_language):
        """
        Process tables from a single PDF page.

        Args:
            pdf_page: Single PDF page object
            pdf_image: Image representation of the PDF page
            output_language: Language for the narration output

        Returns:
            dict: Table extraction results containing narration, content, and token usage
        """
        table_result = {
            'narration': "",
            'content': [],
            'table_results': [],
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
        try:
            self.logger.info(f"Processing tables for single page {1}")

            # Extract tables directly from PDF page object
            table_result = self.get_tables_from_single_page(pdf_page, pdf_image, output_language)

            return table_result

        except Exception as e:
            self.logger.error(f"Error processing tables for page {1}: {str(e)}")
            return table_result


    def get_tables_from_single_page(self, pdf_page, pdf_image, output_language):
        """
        Extract tables from a single PDF page object.

        Args:
            pdf_page: Single PDF page object
            pdf_image: Image representation of the PDF page
            output_language: Language for the narration output

        Returns:
            dict: Contains narration, content, and token usage information
        """
        self.logger.info(f"Table extraction from single page {1}")

        table_narrations = []
        total_input_tokens = 0
        total_output_tokens = 0
        table_results = []

        # Extract tables directly from PDF page object using existing tabula logic
        model_tokens_table, extracted_tables = self.extract_tables_from_page_object(pdf_page, pdf_image)
        # validation_results = validate_extracted_tables(
        #             extracted_tables=extracted_tables,
        #             pdf_image=pdf_image
        #         )
        total_input_tokens += model_tokens_table['input_tokens']
        total_output_tokens += model_tokens_table['output_tokens']

        if len(extracted_tables):
            # Process each extracted table
            for i, table_data in enumerate(extracted_tables):
                if table_data['data'] is not None and len(table_data['data']) > 0:
                    # Format table using tabulate
                    table_formatted = tabulate(table_data['data'], headers='keys', tablefmt='grid', showindex=False)

                    # Generate narration for this table
                    narration, tokens_used = self.generate_table_narration(table_formatted, output_language=output_language)

                    total_input_tokens += tokens_used.get('input_tokens', 0)
                    total_output_tokens += tokens_used.get('output_tokens', 0)

                    if narration:
                        table_narration = f"Table {i+1}: {narration}"
                        table_narrations.append(table_narration)

                    # Add individual table result (if you want to track per-table information)
                    table_results.append({
                        'table_index': i,
                        'table_data': table_data,
                        'table_narration': narration if narration else ""
                    })

        # Create final result dictionary with narration, content and token usage
        result = {
            'narration': '\n'.join(table_narrations) if table_narrations else "",
            'content': extracted_tables,
            'table_results': table_results,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens
        }

        return result







        # except Exception as e:
        #     self.logger.error(f"Error extracting tables from page {1}: {str(e)}")
        #     return "", {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}


    def extract_tables_from_page_object(self, pdf_page, pdf_image):
        """
        Extract tables directly from PDF page object using existing tabula logic.

        Args:
            pdf_page: Single PDF page object
            page_number: Page number (1-based)

        Returns:
            List of pandas DataFrames containing extracted tables
        """
        extracted_tables = []


        # Import tabula for table extraction
        import tabula

        # Since tabula requires a file path, we need to work with the page object differently
        # For now, we'll use the existing tabula logic but adapted for single page

        # Try both stream and lattice methods as in the existing code
        methods = ["stream", "lattice"]
        total_input_tokens = 0
        total_output_tokens = 0
        for method in methods:
            try:
                pdf_buffer = io.BytesIO()
                # Write the single page to the in-memory buffer
                writer = PyPDF2.PdfWriter()
                writer.add_page(pdf_page)
                writer.write(pdf_buffer)
                pdf_buffer.seek(0)  # Reset buffer position to beginning
                if method == "stream":
                    # Use stream method (better for text-based detection)
                    tables = tabula.read_pdf(
                        pdf_buffer,
                        pages=1,  # Always use page 1 since we created a single-page PDF
                        stream=True,
                        guess=True,
                        multiple_tables=True
                    )

                else:
                    # Use lattice method (better for bordered tables)
                    tables = tabula.read_pdf(
                        pdf_buffer,
                        pages=1,  # Always use page 1 since we created a single-page PDF
                        stream=False,
                        guess=True,
                        multiple_tables=True
                    )

                if tables and len(tables) > 0:
                    # Filter out empty tables
                    valid_tables = [table for table in tables if not table.empty and table.shape[0] > 0]
                    extracted_tables.extend(valid_tables)

                    self.logger.info(f"✅ Extracted {len(valid_tables)} tables using {method} method")
                    break  # If we found tables, no need to try other methods

            except Exception as method_error:
                self.logger.warning(f"Method {method} failed: {str(method_error)}")
                continue

        if not extracted_tables:
            self.logger.info("❌ No tables extracted, falling back to alternative extraction")
            # You could add fallback logic here if needed

        score = extractor._table_evaluator(extracted_tables)
        self.logger.info(f"Prediction Score for LLM as Judge when Running on the Tabula Table Extractor is: {score['score']}")

        # Track tokens from table evaluator
        model_tokens_evaluator = score.get('model_tokens', {'input_tokens': 0, 'output_tokens': 0})
        total_input_tokens += model_tokens_evaluator.get('input_tokens', 0)
        total_output_tokens += model_tokens_evaluator.get('output_tokens', 0)
        model_tokens = {}
        model_tokens['input_tokens'] = total_input_tokens
        model_tokens['output_tokens'] = total_output_tokens
        if score['score'] < 5:
            # Use alternative table extraction method if score is low
            extracted_tables, model_tokens_table = extractor.extract_tables(pdf_image, num_sections=6)
            # Add tokens from advanced extraction

            total_input_tokens += model_tokens_table.get('input_tokens', 0)
            total_output_tokens += model_tokens_table.get('output_tokens', 0)
            model_tokens['input_tokens'] = total_input_tokens
            model_tokens['output_tokens'] = total_output_tokens

        return model_tokens, extracted_tables


    def generate_table_narration(self, table_content, output_language):
        """
        Generate narration for a single table using existing logic.

        Args:
            table_content: Formatted table content

        Returns:
            str: Table narration
        """
        try:
            prompt = self.create_zero_shot_prompt(table_content, output_language)
            result = self.invoke_claude_3_with_text(prompt)

            if result and 'content' in result and len(result['content']) > 0:
                return result['content'][0]['text'], result['token_usage']
            else:
                return "", {"input_tokens": 0, "output_tokens": 0}

        except Exception as e:
            self.logger.error(f"Error generating table narration: {str(e)}")
            return ""

    
    def create_zero_shot_prompt(self, table, output_language):
        """
        Create a prompt for table narration.

        Args:
            table: Table content to narrate

        Returns:
            str: Formatted prompt
        """
        prompt = table_narration_prompt.format(table=table, output_language=output_language)
        return prompt

    def invoke_claude_3_with_text(self, prompt, max_retries=10):
        """
        Invokes Anthropic Claude 3 Sonnet to run an inference with exponential backoff retry.

        Args:
            prompt: The prompt for Claude 3
            max_retries: Maximum number of retry attempts

        Returns:
            dict: Inference response from the model
        """
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            config=self.config,
        )

        retry_count = 0
        backoff_time = 1  # Start with 1 second backoff

        while retry_count <= max_retries:
            try:
                response = client.invoke_model(
                    modelId=MODEL_ID,
                    body=json.dumps(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 4000,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": prompt}],
                                }
                            ],
                        }
                    ),
                )

                result = json.loads(response.get("body").read())
                input_tokens = result["usage"]["input_tokens"]
                output_tokens = result["usage"]["output_tokens"]

                self.logger.info(f"- The input length is {input_tokens} tokens.")
                self.logger.info(f"- The output length is {output_tokens} tokens.")

                # Return token information along with result
                result['token_usage'] = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }

                return result

            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    self.logger.error(f"Maximum retries ({max_retries}) exceeded. Final error: {str(e)}")
                    raise

                # Log the error
                self.logger.warning(f"Error during model invocation (attempt {retry_count}/{max_retries}): {str(e)}")
                self.logger.info(f"Retrying in {backoff_time} seconds...")

                # Wait with exponential backoff
                # nosemgrep: arbitrary-sleep
                time.sleep(backoff_time)
                backoff_time *= 2


    def write_results_to_file(self, output_name, table, generated_text):
        """
        Write table narration results to a file.

        Args:
            output_name: Base name for the output file
            table: Table content
            generated_text: Generated narration text
        """
        output_file = output_name + ".txt"
        out_identifier = os.path.basename(output_file)
        print(table)
        new_content = (
            f"{out_identifier}\n"
            f"{table}\n"
            f"{generated_text['content'][0]['text']}\n"
        )

        # Check if file exists and read existing content
        existing_content = ""
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_content = f.read()

        # Only write if content doesn't already exist
        if new_content not in existing_content:
            # Use 'w' mode if file doesn't exist, otherwise use 'a'
            mode = "w" if not os.path.exists(output_file) else "a"
            with open(output_file, mode) as f2:
                f2.write(new_content)
            self.logger.info(f"Content written to {output_file}")
        else:
            self.logger.info("Content already exists in file - skipping write")

    def narrate_table(self, table, output_name, identifier):
        """
        Generate narration for a table using Bedrock model.

        Args:
            table: Table content to narrate
            output_name: Base name for the output file
            identifier: Table identifier

        Returns:
            dict: Token usage information
        """

        # Create prompt and invoke model
        prompt = self.create_zero_shot_prompt(table)
        generated_text = self.invoke_claude_3_with_text(prompt)
        print(generated_text)

        # Write results to file with duplicate checking
        self.write_results_to_file(output_name, table, generated_text)

        # Return token usage
        if 'token_usage' in generated_text:
            return {
                'input_tokens': generated_text['token_usage']['input_tokens'],
                'output_tokens': generated_text['token_usage']['output_tokens']
            }
        return None

    def table_textract(self, bucket_name, original_filename):
        extractor = Textractor(region_name="us-east-1")
        document = extractor.start_document_analysis(
            file_source= f"s3://{bucket_name}/input/{original_filename}",
            features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES],
            save_image=False
        )
        text = document.text
        return text

