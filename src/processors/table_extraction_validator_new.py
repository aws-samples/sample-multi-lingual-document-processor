#!/usr/bin/env python3
"""
Table Extraction Validator

Input:
- CSV file (extracted table)
- PDF page (single page or multi-page PDF with page number)

Output:
- Corrected table CSV
- Validation report
- Confidence score
"""

import csv
import json
import base64
import pandas as pd
import boto3
import os
import argparse
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import io
import time
from src.config.prompts import table_validation_prompt 
from src.config.config import table_validation_model_ID, AWS_REGION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_extracted_tables(extracted_tables, pdf_image):
    """
    Function to validate extracted tables from simple_text_claude_tabula.py
    This function can be called from table_processor.py
    
    Args:
        csv_files_list (list): List of CSV file paths from simple_text_claude_tabula
        pdf_file (str): Path to the original PDF file
        output_dir (str): Output directory for validation results
        page_number (int): Page number in PDF (0-indexed)
    
    Returns:
        dict: Validation results for all tables
    """
    
    try:
        # Initialize validator
        validator = TableExtractionValidator()
        
        # Create output directory
        
        validation_results = {}
        
        logger.info(f"Starting validation for {len(extracted_tables)} tables")
        
        for i, table in enumerate(extracted_tables):
            
            try:
                # Run validation for each table
                # nosemgrep: arbitrary-sleep
                time.sleep(60)
                result = validator.validate_single_table(
                    table=table,
                    pdf_image=pdf_image
                )
                
                # Store result with filename as key
                
                validation_results[table_name] = result
                
                logger.info(f"Validation completed for {table_name}")
                
            except Exception as table_error:
                table_name = os.path.basename(csv_file)
                logger.error(f"Error in table validation for {table_name}: {str(table_error)}")
                
                # Store error result
                validation_results[table_name] = {
                    "validation_summary": {
                        "status": "ERROR",
                        "confidence_score": 0.0,
                        "total_issues_found": 0,
                        "extraction_quality": "FAILED",
                        "error_message": str(table_error)
                    },
                    "issues_identified": [{
                        "issue_id": 1,
                        "type": "csv_parsing_error",
                        "severity": "critical",
                        "description": f"Failed to process CSV file: {str(table_error)}",
                        "location": "csv_loading"
                    }]
                }
                
                # Continue with other tables instead of failing completely
                continue
        
        
        summary = "Validation Successful"
        return summary
        
    except Exception as e:
        logger.error(f"Error in validate_extracted_tables: {str(e)}")
        return {
            "error": str(e),
            "total_tables_validated": 0,
            "validation_timestamp": datetime.now().isoformat()
        }

class TableExtractionValidator:
    def __init__(self, 
                 bedrock_region: str = AWS_REGION,
                 model_id: str = table_validation_model_ID):
        """
        Initialize the Table Extraction Validator
        
        Args:
            bedrock_region: AWS region for Bedrock
            model_id: Bedrock model ID for LLM analysis
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=bedrock_region)
        self.model_id = model_id
        
    def load_csv_table(self, csv_file_path: str) -> pd.DataFrame:
        """
        Load extracted table from CSV file with robust error handling
        
        Args:
            csv_file_path: Path to CSV table file
            
        Returns:
            DataFrame containing the extracted table
        """
        try:
            # Try different encodings to handle various languages
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # First attempt: Standard CSV reading
                    df = pd.read_csv(csv_file_path, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
                except pd.errors.ParserError as pe:
                    logger.warning(f"Parser error with {encoding} encoding: {str(pe)}")
                    # Try with error handling for malformed CSV
                    try:
                        df = pd.read_csv(
                            csv_file_path, 
                            encoding=encoding,
                            error_bad_lines=False,  # Skip bad lines
                            warn_bad_lines=True,    # Warn about skipped lines
                            on_bad_lines='skip'     # Skip malformed lines (pandas 1.3+)
                        )
                        logger.warning(f"Loaded CSV with {encoding} encoding, skipping malformed lines")
                        return df
                    except TypeError:
                        # Fallback for older pandas versions
                        try:
                            df = pd.read_csv(
                                csv_file_path, 
                                encoding=encoding,
                                error_bad_lines=False,
                                warn_bad_lines=True
                            )
                            logger.warning(f"Loaded CSV with {encoding} encoding using legacy error handling")
                            return df
                        except:
                            continue
                    except:
                        continue
            
            # If standard methods fail, try reading line by line and fixing inconsistencies
            logger.warning("Standard CSV reading failed, attempting manual parsing")
            df = self._manual_csv_parse(csv_file_path)
            if df is not None:
                return df
            
            # Final fallback: try with maximum flexibility
            try:
                df = pd.read_csv(
                    csv_file_path, 
                    encoding='utf-8', 
                    errors='replace',
                    sep=None,  # Auto-detect separator
                    engine='python',  # Use Python engine for more flexibility
                    quoting=3  # Ignore quotes
                )
                logger.warning("Loaded CSV with maximum flexibility settings")
                return df
            except Exception as final_error:
                logger.error(f"All CSV loading methods failed: {str(final_error)}")
                raise
            
        except Exception as e:
            logger.error(f"Error loading CSV table: {str(e)}")
            raise
    
    def _manual_csv_parse(self, csv_file_path: str) -> pd.DataFrame:
        """
        Manually parse CSV file to handle inconsistent column counts
        
        Args:
            csv_file_path: Path to CSV file
            
        Returns:
            DataFrame or None if parsing fails
        """
        try:
            import csv
            
            # Read all lines and determine maximum column count
            with open(csv_file_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
            
            if not lines:
                logger.warning("CSV file is empty")
                return pd.DataFrame()
            
            # Parse lines and find max columns
            parsed_rows = []
            max_cols = 0
            
            # Use csv.reader to handle quoted fields properly
            csv_reader = csv.reader(lines)
            for row_num, row in enumerate(csv_reader):
                parsed_rows.append(row)
                max_cols = max(max_cols, len(row))
            
            # Pad all rows to have the same number of columns
            normalized_rows = []
            for row in parsed_rows:
                # Pad with empty strings if row is shorter than max_cols
                padded_row = row + [''] * (max_cols - len(row))
                normalized_rows.append(padded_row[:max_cols])  # Truncate if longer
            
            # Create DataFrame
            if normalized_rows:
                # Use first row as header if it looks like headers, otherwise create generic headers
                if len(normalized_rows) > 1:
                    df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])
                else:
                    df = pd.DataFrame(normalized_rows, columns=[f'Column_{i+1}' for i in range(max_cols)])
                
                logger.info(f"Successfully parsed CSV manually with {len(normalized_rows)} rows and {max_cols} columns")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Manual CSV parsing failed: {str(e)}")
            return None
    
    def get_pdf_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages in the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception as e:
            logger.error(f"Error reading PDF page count: {str(e)}")
            raise
    
    def pdf_page_to_image(self, pdf_path: str, page_number: int = 0, dpi: int = 300) -> bytes:
        """
        Convert PDF page to image bytes using PyPDF2 and pdf2image
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number to convert (0-indexed)
            dpi: Resolution for image conversion
            
        Returns:
            Image bytes in PNG format
        """
        try:
            # First, validate the PDF and page number using PyPDF2
            page_count = self.get_pdf_page_count(pdf_path)
            
            if page_number >= page_count:
                raise ValueError(f"Page number {page_number} exceeds PDF page count {page_count}")
            
            # Convert specific page to image using pdf2image
            # Note: pdf2image uses 1-indexed pages, so we add 1
            images = convert_from_path(
                pdf_path, 
                dpi=dpi,
                first_page=page_number + 1,
                last_page=page_number + 1
            )
            
            if not images:
                raise ValueError(f"Failed to convert page {page_number} to image")
            
            # Get the first (and only) image
            image = images[0]
            
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            logger.info(f"Converted PDF page {page_number} to image ({len(img_bytes)} bytes) using PyPDF2/pdf2image")
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error converting PDF page to image: {str(e)}")
            raise
    
    def csv_to_plain_text(self, df: pd.DataFrame) -> str:
        """
        Convert CSV DataFrame to plain text format for LLM
        
        Args:
            df: DataFrame to convert
            
        Returns:
            Plain text representation of the table
        """
        try:
            # Get basic info about the table
            rows, cols = df.shape
            
            # Convert to string format with proper alignment
            table_str = df.to_string(index=False, max_rows=100, max_cols=20)
            
            # Add metadata
            plain_text = f"""
            TABLE INFORMATION:
            - Dimensions: {rows} rows × {cols} columns
            - Column Names: {', '.join(df.columns.tolist())}

            TABLE DATA:
            {table_str}
            """
            
            # Add sample of data types if helpful
            if len(df.columns) <= 10:  # Only for smaller tables
                dtypes_info = "\nDATA TYPES:\n"
                for col in df.columns:
                    sample_values = df[col].dropna().head(3).tolist()
                    dtypes_info += f"- {col}: {sample_values}\n"
                plain_text += dtypes_info
            
            return plain_text.strip()
            
        except Exception as e:
            logger.error(f"Error converting CSV to plain text: {str(e)}")
            # Fallback to simple string conversion
            return df.to_string(index=False)
    
    def encode_image_bytes_to_base64(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string for LLM processing
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Base64 encoded image string
        """
        try:
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image bytes to base64: {str(e)}")
            raise
    
    def encode_image_file_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string for LLM processing
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image file {image_path}: {str(e)}")
            raise
    
    def create_validation_prompt(self, table_text: str, source_info: str) -> str:
        """
        Create validation prompt for LLM
        
        Args:
            table_text: Plain text representation of the table
            source_info: Information about the source (CSV filename, PDF page, etc.)
            
        Returns:
            Validation prompt string
        """
        
        prompt = table_validation_prompt 
        return prompt
    
    def validate_table_with_llm(self, table, pdf_image: str) -> Dict:
        """
        Validate extracted table using LLM comparison with original PDF page or image
        
        Args:
            table: table file
            pdf_file: Path to PDF file (if provided)
            page_number: Page number in PDF (0-indexed)
            image_file: Path to image file (alternative to PDF)
            
        Returns:
            Dictionary containing validation results and corrections
        """
        try:
            # Load and process CSV
            
            df = pd.DataFrame(table)
            
            logger.info(f"Converting table to plain text (shape: {df.shape})")
            table_text = self.csv_to_plain_text(df)
            
            # Process image input
            
            image_bytes = pdf_image
            image_base64 = self.encode_image_bytes_to_base64(image_bytes)
            

            source_info = "from one image"
            # Create validation prompt
            prompt = self.create_validation_prompt(table_text, source_info)
            
            # Prepare request for Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "temperature": 0,  # Low temperature for consistent validation
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Call Bedrock API
            logger.info("Calling Bedrock API for table validation...")
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            llm_response = response_body['content'][0]['text']
            
            # Extract JSON from response
            try:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    validation_result = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                # Return error result with raw response
                validation_result = {
                    "validation_summary": {
                        "status": "ERROR",
                        "confidence_score": 0.0,
                        "total_issues_found": 0,
                        "extraction_quality": "UNKNOWN"
                    },
                    "issues_identified": [{
                        "issue_id": 1,
                        "type": "parsing_error",
                        "severity": "critical",
                        "description": f"Failed to parse LLM response: {str(e)}",
                        "location": "system"
                    }],
                    "raw_llm_response": llm_response
                }
            
            # Add metadata about the validation process
            validation_result['input_metadata'] = {
                "csv_file": csv_file,
                "pdf_file": pdf_file,
                "page_number": page_number,
                "image_file": image_file,
                "original_table_shape": df.shape,
                "original_columns": df.columns.tolist(),
                "validation_timestamp": datetime.now().isoformat()
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in table validation: {str(e)}")
            return {
                "validation_summary": {
                    "status": "ERROR",
                    "confidence_score": 0.0,
                    "total_issues_found": 0,
                    "extraction_quality": "UNKNOWN"
                },
                "issues_identified": [{
                    "issue_id": 1,
                    "type": "system_error",
                    "severity": "critical",
                    "description": str(e),
                    "location": "system"
                }],
                "error_details": str(e)
            }
    
    def save_corrected_table(self, validation_result: Dict, output_dir: str, base_name: str):
        """
        Save corrected table to CSV file - handles both simple and complex JSON structures
        
        Args:
            validation_result: Validation result containing corrected table
            output_dir: Output directory
            base_name: Base name for output files
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            corrected_table = validation_result.get('corrected_table')
            if not corrected_table:
                logger.warning("No corrected table data found in validation result")
                return None
            
            # Handle different JSON structure formats
            headers = None
            rows = None
            
            logger.info(f"Processing corrected table structure: {type(corrected_table)}")
            
            # Method 1: Simple format (headers + rows arrays)
            if 'headers' in corrected_table and 'rows' in corrected_table:
                headers = corrected_table['headers']
                rows = corrected_table['rows']
                
                # Check if headers are simple strings or complex objects
                if headers and isinstance(headers[0], dict):
                    # Extract header names from complex structure
                    headers = [h.get('name', f"Column_{h.get('index', i)}") 
                              for i, h in enumerate(headers)]
                    logger.info("Extracted headers from complex format")
                
                # Check if rows are simple arrays or complex objects
                if rows and isinstance(rows[0], dict):
                    # Handle complex row format with cell objects
                    simple_rows = []
                    for row in rows:
                        if 'cells' in row:
                            # Extract values from cell objects
                            row_data = []
                            for cell in row['cells']:
                                if isinstance(cell, dict):
                                    # Use formatted_value if available, otherwise value
                                    value = cell.get('formatted_value', cell.get('value', ''))
                                else:
                                    value = str(cell) if cell is not None else ''
                                row_data.append(value)
                            simple_rows.append(row_data)
                        elif isinstance(row, list):
                            # Already simple format
                            simple_rows.append(row)
                        else:
                            logger.warning(f"Unexpected row format: {type(row)}")
                    rows = simple_rows
                    logger.info("Extracted rows from complex format")
                
                logger.info("Using corrected_table format")
            
            # Method 2: Try alternative_simple_format if main format failed
            elif 'alternative_simple_format' in corrected_table:
                alt_format = corrected_table['alternative_simple_format']
                if 'headers' in alt_format and 'rows' in alt_format:
                    headers = alt_format['headers']
                    rows = alt_format['rows']
                    logger.info("Using alternative_simple_format")
            
            # Method 3: Handle case where corrected_table itself has the structure
            elif isinstance(corrected_table, dict) and len(corrected_table) == 2:
                # Check if it's a direct headers/rows structure
                keys = list(corrected_table.keys())
                if 'headers' in keys and 'rows' in keys:
                    headers = corrected_table['headers']
                    rows = corrected_table['rows']
                    logger.info("Using direct headers/rows structure")
            
            # Validate extracted data
            if not headers or not rows:
                logger.warning("Could not extract valid headers and rows from corrected table")
                logger.info(f"Available keys in corrected_table: {list(corrected_table.keys()) if isinstance(corrected_table, dict) else 'Not a dict'}")
                return None
            
            # Ensure headers are strings
            headers = [str(h) if h is not None else f"Column_{i}" for i, h in enumerate(headers)]
            
            # Ensure rows are properly formatted
            processed_rows = []
            for i, row in enumerate(rows):
                if isinstance(row, list):
                    # Convert all values to strings, handle None values
                    processed_row = [str(cell) if cell is not None else '' for cell in row]
                    processed_rows.append(processed_row)
                elif isinstance(row, dict):
                    # If row is still a dict, try to extract values
                    if 'cells' in row:
                        processed_row = [str(cell.get('value', '')) for cell in row['cells']]
                    else:
                        # Use dict values in order
                        processed_row = [str(v) for v in row.values()]
                    processed_rows.append(processed_row)
                else:
                    logger.warning(f"Unexpected row type at index {i}: {type(row)}")
                    processed_rows.append([str(row)])
            
            rows = processed_rows
            
            # Ensure consistent row lengths
            if rows:
                max_cols = max(len(row) for row in rows)
                min_cols = min(len(row) for row in rows)
                
                if max_cols != min_cols:
                    logger.warning(f"Inconsistent row lengths: min={min_cols}, max={max_cols}")
                    # Pad shorter rows
                    for row in rows:
                        while len(row) < max_cols:
                            row.append('')
                
                # Adjust headers to match data columns
                if len(headers) != max_cols:
                    logger.info(f"Adjusting headers: expected {max_cols}, got {len(headers)}")
                    if len(headers) < max_cols:
                        headers.extend([f"Column_{i+1}" for i in range(len(headers), max_cols)])
                    else:
                        headers = headers[:max_cols]
            
            # Create DataFrame from corrected data
            try:
                df_corrected = pd.DataFrame(rows, columns=headers)
                logger.info(f"Created DataFrame with shape: {df_corrected.shape}")
            except Exception as e:
                logger.error(f"DataFrame creation failed: {e}")
                # Fallback: create DataFrame without column names
                df_corrected = pd.DataFrame(rows)
                logger.info(f"Created DataFrame without headers, shape: {df_corrected.shape}")
            
            # Save as CSV
            csv_output_path = os.path.join(output_dir, f"{base_name}.csv")
            df_corrected.to_csv(csv_output_path, index=False, encoding='utf-8')
            
            
            logger.info(f"Corrected table saved to: {csv_output_path}")
            logger.info(f"Final table shape: {df_corrected.shape}")
            
            return csv_output_path
            
        except Exception as e:
            logger.error(f"Error saving corrected table: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    
    def validate_single_table(self, table, pdf_image) -> Dict:
        """
        Complete validation workflow for a single table
        
        Args:
            table: Extracted table data
            pdf_file: Path to PDF file (optional)
            
        Returns:
            Validation result dictionary
        """
        # Set default output directory
        
        
        # Run validation
        validation_result = self.validate_table_with_llm(
            table=table,
            pdf_image=pdf_image
        )
        
       
        # Save corrected table if available
        corrected_path = self.save_corrected_table(validation_result, output_dir, base_name)
        
        # Add output paths to result
        validation_result['output_files'] = {
            'validation_report': os.path.join(output_dir, f"{base_name}_validation_report.json"),
            'validation_summary': os.path.join(output_dir, f"{base_name}_validation_summary.txt"),
            'corrected_table': corrected_path
        }
        
        logger.info(f"Validation completed for: {base_name}")
        return validation_result


