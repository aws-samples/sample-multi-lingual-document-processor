#!/usr/bin/env python3
"""
Simple Text-Based Claude + Tabula Extractor
Works directly with PDF text input instead of images
"""

import boto3
import json
import sys
import os
import PyPDF2
import tabula
import pandas as pd
from datetime import datetime
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_path
from src.processors.table_extraction_functions import extract_tables
from src.config.config import tabula_model_ID, AWS_REGION
from src.config.prompts import histogram, percentage_stacked_bar_chart, bar_chart, table_definition , graph_definitions , claude_tabula_prompt
from src.processors.table_claude_ocr import PDFTableExtractor
from botocore.exceptions import ClientError
import random
import time

extractor = PDFTableExtractor(
    model_id=tabula_model_ID,
    region_name=AWS_REGION,
    output_dir="/tmp/"
)

def convert_pdf_page_to_image(pdf_path: str, page_num: int = 1):
    """Convert PDF page to image using pdf2image"""
    try:
        # Convert specific page to image
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=200)
        
        if images:
            # Get the first (and only) image
            image = images[0]
            
            # Convert PIL image to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_data = buffer.getvalue()
            
            # Convert to base64 for Claude
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return img_base64
        else:
            print(f"No image generated for page {page_num}")
            return ""
    except Exception as e:
        print(f"Error converting PDF page {page_num} to image: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str, page_num: int = 1):
    """Extract text from PDF page using PyPDF2"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if 1 <= page_num <= len(pdf_reader.pages):
                page = pdf_reader.pages[page_num - 1]  # Convert to 0-indexed
                text_content = page.extract_text()
                return text_content
            else:
                print(f"Page {page_num} out of range (1-{len(pdf_reader.pages)})")
                return ""
    except Exception as e:
        print(f"Error extracting text from page {page_num}: {e}")
        return ""

def claude_analyze_text(image_base64: str, text_content, page_number: int):
    """Analyze PDF image with Claude to identify tables"""
    
    bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
    model_id = tabula_model_ID
    
    prompt = claude_tabula_prompt
    
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "temperature": 0,
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
    
    attempt = 0
    max_attempts = 10
    model_tokens = {}
    model_tokens['input_tokens'] = 0
    model_tokens['output_tokens'] = 0
    while attempt < max_attempts:
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
            output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
            
            model_tokens['input_tokens'] = input_tokens
            model_tokens['output_tokens'] = output_tokens
                    
            # If successful, break out of the loop

            break
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException' and attempt < max_attempts - 1:
                attempt += 1
                max_backoff = 1 * (2 ** attempt)
                jitter = random.uniform(0, 0.1 * max_backoff)  # 10% jitter
                backoff_time = max_backoff + jitter
                print(f"  Throttling error occurred. Retrying in {backoff_time:.2f} seconds (attempt {attempt}/{max_attempts})...")
                # nosemgrep: arbitrary-sleep
                time.sleep(backoff_time)
            else:
                raise
    
    response_body = json.loads(response['body'].read())
    content = response_body['content'][0]['text']
    
    # Extract JSON from response
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    if json_start != -1 and json_end != -1:
        json_content = content[json_start:json_end]
        return json.loads(json_content), model_tokens
    else:
        return {"error": "Could not parse JSON from response", "raw_response": content} , model_tokens


def tabula_extract_tables(pdf_path: str, claude_analysis: dict):
    """Extract tables using tabula based on Claude text analysis"""
    
    page_num = claude_analysis.get("page_number", 1)
    tables_info = claude_analysis.get("tables", [])
    tables_info2 = claude_analysis.get('tables_found')
    extracted_tables = []
    
    for table_info in tables_info:
        table_id = table_info.get("table_id", 1)
        method = table_info.get("method")  # Default to stream for text-based
        
        print(f"Extracting Table {table_id} using {method} method...")
        try:
            # For text-based analysis, use stream method with multiple_tables
            if(method == "stream"):
                tables = tabula.read_pdf(
                    pdf_path, 
                    pages=page_num, 
                    stream=True,  # Stream works better for text-based detection
                    guess=True,
                    multiple_tables=True
                )
            else:
                tables = tabula.read_pdf(
                    pdf_path, 
                    pages=page_num, 
                    stream=False,  # Stream works better for text-based detection
                    guess=True,
                    multiple_tables=True
                )
            
            if tables and len(tables) > 0:
                # Select appropriate table based on analysis
                if table_id <= len(tables):
                    table = tables[table_id - 1]
                else:
                    table = tables[0]  # Fallback to first table
                
                extracted_tables.append({
                    "table_id": table_id,
                    "title": table_info.get("title", f"Table {table_id}"),
                    "method": method,
                    "shape": table.shape,
                    "confidence": table_info.get("confidence", 0.5),
                    "data": table,
                    "claude_analysis": table_info,
                    "extraction_timestamp": datetime.now().isoformat()
                })
                
                print(f"✅ Table {table_id}: {table.shape[0]}×{table.shape[1]}")
                
                
            else:
                #print(f"❌ No table extracted for Table {table_id}")
                print(f"falling back to PDF plumber")
                
                if ( tables_info2 == 1):
                    
                    base_filename = Path(pdf_path).stem
                    csv_filename = f"{base_filename}_page1.{1}.csv"
                    print(csv_filename)
                    extract_tables(pdf_path, csv_filename )
                else:
                   print(f"❌ No table extracted for Table {table_id}")
            
            if(len(extracted_tables) == 0 and tables_info2 == 1):
                print(f"falling back to PDF plumber")
                base_filename = Path(pdf_path).stem
                csv_filename = f"{base_filename}_page1.{1}.csv"
                print(csv_filename)
                extract_tables(pdf_path, csv_filename )
            else:
                print(f"❌ No table extracted for Table {table_id}")
                
                
                
        except Exception as e:
            print(f"❌ Error extracting Table {table_id}: {e}")
            print(f"falling back to PDF plumber")
            tables_info1 = claude_analysis.get('tables_found')
            print(tables_info1)
            if ( tables_info1 == 1):
                base_filename = Path(pdf_path).stem
                csv_filename = f"{base_filename}_page1.{1}.csv"
                print(csv_filename)
                extract_tables(pdf_path, csv_filename )
            else:
                 print(f"❌ No table extracted for Table {table_id} with PDF plumber")
    
    return extracted_tables

def save_extracted_tables(extracted_tables: list, base_filename,output_dir: str, pdf_name: str, page_num: int):
    """Save extracted tables as CSV files with metadata"""
    
    if not extracted_tables:
        print("No tables to save")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(pdf_name).stem
    saved_files = []
    i =0
    for table_info in extracted_tables:
        table_id = table_info["table_id"]
        table_data = table_info["data"]
        
        # Generate filenames
        
        csv_filename = f"{base_filename}_page1.{i+1}.csv"
        metadata_filename = f"{base_name}_page{page_num}_table{table_id}_metadata.json"
        
        csv_path = os.path.join("/tmp/table_output", csv_filename)
        
        try:
            # Save CSV file
            table_data.to_csv(csv_path, index=False, encoding='utf-8')
            
            i= i+1
            
            
        except Exception as e:
            print(f"❌ Error saving Table {table_id}: {e}")
    
    return saved_files





def process_pdf_tables(pdf_file, base_filename,page_num=1,output_dir="/tmp/table_output" ):
    """
    Main function to process PDF tables that can be called from text_processor.py
    
    Args:
        pdf_file (str): Path to the PDF file
        page_num (int): Page number to process (default: 1)
        output_dir (str): Output directory for extracted tables (default: "extracted_tables")
    
    Returns:
        dict: Processing results with extracted tables and metadata
    """
    
    print(f"🚀 Simple Text-Based Claude + Tabula Extractor")
    print(f"📄 Processing: {pdf_file} (Page {page_num})")
    print(f"📝 Method: Direct text analysis (no image conversion)")
    print(f"📁 Output Directory: {output_dir}")
    print("-" * 70)
    
    try:
        # Step 1: Convert PDF page to image
        print("Step 1: Converting PDF page to image...")
        print("Step 1: Extracting text from PDF...")
        text_content = extract_text_from_pdf(pdf_file, page_num)
        print(text_content)
        image_base64 = convert_pdf_page_to_image(pdf_file, page_num)
        
        if not image_base64:
            print("❌ No image content extracted from PDF")
            return {"error": "No image content extracted from PDF", "success": False}
        
        print(f"✅ Image converted: {len(image_base64)} characters (base64)")
        
        # Step 2: Claude image analysis
        print(f"\nStep 2: Analyzing image with Claude...")
        claude_result, model_tokens1 = claude_analyze_text(image_base64,text_content ,page_num)
        print(claude_result)
        
        # Track total tokens
        total_input_tokens = model_tokens1['input_tokens']
        total_output_tokens = model_tokens1['output_tokens']
        
        if "error" in claude_result:
            print(f"❌ Claude analysis failed: {claude_result['error']}")
            return {"error": f"Claude analysis failed: {claude_result['error']}", "success": False}
        
        tables_found = claude_result.get("tables_found", 0)
        print(f"✅ Claude found {tables_found} tables in text")
        
        for table in claude_result.get("tables", []):
            print(f"  - Table {table['table_id']}: {table['title']} (confidence: {table.get('confidence', 'N/A')})")
        
        if tables_found == 0:
            print("ℹ️ No tables detected in text. This might be because:")
            print("   - The PDF contains image-based tables")
            print("   - The text extraction didn't capture table structure")
            print("   - The document doesn't contain tabular data")
            return {
                "error": "No tables detected in text", 
                "success": False, 
                "tables_found": 0,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }
        
        # Step 3: Tabula extraction
        print(f"\nStep 3: Extracting tables with tabula...")
        
        extracted_tables = tabula_extract_tables(pdf_file, claude_result)
        score = extractor._table_evaluator(extracted_tables)
        print("ℹ️ Prediction Score for LLM as Judge when Running on the Tabula Table Extractor is:", score['score'])
        
        # Track tokens from table evaluator
        model_tokens_evaluator = score.get('model_tokens', {'input_tokens': 0, 'output_tokens': 0})
        total_input_tokens += model_tokens_evaluator.get('input_tokens', 0)
        total_output_tokens += model_tokens_evaluator.get('output_tokens', 0)
        
        if score['score'] < 5:
            extracted_tables, model_tokens_table = extractor.extract_tables(
                                                    pdf_file=pdf_file,
                                                    output_path="/tmp/consolidated_tables.md",
                                                    num_sections=6  # How many sections to split large tables into
                                                )
            # Add tokens from extractor.extract_tables
            total_input_tokens += model_tokens_table.get('input_tokens', 0)
            total_output_tokens += model_tokens_table.get('output_tokens', 0)
        else:
            # If we didn't run extractor.extract_tables, set model_tokens_table to zeros
            model_tokens_table = {'input_tokens': 0, 'output_tokens': 0}
        
        
        # Step 4: Save tables
        print(f"\nStep 4: Saving extracted tables...")
        saved_files = save_extracted_tables(extracted_tables,base_filename, output_dir, pdf_file, page_num)
        print(f"\n📊 Final Results:")
        print(f"Tables found by Claude: {tables_found}")
        print(f"Tables extracted by tabula: {len(extracted_tables)}")
        print(f"Tables saved: {len(saved_files)}")
        print(f"Success rate: {len(saved_files) / max(tables_found, 1) * 100:.1f}%")
        
        # Display extracted data preview
        for table in extracted_tables:
            print(f"\n📋 {table['title']} ({table['shape'][0]}×{table['shape'][1]}) - Confidence: {table['confidence']}")
            print(table['data'].head(3))
            print("...")
        
        print(f"\n📁 All files saved to: {output_dir}")
        print(f"✅ Text-based extraction completed!")
        
        # Print token information
        print(f"📊 Token Usage:")
        print(f"Total Input Tokens: {total_input_tokens}")
        print(f"Total Output Tokens: {total_output_tokens}")
        print(f"Total Tokens: {total_input_tokens + total_output_tokens}")
        
        return {
            "success": True,
            "tables_found": tables_found,
            "tables_extracted": len(extracted_tables),
            "tables_saved": len(saved_files),
            "success_rate": len(saved_files) / max(tables_found, 1) * 100,
            "claude_analysis": claude_result,
            "extracted_tables": extracted_tables,
            "saved_files": saved_files,
            "output_dir": output_dir, 
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Check if any tables were extracted despite the error
        extracted_files = []
        if os.path.exists(output_dir):
            extracted_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        
        if extracted_files:
            print(f"⚠️ Error occurred but {len(extracted_files)} table files were found")
            return {"success": True, "tables_extracted": len(extracted_files), "error": str(e)}
        else:
            return {"error": str(e), "success": False, "traceback": traceback.format_exc()}
