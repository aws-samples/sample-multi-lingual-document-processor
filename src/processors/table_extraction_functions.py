#!/usr/bin/env python3
"""
Table Extraction Functions using pdfplumber
Modular functions for extracting tables from PDF files that can be called from other scripts
"""

import os
import json
import pandas as pd
import pdfplumber
import logging
from datetime import datetime
from tabulate import tabulate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_table_data(table):
    """Clean and structure table data"""
    if not table or len(table) == 0:
        return None
        
    # Remove empty rows and columns
    cleaned_table = []
    for row in table:
        if row and any(cell and str(cell).strip() for cell in row):
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
    
    return cleaned_table if cleaned_table else None


def format_table_display(table, pdf_name, page_num, table_num):
    """Format table for display with proper structure"""
    if not table or len(table) == 0:
        return "Empty table"
    
    try:
        # Create DataFrame with proper headers
        if len(table) > 1:
            headers = table[0] if table[0] else [f"Column_{i+1}" for i in range(len(table[0]))]
            data = table[1:]
        else:
            headers = [f"Column_{i+1}" for i in range(len(table[0]))]
            data = table
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Format as table string
        table_str = f"\n{'='*100}\n"
        table_str += f"PDF: {pdf_name} | TABLE {table_num} - PAGE {page_num}\n"
        table_str += f"{'='*100}\n"
        table_str += tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        table_str += f"\n{'='*100}\n"
        table_str += f"Dimensions: {len(data)} rows × {len(headers)} columns\n"
        table_str += f"{'='*100}\n"
        
        return table_str
        
    except Exception as e:
        logger.error(f"Error formatting table: {str(e)}")
        return f"Error formatting table: {str(e)}"


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")


def extract_tables_from_pdf(document, csv_filename,output_dir="/tmp/table_output", save_files=True, display_tables=False):
    """
    Extract tables from a PDF document using pdfplumber
    
    Args:
        document (str): Path to the PDF file
        output_dir (str, optional): Directory to save output files. If None, creates default directory
        save_files (bool): Whether to save extracted tables to files
        display_tables (bool): Whether to print tables to console
    
    Returns:
        dict: Dictionary containing extraction results and metadata
    """
    # Validate input
    if not os.path.exists(document):
        logger.error(f"PDF file not found: {document}")
        return None
    
    pdf_name = os.path.splitext(os.path.basename(document))[0]
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join("table_extraction_results", f"{pdf_name}_results")
    
    if save_files:
        create_output_directory(output_dir)
    
    logger.info(f"Starting table extraction from {pdf_name} with pdfplumber...")
    results = []
    all_tables_text = ""
    
    try:
        with pdfplumber.open(document) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages from {pdf_name}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                logger.info(f"Processing page {page_num}/{total_pages} from {pdf_name}...")
                
                # Extract tables from the page
                tables = page.extract_tables()
                
                if tables:
                    for table_num, table in enumerate(tables, 1):
                        logger.info(f"Found table {table_num} on page {page_num} in {pdf_name}")
                        
                        # Clean table data
                        cleaned_table = clean_table_data(table)
                        
                        if cleaned_table and len(cleaned_table) > 0:
                            # Create structured DataFrame
                            if len(cleaned_table) > 1:
                                headers = cleaned_table[0]
                                data = cleaned_table[1:]
                            else:
                                headers = [f"Column_{i+1}" for i in range(len(cleaned_table[0]))]
                                data = cleaned_table
                            
                            df = pd.DataFrame(data, columns=headers)
                            
                            # Prepare file paths
                            csv_filename = None
                            json_filename = None
                            save_files=True
                            if save_files:
                                # Save as CSV
                                
                                csv_filename = csv_filename
                                df.to_csv(csv_filename, index=False, encoding='utf-8')
                                
                                
                           
                else:
                    logger.info(f"No tables found on page {page_num} ")
                    
    except Exception as e:
        logger.error(f"Error with pdfplumber extraction from : {str(e)}")
        return None
    
   
    return "done"





def extract_and_summarize(document, output_dir=None, save_files=True, display_tables=False, display_summary=True):
    """
    Complete table extraction and summary generation in one function
    
    Args:
        document (str): Path to the PDF file
        output_dir (str, optional): Directory to save output files
        save_files (bool): Whether to save extracted tables to files
        display_tables (bool): Whether to print tables to console
        display_summary (bool): Whether to print summary to console
    
    Returns:
        dict: Complete extraction results with summary
    """
    # Extract tables
    results = extract_tables_from_pdf(document, output_dir, save_files, display_tables)
    
    if not results:
        logger.error("Table extraction failed")
        return None
    
    

# Main function that can be called from other scripts
def extract_tables(document,csv_filename,output_dir="/tmp/table_output", save_files=True ,**kwargs):
    """
    Simple function to extract tables from a document
    
    Args:
        document (str): Path to the PDF document
        **kwargs: Additional arguments:
            - output_dir (str): Directory to save output files
            - save_files (bool): Whether to save extracted tables to files (default: True)
            - display_tables (bool): Whether to print tables to console (default: False)
            - display_summary (bool): Whether to print summary to console (default: True)
    
    Returns:
        dict: Extraction results
    """
    pdf_name = os.path.splitext(os.path.basename(document))[0]

    try:
        with pdfplumber.open(document) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages from {pdf_name}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                logger.info(f"Processing page {page_num}/{total_pages} from {pdf_name}...")
                
                # Extract tables from the page
                tables = page.extract_tables()
                
                if tables:
                    for table_num, table in enumerate(tables, 1):
                        logger.info(f"Found table {table_num} on page {page_num} in {pdf_name}")
                        
                        # Clean table data
                        cleaned_table = clean_table_data(table)
                        
                        if cleaned_table and len(cleaned_table) > 0:
                            # Create structured DataFrame
                            if len(cleaned_table) > 1:
                                headers = cleaned_table[0]
                                data = cleaned_table[1:]
                            else:
                                headers = [f"Column_{i+1}" for i in range(len(cleaned_table[0]))]
                                data = cleaned_table
                            
                            df = pd.DataFrame(data, columns=headers)
                            print(df)
                            # Prepare file paths
                            
                            save_files=True
                            if save_files:
                                # Save as CSVif table_num == 1:
                                    # First table uses the original filename
                                if table_num == 1:
                                    final_csv_filename = csv_filename
                                else:
                                    # Additional tables get a suffix to avoid overwriting
                                    base_name = os.path.splitext(csv_filename)[0]
                                    extension = os.path.splitext(csv_filename)[1]
                                    # Change from .1.csv to .2.csv, .3.csv etc for additional tables
                                    final_csv_filename = f"{base_name[:-1]}{table_num}{extension}"
                                
                                df.to_csv(f"{output_dir}/{final_csv_filename}", index=False, encoding='utf-8')
                                print(f"table files saved with PDF plumber: {final_csv_filename}")
                                
                                #csv_filename = csv_filename
                                # df.to_csv(f"{output_dir}/{csv_filename}", index=False, encoding='utf-8')
                                # print("table files save with PDF plumber")

    except Exception as e:
        logger.error(f"Error with pdfplumber extraction from : {str(e)}")
        return None
    

