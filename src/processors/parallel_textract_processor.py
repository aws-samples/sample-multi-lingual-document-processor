"""
Parallel Textract Processor

This module provides functionality to process PDF documents using Amazon Textract
in parallel, with each page being processed by a separate process.
"""

import os
import time
import json
import logging
import boto3
import multiprocessing
from multiprocessing import Pool
from botocore.config import Config
import tempfile
from pathlib import Path
import pdf2image
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelTextractProcessor:
    """
    Process PDF documents using Amazon Textract in parallel.
    Each page is processed by a separate process.
    """
    
    def __init__(self, region_name: str = "us-east-1", max_workers: int = None):
        """
        Initialize the parallel Textract processor.
        
        Args:
            region_name: AWS region name
            max_workers: Maximum number of worker processes (defaults to CPU count)
        """
        self.region_name = region_name
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Configure boto3 with longer timeout and retry settings
        self.config = Config(
            read_timeout=300,
            connect_timeout=300,
            retries=dict(max_attempts=5)
        )
        
        logger.info(f"Initialized ParallelTextractProcessor with {self.max_workers} workers in region {self.region_name}")
    
    def _extract_page_from_pdf(self, pdf_path: str, page_num: int) -> Image.Image:
        """
        Extract a single page from a PDF as an image.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to extract (1-based)
            
        Returns:
            PIL Image object
        """
        try:
            images = pdf2image.convert_from_path(
                pdf_path, 
                first_page=page_num, 
                last_page=page_num,
                dpi=300
            )
            return images[0]
        except Exception as e:
            logger.error(f"Error extracting page {page_num} from PDF {pdf_path}: {str(e)}")
            raise
    
    def _process_page(self, args: Tuple[str, int, str, bool]) -> Dict[str, Any]:
        """
        Process a single page with Textract.
        This function is designed to be called by multiprocessing.Pool.
        
        Args:
            args: Tuple containing (pdf_path, page_num, output_dir, save_images)
            
        Returns:
            Dict containing the Textract results and metadata
        """
        pdf_path, page_num, output_dir, save_images = args
        
        start_time = time.time()
        logger.info(f"Processing page {page_num} from {os.path.basename(pdf_path)}")
        
        try:
            # Extract the page as an image
            image = self._extract_page_from_pdf(pdf_path, page_num)
            
            # Save the image if requested
            image_path = None
            if save_images:
                os.makedirs(output_dir, exist_ok=True)
                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                logger.info(f"Saved page {page_num} image to {image_path}")
            
            # Convert image to bytes for Textract
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Initialize Textract client
            textract_client = boto3.client(
                'textract',
                region_name=self.region_name,
                config=self.config
            )
            
            # Call Textract API
            response = textract_client.detect_document_text(
                Document={'Bytes': img_bytes}
            )
            
            # Save the Textract results
            result_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}_textract.json"
            result_path = os.path.join(output_dir, result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(response, f, indent=2)
            
            # Extract plain text from Textract response
            extracted_text = self._extract_text_from_response(response)
            text_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}_text.txt"
            text_path = os.path.join(output_dir, text_filename)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            processing_time = time.time() - start_time
            logger.info(f"Completed processing page {page_num} in {processing_time:.2f} seconds")
            
            return {
                'page_num': page_num,
                'success': True,
                'textract_result_path': result_path,
                'text_path': text_path,
                'image_path': image_path,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return {
                'page_num': page_num,
                'success': False,
                'error': str(e)
            }
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract plain text from Textract response.
        
        Args:
            response: Textract API response
            
        Returns:
            Extracted text as a string
        """
        text = ""
        for item in response.get("Blocks", []):
            if item.get("BlockType") == "LINE":
                text += item.get("Text", "") + "\n"
        return text
    
    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages
        """
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return len(pdf_reader.pages)
        except Exception as e:
            logger.error(f"Error getting page count for {pdf_path}: {str(e)}")
            raise
    
    def process_pdf(self, 
                   pdf_path: str, 
                   output_dir: str = None, 
                   save_images: bool = False,
                   specific_pages: List[int] = None) -> Dict[str, Any]:
        """
        Process a PDF document with Textract in parallel.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save results (defaults to a temp directory)
            save_images: Whether to save page images
            specific_pages: List of specific page numbers to process (1-based)
                           If None, all pages are processed
            
        Returns:
            Dict containing processing results and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), 
                                     f"textract_results_{os.path.splitext(os.path.basename(pdf_path))[0]}")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Results will be saved to {output_dir}")
        
        # Get page count
        total_pages = self._get_pdf_page_count(pdf_path)
        logger.info(f"PDF has {total_pages} pages")
        
        # Determine which pages to process
        if specific_pages:
            pages_to_process = [p for p in specific_pages if 1 <= p <= total_pages]
            logger.info(f"Processing {len(pages_to_process)} specific pages: {pages_to_process}")
        else:
            pages_to_process = list(range(1, total_pages + 1))
            logger.info(f"Processing all {total_pages} pages")
        
        if not pages_to_process:
            logger.warning("No valid pages to process")
            return {
                'success': False,
                'error': 'No valid pages to process',
                'pdf_path': pdf_path,
                'output_dir': output_dir
            }
        
        # Prepare arguments for parallel processing
        process_args = [(pdf_path, page_num, output_dir, save_images) 
                       for page_num in pages_to_process]
        
        # Process pages in parallel
        start_time = time.time()
        with Pool(processes=min(self.max_workers, len(pages_to_process))) as pool:
            results = pool.map(self._process_page, process_args)
        
        # Combine results
        successful_pages = [r for r in results if r.get('success', False)]
        failed_pages = [r for r in results if not r.get('success', False)]
        
        total_time = time.time() - start_time
        logger.info(f"Completed processing {len(successful_pages)} pages in {total_time:.2f} seconds")
        
        if failed_pages:
            logger.warning(f"Failed to process {len(failed_pages)} pages")
        
        # Create a combined text file with all extracted text
        combined_text_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_combined_text.txt")
        
        with open(combined_text_path, 'w', encoding='utf-8') as combined_file:
            # Sort successful results by page number
            for result in sorted(successful_pages, key=lambda x: x['page_num']):
                page_num = result['page_num']
                text_path = result['text_path']
                
                combined_file.write(f"\n\n--- PAGE {page_num} ---\n\n")
                
                try:
                    with open(text_path, 'r', encoding='utf-8') as page_file:
                        combined_file.write(page_file.read())
                except Exception as e:
                    logger.error(f"Error reading text file {text_path}: {str(e)}")
                    combined_file.write(f"[Error reading text for page {page_num}]")
        
        logger.info(f"Combined text saved to {combined_text_path}")
        
        return {
            'success': True,
            'pdf_path': pdf_path,
            'output_dir': output_dir,
            'total_pages': total_pages,
            'processed_pages': len(successful_pages),
            'failed_pages': len(failed_pages),
            'processing_time': total_time,
            'combined_text_path': combined_text_path,
            'page_results': results
        }
    
    def process_pdf_with_analyze_document(self, 
                                         pdf_path: str, 
                                         output_dir: str = None,
                                         save_images: bool = False,
                                         specific_pages: List[int] = None,
                                         features: List[str] = None) -> Dict[str, Any]:
        """
        Process a PDF document with Textract's AnalyzeDocument API in parallel.
        This method supports table extraction and other advanced features.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save results (defaults to a temp directory)
            save_images: Whether to save page images
            specific_pages: List of specific page numbers to process (1-based)
                           If None, all pages are processed
            features: List of Textract features to enable (e.g., ['TABLES', 'FORMS'])
                     Defaults to ['TABLES']
            
        Returns:
            Dict containing processing results and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), 
                                     f"textract_results_{os.path.splitext(os.path.basename(pdf_path))[0]}")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Results will be saved to {output_dir}")
        
        # Set default features if not provided
        if features is None:
            features = ['TABLES']
        
        # Get page count
        total_pages = self._get_pdf_page_count(pdf_path)
        logger.info(f"PDF has {total_pages} pages")
        
        # Determine which pages to process
        if specific_pages:
            pages_to_process = [p for p in specific_pages if 1 <= p <= total_pages]
            logger.info(f"Processing {len(pages_to_process)} specific pages: {pages_to_process}")
        else:
            pages_to_process = list(range(1, total_pages + 1))
            logger.info(f"Processing all {total_pages} pages")
        
        if not pages_to_process:
            logger.warning("No valid pages to process")
            return {
                'success': False,
                'error': 'No valid pages to process',
                'pdf_path': pdf_path,
                'output_dir': output_dir
            }
        
        # Process each page in parallel
        start_time = time.time()
        results = []
        
        with Pool(processes=min(self.max_workers, len(pages_to_process))) as pool:
            process_args = []
            
            for page_num in pages_to_process:
                process_args.append((
                    pdf_path,
                    page_num,
                    output_dir,
                    save_images,
                    features
                ))
            
            results = pool.map(self._process_page_with_analyze_document, process_args)
        
        # Combine results
        successful_pages = [r for r in results if r.get('success', False)]
        failed_pages = [r for r in results if not r.get('success', False)]
        
        total_time = time.time() - start_time
        logger.info(f"Completed processing {len(successful_pages)} pages in {total_time:.2f} seconds")
        
        if failed_pages:
            logger.warning(f"Failed to process {len(failed_pages)} pages")
        
        return {
            'success': True,
            'pdf_path': pdf_path,
            'output_dir': output_dir,
            'total_pages': total_pages,
            'processed_pages': len(successful_pages),
            'failed_pages': len(failed_pages),
            'processing_time': total_time,
            'page_results': results
        }
    
    def _process_page_with_analyze_document(self, args: Tuple) -> Dict[str, Any]:
        """
        Process a single page with Textract's AnalyzeDocument API.
        This function is designed to be called by multiprocessing.Pool.
        
        Args:
            args: Tuple containing (pdf_path, page_num, output_dir, save_images, features)
            
        Returns:
            Dict containing the Textract results and metadata
        """
        pdf_path, page_num, output_dir, save_images, features = args
        
        start_time = time.time()
        logger.info(f"Processing page {page_num} from {os.path.basename(pdf_path)} with AnalyzeDocument")
        
        try:
            # Extract the page as an image
            image = self._extract_page_from_pdf(pdf_path, page_num)
            
            # Save the image if requested
            image_path = None
            if save_images:
                os.makedirs(output_dir, exist_ok=True)
                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                logger.info(f"Saved page {page_num} image to {image_path}")
            
            # Convert image to bytes for Textract
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Initialize Textract client
            textract_client = boto3.client(
                'textract',
                region_name=self.region_name,
                config=self.config
            )
            
            # Call Textract AnalyzeDocument API
            response = textract_client.analyze_document(
                Document={'Bytes': img_bytes},
                FeatureTypes=features
            )
            
            # Save the Textract results
            result_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}_analyze.json"
            result_path = os.path.join(output_dir, result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(response, f, indent=2)
            
            # Extract tables if available
            tables = []
            if 'TABLES' in features:
                tables = self._extract_tables_from_response(response)
                
                # Save tables as CSV files
                for i, table in enumerate(tables):
                    table_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}_table_{i+1}.csv"
                    table_path = os.path.join(output_dir, table_filename)
                    
                    with open(table_path, 'w', newline='', encoding='utf-8') as f:
                        for row in table:
                            f.write(','.join([f'"{cell}"' if cell else '""' for cell in row]) + '\n')
            
            processing_time = time.time() - start_time
            logger.info(f"Completed processing page {page_num} in {processing_time:.2f} seconds")
            
            return {
                'page_num': page_num,
                'success': True,
                'textract_result_path': result_path,
                'image_path': image_path,
                'tables_found': len(tables),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num} with AnalyzeDocument: {str(e)}")
            return {
                'page_num': page_num,
                'success': False,
                'error': str(e)
            }
    
    def _extract_tables_from_response(self, response: Dict[str, Any]) -> List[List[List[str]]]:
        """
        Extract tables from Textract AnalyzeDocument response.
        
        Args:
            response: Textract API response
            
        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cell values
        """
        # Create a mapping of block IDs to blocks
        blocks = {block['Id']: block for block in response.get('Blocks', [])}
        
        tables = []
        
        # Find all table blocks
        for block_id, block in blocks.items():
            if block['BlockType'] == 'TABLE':
                table_data = []
                
                # Get all cells in this table
                for relationship in block.get('Relationships', []):
                    if relationship['Type'] == 'CHILD':
                        # Sort cells by row and column
                        cells = [blocks[cell_id] for cell_id in relationship['Ids'] 
                                if blocks[cell_id]['BlockType'] == 'CELL']
                        
                        # Group cells by row
                        max_row = max(cell['RowIndex'] for cell in cells)
                        max_col = max(cell['ColumnIndex'] for cell in cells)
                        
                        # Initialize table with empty cells
                        table = [[''] * max_col for _ in range(max_row)]
                        
                        # Fill in cell values
                        for cell in cells:
                            row_idx = cell['RowIndex'] - 1  # Convert to 0-based index
                            col_idx = cell['ColumnIndex'] - 1  # Convert to 0-based index
                            
                            # Get cell text
                            cell_text = ''
                            for rel in cell.get('Relationships', []):
                                if rel['Type'] == 'CHILD':
                                    for word_id in rel['Ids']:
                                        if blocks[word_id]['BlockType'] in ['WORD', 'LINE']:
                                            cell_text += blocks[word_id].get('Text', '') + ' '
                            
                            table[row_idx][col_idx] = cell_text.strip()
                        
                        table_data = table
                
                if table_data:
                    tables.append(table_data)
        
        return tables


def process_pdf_with_textract(pdf_path: str, 
                             output_dir: str = None, 
                             region_name: str = "us-east-1",
                             max_workers: int = None,
                             save_images: bool = False,
                             specific_pages: List[int] = None,
                             extract_tables: bool = False) -> Dict[str, Any]:
    """
    Convenience function to process a PDF with Textract in parallel.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save results
        region_name: AWS region name
        max_workers: Maximum number of worker processes
        save_images: Whether to save page images
        specific_pages: List of specific page numbers to process (1-based)
        extract_tables: Whether to extract tables using AnalyzeDocument
        
    Returns:
        Dict containing processing results and metadata
    """
    processor = ParallelTextractProcessor(region_name=region_name, max_workers=max_workers)
    
    if extract_tables:
        return processor.process_pdf_with_analyze_document(
            pdf_path=pdf_path,
            output_dir=output_dir,
            save_images=save_images,
            specific_pages=specific_pages,
            features=['TABLES']
        )
    else:
        return processor.process_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            save_images=save_images,
            specific_pages=specific_pages
        )


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Process PDF documents with Amazon Textract in parallel")
    parser.add_argument("--pdf_path", default="data/RTR46a03 (2).pdf", help="Path to the PDF file")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--region", default="us-east-1", help="AWS region name")
    parser.add_argument("--workers", default=4, type=int, help="Maximum number of worker processes")
    parser.add_argument("--save-images", action="store_true", help="Save page images")
    parser.add_argument("--pages", type=int, nargs="+", help="Specific page numbers to process")
    parser.add_argument("--extract-tables", action="store_true", help="Extract tables using AnalyzeDocument")
    # parser.add_argument("--s3-bucket", help="S3 bucket name (if processing from S3)")
    # parser.add_argument("--s3-key", help="S3 key (if processing from S3)")
    # parser.add_argument("--demo", action="store_true", help="Run a demonstration with detailed output")
    
    args = parser.parse_args()
    
    # Standard mode with minimal output
    start_time = time.time()
    
    result = process_pdf_with_textract(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        region_name=args.region,
        max_workers=args.workers,
        save_images=args.save_images,
        specific_pages=args.pages,
        extract_tables=args.extract_tables
    )
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Processed {result['processed_pages']} of {result['total_pages']} pages successfully")
    
    if result.get('failed_pages', 0) > 0:
        print(f"Failed pages: {result['failed_pages']}")
        
    print(f"Results saved to: {result['output_dir']}")
    
    if result.get('combined_text_path'):
        print(f"Combined text saved to: {result['combined_text_path']}")
