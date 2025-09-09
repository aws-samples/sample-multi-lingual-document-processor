"""
File Processor class for handling file operations.
"""

import os
import logging
from pathlib import Path
import glob
from PyPDF2 import PdfReader, PdfWriter
import json
from src.utils.postprocessing import remove_tags_simple
from pdf2image import convert_from_bytes
from io import BytesIO
import pypdfium2 as pdfium
import tabulate
import re
class FileProcessor:
    """
    Handles file operations like splitting PDFs and combining text files.
    """
    
    def __init__(self):
        """
        Initialize the file processor.
        """
        self.logger = logging.getLogger()

    def split_pdf_from_s3(self, file_key):
        """
        Split a PDF document from S3 into individual pages.

        Args:
            file_key (str): Path/key of the PDF file in the bucket
        """
        try:
            # Create PDF reader object
            pdf = PdfReader(file_key)
            pdf_image = pdfium.PdfDocument(file_key)

            output_pdfs = []
            output_images = []
            for page in range(len(pdf.pages)):
                pdf_writer = PdfWriter()
                current_page = pdf.pages[page]
                pdf_writer.add_page(page=current_page)
                output_pdfs.append(current_page)

                pdf_page = pdf_image[page]
                page_image = pdf_page.render(scale=4).to_pil()
                output_images.append(page_image)

            return output_pdfs, output_images

        except Exception as e:
            raise Exception(f"Error processing PDF from S3: {str(e)}")

    def final_report(self, table_content, image_content, text_content):
        """
        Generate the final report by combining text, image, and table content from dictionaries.
        
        Args:
            table_content: Dictionary containing table extraction results
            image_content: Dictionary containing image narration results
            text_content: Dictionary containing text extraction results

        Returns:
            dict: Report dictionary containing combined content and token usage information
        """
        # final_dir = "/tmp/final_output"
        # Path(final_dir).mkdir(exist_ok=True)
        
        # Calculate total tokens
        total_input_tokens = (
            (image_content.get('input_tokens', 0) if image_content else 0) +
            (table_content.get('input_tokens', 0) if table_content else 0)
        )
        
        total_output_tokens = (
            (image_content.get('output_tokens', 0) if image_content else 0) +
            (table_content.get('output_tokens', 0) if table_content else 0)
        )
        
        total_tokens = total_input_tokens + total_output_tokens
        
        # Collect all the content to be included in the text representation
        report_text_content = []
        
        # Add image content if available
        if image_content and image_content.get('narration'):
            report_text_content.extend([
                "<image>",
                image_content['narration'],
                "</image>"
            ])
            self.logger.info("Added image narration to final report")
        else:
            self.logger.info("No image narration to add")
        
        # Add text content if available
        if text_content['narration'] and isinstance(text_content['narration'], str) and text_content['narration'].strip():
            report_text_content.extend([
                "<text>",
                text_content['narration'],
                "</text>"
            ])
            self.logger.info("Added text extraction to final report")
        else:
            self.logger.info("No text content to add")

        # Add table content if available
        
        # if table_content and table_content.get('narration'):
        #     report_text_content.extend([
        #         "<table>",
        #         table_content['narration'],
        #         "</table>"
        #     ])
        #     self.logger.info("Added table narration to final report")
        # else:
        #     self.logger.info("No table narration to add")
        
        # OPTION 1
        # if table_content and table_content.get('content') and isinstance(table_content['content'], list):
        #     # First add all tables
        #     for i, table in enumerate(table_content['content']):
        #         if table and isinstance(table, dict) and table.get('data') is not None:
                    
        #             # table_formatted = tabulate(table['data'], headers='keys', tablefmt='grid', showindex=False)
        #             try:
        #                 report_text_content.extend([
        #                     f"<table {i+1}>",
        #                     str(table['data']),
        #                     f"</table {i+1}>"
        #                 ])
        #                 self.logger.info(f"Added table {i+1}")
        #             except Exception as e:
        #                 self.logger.warning(f"Could not format table {i+1}: {str(e)}")
            
        #     # Then add the narration once at the end
        #     if table_content.get('narration'):
        #         report_text_content.extend([
        #             "<table_narration>",
        #             table_content['narration'],
        #             "</table_narration>"
        #         ])
        #         self.logger.info("Added table narration to final report")
        # else:
        #     self.logger.info("No tables to add")

        # OPTION 2
        if table_content and table_content.get('narration'):
            for table in table_content['content']:
                report_text_content.extend([
                    "<table>",
                    table_content['narration'],
                    "</table>"
                ])
                self.logger.info("Added table narration to final report")
        else:
            self.logger.info("No table narration to add")
        # Combine content into a string
        report_text = '\n'.join(filter(None, report_text_content)) if report_text_content else ""
        
        # Create the output file (for reference)
        # output_file = os.path.join(final_dir, "final_report.txt")
        # try:
        #     with open(output_file, 'w', encoding='utf-8') as f:
        #         f.write(report_text)
        #     self.logger.info(f"Final report text saved to: {output_file}")
        # except Exception as e:
        #     self.logger.error(f"Error writing final report to file: {str(e)}")
        
        # Create comprehensive report dictionary
        report_dict = {
            'report_text': report_text,
            'table_content': table_content,
            'image_content': image_content,
            'text_content': text_content,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'image': image_content.get('image') if image_content else None
        }
        
        self.logger.info(f"Final report generated successfully. Total tokens: {total_tokens}")
        return report_dict


    
    def combine_text_files(self, image_content, table_content, text_content, localization_content):
        """
        Combine multiple text files into a single output file.
        
        Args:
            localization_content (str): Content from the localization function
        """
        
        pattern = r'<table>page\d+_page1\.\d+</table>'
        
        
        # Count matches before removal
        matches = re.findall(pattern, localization_content['result'])
        print(f"Found {len(matches)} table tags to remove")
        
        # Remove the tags
        cleaned_content = re.sub(pattern, '', localization_content['result'])
        
        
        self.logger.info(f"All files have been combined")
        results = {}
        results['cleaned_content'] = cleaned_content
        results['image_content'] = image_content
        results['table_content'] = table_content
        results['text_content'] = text_content
        return results
