

import base64
from io import BytesIO, StringIO
from pathlib import Path
import pandas as pd
import os
import re
from PIL import Image
import os
import re
from src.config.config import OUTPUT_DIR
import boto3
import json
from tabulate import tabulate
import copy
import io

# Convert PIL image to base64
def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')  # or 'JPEG' depending on your needs
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
    return base64_encoded


def save_processed_content(final_report, localization_content, page_num, logger, original_filename, output_dir=None):
    """
    Save processed content (text, tables, images) from PDF page processing.

    Args:
        final_report (dict): The final report containing all processed content
        localization_content (dict): The localized content
        page_num (int): Index of the page being processed (0-based)
        logger: Logger object for logging messages
        original_filename (str): Original filename of the PDF being processed
        output_dir (Path, optional): Output directory path. If None, uses OUTPUT_DIR

    Returns:
        dict: Dictionary with paths of saved files
    """
    saved_files = {}

    try:
        # Create main output directory
        output_dir = Path(output_dir or OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)


        # Save processed text report with page number
        output_processed_text_path = output_dir / f"{Path(original_filename).stem}_page{page_num}_processed.txt"
        output_processed_text_path.write_text(localization_content['result'], encoding='utf-8')
        logger.info(f"Saved processed text report for page {page_num} to: {output_processed_text_path}")
        saved_files['processed_text'] = str(output_processed_text_path)

        # Create tables directory and save tables
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        saved_files['tables'] = []

        # Save table content if available
        if 'table_content' in final_report and final_report['table_content'] and 'content' in final_report['table_content']:
            table_content = final_report['table_content']

            # Save all tables to CSV files with page number in filename
            for k, table in enumerate(table_content['content']):
                try:
                    if table and isinstance(table, dict) and 'data' in table and table['data'] is not None:
                        table_df = pd.DataFrame(table['data'])
                        table_csv_path = tables_dir / f"page{page_num}_table_{k+1}.csv"
                        table_df.to_csv(table_csv_path, index=False)
                        logger.info(f"Saved table {k+1} from page {page_num} to: {table_csv_path}")
                        saved_files['tables'].append(str(table_csv_path))
                except Exception as e:
                    logger.error(f"Error saving table {k+1} from page {page_num}: {str(e)}")

            # Save table narrations to a text file with page number
            if 'narration' in table_content and table_content['narration']:
                table_narration_path = tables_dir / f"page{page_num}_table_narrations.txt"
                try:
                    with open(table_narration_path, 'w', encoding='utf-8') as f:
                        f.write(table_content['narration'])
                    logger.info(f"Saved table narrations for page {page_num} to: {table_narration_path}")
                    saved_files['table_narration'] = str(table_narration_path)
                except Exception as e:
                    logger.error(f"Error saving table narrations for page {page_num}: {str(e)}")
        else:
            logger.info(f"No table content to save for page {page_num}")

        # Create images directory and save images
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        saved_files['images'] = []

        # Save image if available
        if 'image_content' in final_report and final_report['image_content']:
            image_content = final_report['image_content']

            # Check if image is directly in image_content
            if 'image' in final_report and final_report['image'] is not None:
                try:
                    image = final_report['image']
                    image_path = images_dir / f"page{page_num}_image.png"
                    image.save(image_path, format='PNG')
                    logger.info(f"Saved image from page {page_num} to: {image_path}")
                    saved_files['images'].append(str(image_path))
                except Exception as e:
                    logger.error(f"Error saving image from page {page_num}: {str(e)}")

            # Save image narration to a text file with page number
            if 'narration' in image_content and image_content['narration']:
                image_narration_path = images_dir / f"page{page_num}_image_narration.txt"
                try:
                    with open(image_narration_path, 'w', encoding='utf-8') as f:
                        f.write(image_content['narration'])
                    logger.info(f"Saved image narration for page {page_num} to: {image_narration_path}")
                    saved_files['image_narration'] = str(image_narration_path)
                except Exception as e:
                    logger.error(f"Error saving image narration for page {page_num}: {str(e)}")
        else:
            logger.info(f"No image content to save for page {page_num}")

        # Log completion for this page
        logger.info(f"Successfully saved all content for page {page_num}")

    except Exception as e:
        logger.error(f"Error saving output files for page {page_num}: {str(e)}")
        saved_files['error'] = str(e)

    return saved_files



# def dataframe_to_markdown(df):
#     """Convert a pandas DataFrame to a markdown table string"""
#     if df is None or df.empty:
#         return "Empty table"

#     # Convert to string buffer
#     buffer = StringIO()
#     df.to_markdown(buf=buffer, index=False)
#     md_table = buffer.getvalue()

#     # If to_markdown fails, fall back to more basic formatting
#     if not md_table:
#         headers = df.columns.tolist()
#         header_row = "| " + " | ".join(str(h) for h in headers) + " |"
#         separator = "| " + " | ".join("-" * len(str(h)) for h in headers) + " |"

#         rows = []
#         for _, row in df.iterrows():
#             rows.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")

#         md_table = header_row + "\n" + separator + "\n" + "\n".join(rows)

#     return md_table

# def upload_to_s3(local_path, s3_key):
#     """Upload a file to S3 bucket"""
#     s3_client = boto3.client('s3')
#     try:
#         s3_client.upload_file(local_path, S3_BUCKET, s3_key)
#         return f"s3://{S3_BUCKET}/{s3_key}"
#     except Exception as e:
#         print(f"Error uploading to S3: {str(e)}")
#         return None

# def upload_df_to_s3(df, s3_key):
#     """Upload a DataFrame directly to S3 as CSV"""
#     s3_client = boto3.client('s3')
#     try:
#         csv_buffer = StringIO()
#         df.to_csv(csv_buffer, index=False)
#         s3_client.put_object(
#             Bucket=S3_BUCKET,
#             Key=s3_key,
#             Body=csv_buffer.getvalue()
#         )
#         return f"s3://{S3_BUCKET}/{s3_key}"
#     except Exception as e:
#         print(f"Error uploading DataFrame to S3: {str(e)}")
#         return None

# def upload_image_to_s3(img, s3_key):
#     """Upload a PIL Image directly to S3"""
#     s3_client = boto3.client('s3')
#     try:
#         buffer = BytesIO()
#         img.save(buffer, format="PNG")
#         buffer.seek(0)
#         s3_client.put_object(
#             Bucket=S3_BUCKET,
#             Key=s3_key,
#             Body=buffer.getvalue()
#         )
#         return f"s3://{S3_BUCKET}/{s3_key}"
#     except Exception as e:
#         print(f"Error uploading image to S3: {str(e)}")
#         return None

# def generate_report(combined_all, output_dir="output_report"):
#     """Generate a formatted report from the combined_all data structure and save to S3"""

#     # Create output directory if it doesn't exist (for local files)
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

#     # Initialize S3 paths for the report components
#     s3_report_path = f"{output_dir}/report.txt"
#     s3_images_prefix = f"{output_dir}/images"
#     s3_tables_prefix = f"{output_dir}/tables"

#     # Create a report in memory first
#     report_content = StringIO()

#     # Process each page
#     for page_num, page_data in enumerate(combined_all, 1):
#         report_content.write(f"Page no:{page_num}\n")

#         # Extract the cleaned content
#         cleaned_content = page_data.get('cleaned_content', '')

#         # Process text, images, and tables in the content
#         if cleaned_content:
#             # Split content by tags
#             parts = re.split(r'(<image>|</image>|<table>|</table>|<text>|</text>)', cleaned_content)

#             current_tag = None
#             content_buffer = ""

#             for part in parts:
#                 import ipdb;ipdb.set_trace()
#                 if part in ("<image>", "<table>", "<text>"):
#                     current_tag = part
#                     content_buffer = ""
#                 elif part in ("</image>", "</table>", "</text>"):
#                     # Process the completed tag
#                     if current_tag == "<image>" and content_buffer.strip():
#                         report_content.write("<image>\n")

#                         # Process and upload image if available
#                         if page_data.get('image_content') and page_data['image_content'].get('image'):
#                             # Save locally first
#                             local_image_path = os.path.join(output_dir, "images", f"image_{page_num}.png")
#                             page_data['image_content']['image'].save(local_image_path)

#                             # Upload to S3
#                             s3_image_key = f"{s3_images_prefix}/image_{page_num}.png"
#                             s3_url = upload_image_to_s3(page_data['image_content']['image'], s3_image_key)

#                             if s3_url:
#                                 report_content.write(f"[Image saved to {s3_url}]\n")
#                             else:
#                                 report_content.write(f"[Image saved locally to {local_image_path}]\n")

#                         # Write the image description
#                         report_content.write(content_buffer)
#                         report_content.write("\n</image>\n\n")

#                     elif current_tag == "<table>" and content_buffer.strip():
#                         report_content.write("<table>\n")
#                         report_content.write(content_buffer + "\n")  # Write the table reference itself

#                         # Check if this is a table reference
#                         table_ref_match = re.search(r'page_(\d+)_page(\d+)\.(\d+)', content_buffer.strip(), re.IGNORECASE)

#                         if table_ref_match or "page" in content_buffer.lower():
#                             # Process tables in the content array
#                             if page_data.get('table_content') and page_data['table_content'].get('content'):
#                                 # First process markdown and csv files
#                                 for idx, table_item in enumerate(page_data['table_content']['content']):
#                                     # Check for markdown representation
#                                     if 'claude_analysis' in table_item and 'content' in table_item['claude_analysis']:
#                                         content = table_item['claude_analysis']['content']
#                                         if '|' in content and '-' in content:  # Simple check for markdown table
#                                             report_content.write(f"Table {idx+1} (markdown):\n\n")
#                                             report_content.write(f"```\n{content}\n```\n\n")

#                                     # Always save CSV files regardless of markdown
#                                     if 'data' in table_item:
#                                         try:
#                                             # Convert to dataframe if necessary
#                                             if not isinstance(table_item['data'], pd.DataFrame):
#                                                 df = pd.DataFrame(table_item['data'])
#                                             else:
#                                                 df = table_item['data']

#                                             # Save locally
#                                             local_table_path = os.path.join(output_dir, "tables", f"table_{page_num}_{idx+1}.csv")
#                                             df.to_csv(local_table_path, index=False)

#                                             # Upload to S3
#                                             s3_table_key = f"{s3_tables_prefix}/table_{page_num}_{idx+1}.csv"
#                                             s3_url = upload_df_to_s3(df, s3_table_key)

#                                             if s3_url:
#                                                 report_content.write(f"Table {idx+1}: [Table saved to {s3_url}]\n\n")
#                                             else:
#                                                 report_content.write(f"Table {idx+1}: [Table saved locally to {local_table_path}]\n\n")
#                                         except Exception as e:
#                                             report_content.write(f"Error saving table data: {str(e)}\n\n")

#                             # Then add narration once (prioritize table_results narration if it exists)
#                             narration_added = False

#                             # First check for narration in table_results
#                             if page_data['table_content'].get('table_results'):
#                                 for idx, table_result in enumerate(page_data['table_content']['table_results']):
#                                     if 'table_narration' in table_result and table_result['table_narration'].strip():
#                                         report_content.write(f"{table_result['table_narration']}\n\n")
#                                         narration_added = True
#                                         break

#                             # If no narration from table_results, use the general narration
#                             if not narration_added and page_data['table_content'].get('narration'):
#                                 report_content.write(page_data['table_content']['narration'] + "\n\n")
#                                 narration_added = True

#                             # Process any table results files that need to be saved
#                             if page_data['table_content'].get('table_results'):
#                                 for idx, table_result in enumerate(page_data['table_content']['table_results']):
#                                     # Save CSV from table data
#                                     if 'table_data' in table_result and 'data' in table_result['table_data']:
#                                         try:
#                                             data = table_result['table_data']['data']
#                                             if not isinstance(data, pd.DataFrame):
#                                                 df = pd.DataFrame(data)
#                                             else:
#                                                 df = data

#                                             # Save locally
#                                             # local_table_path = os.path.join(output_dir, "tables", f"table_{page_num}_result_{idx+1}.csv")
#                                             # df.to_csv(local_table_path, index=False)

#                                             # Upload to S3
#                                             s3_table_key = f"{s3_tables_prefix}/table_{page_num}_result_{idx+1}.csv"
#                                             s3_url = upload_df_to_s3(df, s3_table_key)

#                                             if s3_url:
#                                                 report_content.write(f"Table Result {idx+1}: [Table saved to {s3_url}]\n\n")
#                                             else:
#                                                 report_content.write(f"Table Result {idx+1}: [Table saved locally to {local_table_path}]\n\n")
#                                         except Exception as e:
#                                             report_content.write(f"Error saving table result data: {str(e)}\n\n")
#                         else:
#                             # Not a reference - just include the content directly
#                             # But still check for any tables in the data
#                             if page_data.get('table_content'):
#                                 # Process tables in the content array
#                                 if page_data['table_content'].get('content'):
#                                     for idx, table_item in enumerate(page_data['table_content']['content']):
#                                         # Always save CSV files regardless of markdown
#                                         if 'data' in table_item:
#                                             try:
#                                                 # Convert to dataframe if necessary
#                                                 if not isinstance(table_item['data'], pd.DataFrame):
#                                                     df = pd.DataFrame(table_item['data'])
#                                                 else:
#                                                     df = table_item['data']

#                                                 # Save locally
#                                                 local_table_path = os.path.join(output_dir, "tables", f"table_{page_num}_{idx+1}.csv")
#                                                 df.to_csv(local_table_path, index=False)

#                                                 # Upload to S3
#                                                 s3_table_key = f"{s3_tables_prefix}/table_{page_num}_{idx+1}.csv"
#                                                 s3_url = upload_df_to_s3(df, s3_table_key)

#                                                 if s3_url:
#                                                     report_content.write(f"Table {idx+1}: [Table saved to {s3_url}]\n\n")
#                                                 else:
#                                                     report_content.write(f"Table {idx+1}: [Table saved locally to {local_table_path}]\n\n")
#                                             except Exception as e:
#                                                 report_content.write(f"Error saving table data: {str(e)}\n\n")

#                                 # Add narration only once
#                                 narration_added = False
#                                 if page_data['table_content'].get('narration') and not narration_added:
#                                     report_content.write(page_data['table_content']['narration'] + "\n\n")

#                         report_content.write("\n</table>\n\n")

#                     elif current_tag == "<text>" and content_buffer.strip():
#                         report_content.write("<text>\n")
#                         report_content.write(content_buffer)
#                         report_content.write("\n</text>\n\n")

#                     current_tag = None
#                 elif current_tag is not None:
#                     content_buffer += part

#         # If there's no explicit text/table/image tags but we have content
#         if not any(tag in cleaned_content for tag in ["<text>", "<table>", "<image>"]):
#             # Check if it's likely a table page
#             if page_data.get('table_content') and (
#                 page_data['table_content'].get('content') or
#                 page_data['table_content'].get('table_results')
#             ):
#                 report_content.write("<table>\n")

#                 # First save markdown content and CSV files
#                 if page_data['table_content'].get('content'):
#                     for idx, table_item in enumerate(page_data['table_content']['content']):
#                         # Check for markdown first
#                         if 'claude_analysis' in table_item and 'content' in table_item['claude_analysis']:
#                             content = table_item['claude_analysis']['content']
#                             if '|' in content and '-' in content:  # Simple check for markdown table
#                                 report_content.write(f"Table {idx+1} (markdown):\n\n")
#                                 report_content.write(f"```\n{content}\n```\n\n")

#                         # Always save CSV files
#                         if 'data' in table_item:
#                             try:
#                                 # Convert to dataframe if necessary
#                                 if not isinstance(table_item['data'], pd.DataFrame):
#                                     df = pd.DataFrame(table_item['data'])
#                                 else:
#                                     df = table_item['data']

#                                 # Save locally
#                                 local_table_path = os.path.join(output_dir, "tables", f"table_{page_num}_{idx+1}.csv")
#                                 df.to_csv(local_table_path, index=False)

#                                 # Upload to S3
#                                 s3_table_key = f"{s3_tables_prefix}/table_{page_num}_{idx+1}.csv"
#                                 s3_url = upload_df_to_s3(df, s3_table_key)

#                                 if s3_url:
#                                     report_content.write(f"Table {idx+1}: [Table saved to {s3_url}]\n\n")
#                                 else:
#                                     report_content.write(f"Table {idx+1}: [Table saved locally to {local_table_path}]\n\n")
#                             except Exception as e:
#                                 report_content.write(f"Error saving table data: {str(e)}\n\n")

#                 # Then add narration once (prioritize table_results narration if it exists)
#                 narration_added = False

#                 # First check for narration in table_results
#                 if page_data['table_content'].get('table_results'):
#                     for idx, table_result in enumerate(page_data['table_content']['table_results']):
#                         if 'table_narration' in table_result and table_result['table_narration'].strip():
#                             report_content.write(f"{table_result['table_narration']}\n\n")
#                             narration_added = True
#                             break

#                 # If no narration from table_results, use the general narration
#                 if not narration_added and page_data['table_content'].get('narration'):
#                     report_content.write(page_data['table_content']['narration'] + "\n\n")

#                 # Process table results files that need to be saved
#                 if page_data['table_content'].get('table_results'):
#                     for idx, table_result in enumerate(page_data['table_content']['table_results']):
#                         # Save data from table results
#                         if 'table_data' in table_result and 'data' in table_result['table_data']:
#                             try:
#                                 data = table_result['table_data']['data']
#                                 if not isinstance(data, pd.DataFrame):
#                                     df = pd.DataFrame(data)
#                                 else:
#                                     df = data

#                                 # Save locally
#                                 local_table_path = os.path.join(output_dir, "tables", f"table_{page_num}_result_{idx+1}.csv")
#                                 df.to_csv(local_table_path, index=False)

#                                 # Upload to S3
#                                 s3_table_key = f"{s3_tables_prefix}/table_{page_num}_result_{idx+1}.csv"
#                                 s3_url = upload_df_to_s3(df, s3_table_key)

#                                 if s3_url:
#                                     report_content.write(f"Table Result {idx+1}: [Table saved to {s3_url}]\n\n")
#                                 else:
#                                     report_content.write(f"Table Result {idx+1}: [Table saved locally to {local_table_path}]\n\n")
#                             except Exception as e:
#                                 report_content.write(f"Error saving table result data: {str(e)}\n\n")

#                 report_content.write("\n</table>\n\n")

#             # Handle image content if available but not tagged
#             elif page_data.get('image_content') and page_data['image_content'].get('image'):
#                 report_content.write("<image>\n")

#                 # Save locally
#                 local_image_path = os.path.join(output_dir, "images", f"image_{page_num}.png")
#                 page_data['image_content']['image'].save(local_image_path)

#                 # Upload to S3
#                 s3_image_key = f"{s3_images_prefix}/image_{page_num}.png"
#                 s3_url = upload_image_to_s3(page_data['image_content']['image'], s3_image_key)

#                 if s3_url:
#                     report_content.write(f"[Image saved to {s3_url}]\n")
#                 else:
#                     report_content.write(f"[Image saved locally to {local_image_path}]\n")

#                 if page_data['image_content'].get('narration'):
#                     report_content.write(page_data['image_content']['narration'])

#                 report_content.write("\n</image>\n\n")

#         # Add page separator
#         if page_num < len(combined_all):
#             report_content.write("\n" + "-" * 80 + "\n\n")

#     # Save the report locally
#     local_report_path = os.path.join(output_dir, "report.txt")
#     with open(local_report_path, "w", encoding="utf-8") as f:
#         f.write(report_content.getvalue())

#     # Upload the final report to S3
#     s3_client = boto3.client('s3')
#     try:
#         s3_client.put_object(
#             Bucket=S3_BUCKET,
#             Key=s3_report_path,
#             Body=report_content.getvalue()
#         )
#         s3_report_url = f"s3://{S3_BUCKET}/{s3_report_path}"
#         print(f"Report uploaded to S3: {s3_report_url}")
#     except Exception as e:
#         s3_report_url = None
#         print(f"Error uploading report to S3: {str(e)}")
#         print(f"Report saved locally to: {local_report_path}")

#     return s3_report_url or local_report_path



def download_s3_file_to_memory(s3_uri):
    """
    Download a file from S3 to memory using a S3 URI

    Parameters:
    s3_uri (str): S3 URI in the format s3://bucket-name/path/to/file

    Returns:
    dict: JSON content loaded into memory
    """
    # Parse the S3 URI to extract bucket and key
    match = re.match(r's3://([^/]+)/(.+)', s3_uri)
    if not match:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    bucket_name = match.group(1)
    object_key = match.group(2)

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Create a bytes buffer to store the file content
        file_buffer = BytesIO()

        # Download the file to the buffer
        s3_client.download_fileobj(bucket_name, object_key, file_buffer)

        # Reset buffer position to the beginning
        file_buffer.seek(0)

        # Load JSON from the buffer
        json_content = json.load(file_buffer)

        print(f"Successfully loaded JSON from {s3_uri}")
        return json_content

    except Exception as e:
        print(f"Error downloading or parsing file: {e}")
        return None



# NEW ONE
def process_report(combined_all, processing_reports, bucket, output_dir, isBDA=False):
    """
    Process the combined_all data to generate a report with tables and images

    Args:
        combined_all: List of dictionaries containing content data
        processing_reports: Dict containing the tokens and times
        output_dir: Output directory for the report and assets
        isBDA: Boolean indicating if the input is from BDA processing

    Returns:
        str: Path to the generated report
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    processing_report_path = os.path.join(output_dir, 'processing_report.json')
    with open(processing_report_path, 'w') as f:
        json.dump(processing_reports, f, indent=4)


    if isBDA:

        report_text = combined_all['final_report']
        image_names = combined_all['image_names']
        table_names = combined_all['table_names']
        dataframes = combined_all['dataframes']
        cropped_images = combined_all['cropped_images']
        for image_name, cropped_img in zip(image_names, cropped_images):
            img_filename = f"images/{image_name}.png"
            full_img_path = os.path.join(output_dir, img_filename)
            cropped_img.save(full_img_path)
        for table_name, dataframe in zip(table_names, dataframes):
            csv_filename = f"tables/{table_name}.csv"
            full_csv_path = os.path.join(output_dir, csv_filename)
            dataframe.to_csv(full_csv_path, index=False)

    else:

        report_lines = []   
        for page_idx, page_data in enumerate(combined_all):
            page_number = page_idx + 1
            page_lines = []
            page_lines.append(f"Page no:{page_number}")

            # Process cleaned_content (contains text and images)
            if 'cleaned_content' in page_data and page_data['cleaned_content']:
                content = page_data['cleaned_content']

                # Extract text blocks
                text_blocks = re.findall(r'<text>(.*?)</text>', content, re.DOTALL)
                for text in text_blocks:
                    page_lines.append(f"<text>{text.strip()}</text>")

                # Extract image blocks
                image_blocks = re.findall(r'<image>(.*?)</image>', content, re.DOTALL)
                for img_idx, image_block in enumerate(image_blocks):
                    img_filename = f"images/image_page_{page_number}_{img_idx+1}.png"
                    full_img_path = os.path.join(output_dir, img_filename)

                    # Save image if available
                    if 'image_content' in page_data and hasattr(page_data['image_content'].get('image'), 'save'):
                        try:
                            page_data['image_content']['image'].save(full_img_path)
                            page_lines.append(f"![Figure {page_number}.{img_idx+1}]({img_filename})")
                            page_lines.append(f"*{image_block.strip()}*")
                        except Exception:
                            pass

            # Process tables
            if 'table_content' in page_data:
                table_data = page_data['table_content']

                # Check for structured table results

                if table_data.get('table_results'):
                    for table_idx, table in enumerate(table_data['table_results']):
                        table_id = f"table_page_{page_number}_{table_idx+1}"
                        csv_filename = f"tables/{table_id}.csv"
                        full_csv_path = os.path.join(output_dir, csv_filename)

                        # Process and save table data

                        if isinstance(table, dict) and 'data' in table['table_data'] and len(table['table_data']['data'])>0:
                            try:
                                df = pd.DataFrame(table['table_data']['data'])
                                df.to_csv(full_csv_path, index=False)
                                # Create markdown representation of table
                                table_md = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

                                page_lines.append(f"\n**{table_id}**\n")
                                page_lines.append(table_md)
                                # Add narration if available
                                if 'table_narration' in table:
                                    page_lines.append(f"\n*{table['table_narration']}*\n")
                            except Exception:
                                pass

                # Extract tables from table_content.content if no structured results
                elif table_data.get('content'):
                    # Try to parse table content from text
                    table_text = extract_table_text(table_data['content'])
                    if table_text:
                        table_id = f"Table_page_{page_number}_text"
                        page_lines.append(f"\n**{table_id}**\n")
                        page_lines.append("```")
                        page_lines.append(table_text)
                        page_lines.append("```")

                        # Save table text to file
                        with open(os.path.join(output_dir, f"tables/{table_id}.txt"), 'w', encoding='utf-8') as f:
                            f.write(table_text)


            # If page is empty, mark it as blank
            if len(page_lines) == 1:  # Only has the page number
                page_lines.append("This page is blank originally")

            report_lines.append("\n\n".join(page_lines))

        # Write final report
        report_text = "\n\n".join(report_lines)

    report_path = os.path.join(output_dir, "final_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Upload to S3
    #upload_to_s3(output_dir, bucket)

    return report_path

def extract_table_text(content_list):
    """Extract table text from content list"""
    if not content_list:
        return ""

    text_parts = []
    for item in content_list:
        if isinstance(item, dict) and 'text' in item:
            text_parts.append(item['text'])
        elif isinstance(item, dict) and 'type' in item and item['type'] == 'text' and 'text' in item:
            text_parts.append(item['text'])

    return "\n".join(text_parts)

def upload_to_s3(local_dir, bucket_name, prefix=""):
    """Upload directory contents to S3 bucket"""
    try:
        s3_client = boto3.client('s3')

        for root, _, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = os.path.join(prefix, relative_path).replace("\\", "/")

                s3_client.upload_file(local_path, bucket_name, s3_key)
    except Exception:
        pass

def parse_table(table_text):
    """Try to parse a text representation of a table into a DataFrame"""
    try:
        # Check if this is a markdown table
        if '|' in table_text and '+' not in table_text:
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            # Filter out separator lines like |---|---|
            data_lines = [line for line in lines if not all(c in '|-:' for c in line.replace(' ', ''))]

            rows = []
            for line in data_lines:
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                rows.append(cells)

            if rows:
                headers = rows[0]
                data = rows[1:] if len(rows) > 1 else []
                return pd.DataFrame(data, columns=headers)

        # Try for ASCII/fixed-width table format
        if '+' in table_text and '|' in table_text:
            # This is complex and might need a specialized parser
            pass

        # Fall back to basic parsing
        lines = [line for line in table_text.split('\n') if line.strip()]
        data = []
        for line in lines:
            if line.strip():
                cells = re.split(r'\s{2,}', line.strip())
                data.append(cells)

        if data:
            headers = data[0]
            rows = data[1:]
            return pd.DataFrame(rows, columns=headers)

    except Exception:
        pass

    return None

def find_tables_in_text(text):
    """
    Find and extract tables from text content
    Returns a list of dictionaries with table info
    """
    tables = []

    # Match both HTML-style tables and markdown tables
    table_patterns = [
        r'<table>(.*?)</table>',              # HTML table tags
        r'(\|.*\|\n\|[-:]+\|\n(\|.*\|\n)+)'   # Markdown tables
    ]

    for pattern in table_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            table_content = match[0] if isinstance(match, tuple) else match
            table_lines = table_content.strip().split('\n')

            # Try to identify a table ID or title
            table_id = table_lines[0].strip() if table_lines else "Unknown table"

            tables.append({
                'id': table_id,
                'content': table_content,
                'data': parse_table(table_content)
            })

    return tables


def generate_full_report_from_BDA(doc_bda_content, pdf_images, image_processor, table_processor):
    """
    Generate a report from the pages in doc_bda_content and extract FIGURE and TABLE elements

    Parameters:
    doc_bda_content (dict): The document content with pages and elements
    pdf_images (list): List of PIL images from PDF pages
    image_processor: Instance of ImageProcessor for generating image narrations
    table_processor: Instance of TableProcessor for generating table narrations

    Returns:
    tuple: (str, list) - The generated report text and list of extracted elements with images
    """
    report = []
    elements = doc_bda_content.get('elements', [])
    input_tokens = 0
    output_tokens = 0
    # Process each page

    final_results = {}
    image_names = []
    table_names = []
    dataframes = []
    cropped_images = []
    for i, element in enumerate(elements):
        element_type = element['type']
        markdown_content = element.get('representation', {}).get('markdown', '')
        # page_num = element['page_indices'][0]
        locations = element.get('locations', [])
        for loc in range(len(locations)):
            page_index = locations[loc]['page_index']
            original_image = pdf_images[page_index]
            actual_width, actual_height = original_image.size

            # Get bounding box
            bbox = locations[loc].get('bounding_box', {})
            # Convert normalized coordinates to pixels
            left = int(bbox.get('left', 0) * actual_width)
            top = int(bbox.get('top', 0) * actual_height)
            width = int(bbox.get('width', 0) * actual_width)
            height = int(bbox.get('height', 0) * actual_height)

            # Ensure coordinates are within bounds
            left = max(0, left)
            top = max(0, top)
            right = min(actual_width, left + width)
            bottom = min(actual_height, top + height)

            # Crop the image
            cropped_img = original_image.crop((left, top, right, bottom))
            # Add page to report
            report.append(f"Page no:{page_index+1}")

            if 'FIGURE' in element_type:

                img_narration = image_processor.image_narration_gen(cropped_img, 'english')
                content_loc = img_narration['narration'].find("Content")
                img_content = img_narration['narration'][content_loc+11:]
                report.append(f"{markdown_content}")
                report.extend([
                        "<image>",
                        img_content if img_content else "No narration available",
                        "</image>"
                    ])
                input_tokens += img_narration.get('input_tokens', 0)
                output_tokens += img_narration.get('output_tokens', 0)
                image_names.append(element['id'])
                cropped_images.append(cropped_img)

            elif 'TABLE' in element_type:
                table_narration = table_processor.generate_table_narration(markdown_content, 'english')
                report.append(f"{markdown_content}")
                report.extend([
                        "<table>",
                        table_narration[0],
                        "</table>"
                    ])
                buffer = io.StringIO(markdown_content)
                rows = []
                for row in buffer.readlines():
                    if not row.strip().startswith('|'):
                        continue
                    if row.strip().endswith('|'):
                        if row.endswith('|\n'):
                            tmp = row.strip()[1:-1]  # Remove | and newline
                        else:
                            tmp = row[1:-1]  # Just remove |
                    else:
                        continue  # Skip malformed rows
                    clean_line = [col.strip() for col in tmp.split('|')]

                    rows.append(clean_line)
                if len(rows) >= 3:
                    if all(('-' in cell or ':' in cell) for cell in rows[1]):
                        rows = rows[:1] + rows[2:]

                df = pd.DataFrame(rows)
                dataframes.append(df)
                table_names.append(element['id'])
                input_tokens += table_narration[1].get('input_tokens', 0)
                output_tokens += table_narration[1].get('output_tokens', 0)
            elif markdown_content.strip():
                # report.append(f"{markdown_content}")
                report.extend([
                        "<text>",
                        markdown_content,
                        "</text>"
                    ])
            else:
                report.append("This page is blank originally")
            report.append("")


    # Join the report
    final_report = "\n".join(report)
    final_results['final_report'] = final_report
    final_results['input_tokens'] = input_tokens
    final_results['output_tokens'] = output_tokens
    final_results['image_names'] = image_names
    final_results['table_names'] = table_names
    final_results['dataframes'] = dataframes
    final_results['cropped_images'] = cropped_images
    return final_results, input_tokens, output_tokens


