import re
import os
def process_file(input_file):
    with open(input_file, 'r') as file:
        content = file.read()

    # Extract the page number from the top of the file
    page_number_match = re.search(r'Page no:(\d+)', content)
    if page_number_match:
        page_number = page_number_match.group(1)
    else:
        print("Page number not found in the file.")
        return

    # Find all table tags and their contents
    table_pattern = re.compile(r'<table>(.*?)</table>', re.DOTALL)
    tables = table_pattern.findall(content)

    for table in tables:
        # Extract the identifier
        identifier = table.strip()

        # Split the identifier
        identifier_parts = identifier.split('_')
        
        if len(identifier_parts) >= 2:
            current_page = identifier_parts[0]
            
            # Check if the page number in the identifier matches the actual page number
            if current_page != f"page{page_number}":
                # Replace the incorrect page number with the correct one
                new_identifier = f"page{page_number}_" + "_".join(identifier_parts[1:])
                
                # Replace the old identifier with the new one in the content
                content = content.replace(f"<table>{identifier}</table>", 
                                          f"<table>{new_identifier}</table>")
                print(f"Replaced {identifier} with {new_identifier}")
            else:
                print(f"Identifier {identifier} is correct")
        else:
            print(f"Invalid identifier format: {identifier}")

    tables = table_pattern.findall(content)

    for i, table in enumerate(tables):
        # Extract the identifier
        identifier = table.strip()

        # Construct the file path
        file_path = f"/tmp/table_narration/{identifier}.txt"

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                table_content = file.read()

            # Replace the identifier with the new table name
            #changed here ****
            #page_number = identifier.split('_')[0][-1]
            page_number = re.search(r'(\d+)', identifier.split('_')[0]).group(1)
            new_table_name = f"table_{i+1}_page{page_number}"

            # Replace the old table content with the new content
            #changes here : commented the below line
            table_content = table_content.replace(f"{identifier}.txt", "")
            #This is an extra line: not required 
            #table_content = table_content.replace(f"{identifier}.txt", f"{identifier}")
            #changes here : commented the below line
            #content = content.replace(f"<table>{identifier}</table>", 
             #                         f"<table>{new_table_name}\n{table_content}</table>")
            #This line is added
            content = content.replace(f"<table>{identifier}</table>", 
                                      f"<table> Table_{identifier}\n{table_content}</table>")
            #content = content.replace(f"<table>{new_table_name}\n{table_content}</table>")
        else:
            print(f"File not found: {file_path}")
            
    def replace_image(match):
        nonlocal image_count
        image_count += 1
        identifier = f'image{image_count}_Page{page_number}'
        image_content = match.group(1)
        return f'<image>\n{identifier}\n{image_content}</image>'
    
    # Initialize image count
    image_count = 0

    # Find and replace all <image> tags
    updated_content = re.sub(r'<image>(.*?)</image>', replace_image, content, flags=re.DOTALL)

    
    
    
    # Write the modified content to the output file
    return  updated_content
    
    
    
def extract_paragraphs(text):
    """
    Extract paragraphs from a text file, removing short lines, special character lines,
    and tabular structures.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to save the cleaned output
    """
    try:
        # Read the input text file
        # with open(input_file, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()
        
        clean_paragraphs = []
        current_paragraph = []
        
        for line in text:
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_paragraph:
                    # Join current paragraph and add to results
                    clean_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Count words in the line
            words = line.split()
            word_count = len(words)
            
            # Skip short lines (less than 5 words)
            # if word_count < 5:
            #     continue
            
            
            if re.search(r'[\t│|]|( {2,})', line) or re.match(r'^[^a-zA-Z0-9\s]*$', line):
                continue
                
            # If we got here, the line is likely part of a paragraph
            current_paragraph.append(line)
        
        # Add the last paragraph if there's any remaining
        if current_paragraph:
            clean_paragraphs.append(' '.join(current_paragraph))
        
        
        return clean_paragraphs
    
    except Exception as e:
        return False, f"Error: {str(e)}"


def remove_tags_simple(filename):
    """
    Simple function to remove table tags from a file
    
    Args:
        filename (str): Path to the file to clean
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return False
    
    try:
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
            print(content)
        
    
        pattern = r'<table>page\d+_page1\.\d+</table>'
        
        
        # Count matches before removal
        matches = re.findall(pattern, content)
        print(f"Found {len(matches)} table tags to remove")
        
        # Remove the tags
        cleaned_content = re.sub(pattern, '', content)
        
        # Write back to original file
        # with open(filename, 'w', encoding='utf-8') as f:
        #     f.write(cleaned_content)
        
        print(f"Successfully removed {len(matches)} table tags from {filename}")
        print(f"Original size: {len(content)} characters")
        print(f"Cleaned size: {len(cleaned_content)} characters")
        print(f"Characters removed: {len(content) - len(cleaned_content)}")
        
        return cleaned_content
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False