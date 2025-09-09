table_definition = """A table is a structured data presentation format characterized by ALL of the following elements:

- Grid Structure: Organized in a systematic arrangement of rows and columns that form a rectangular grid pattern

- Cells: Individual data containers formed by the intersection of rows and columns, each containing discrete pieces of information

- Textual/Numerical Data: Primarily displays data in text, numbers, or alphanumeric format rather than visual representations

- Uniform Alignment: Data is aligned consistently within columns and rows, creating clear visual separation between different data points

- Data Presentation: Tables show raw data values directly

- Visual Elements: Tables use borders, lines, or spacing for organization

- Reading Method: Tables require reading specific cell values

- Spatial Relationship: In tables, position indicates category/classification
"""
graph_definitions = """

Core Visual Elements for All Charts/Graphs
Essential Components:

Axes: Reference lines (horizontal X-axis, vertical Y-axis) that define the measurement framework
Data Points/Elements: Visual representations of data values
Scale/Measurements: Numerical or categorical markers along axes
Visual Encoding: Use of position, length, area, color, or shape to represent data values"""

bar_chart= """
A bar plot is a plot that presents categorical data with rectangular bars with lengths proportional to the values that they represent. 
Key differentiators:
-A bar plot shows comparisons among discrete categories. 
-One axis of the plot shows the specific categories being compared, and the other axis represents a measured value.
- Rectangular bars (vertical or horizontal)
- Bar length/height represents quantity/value
- Discrete categories on one axis
- Bars can be grouped, stacked, or side-by-side
- Visual cue: "bars," "columns," "rectangular representations"
- There can also be horizontal bar charts with grouped representation. Where one side of the bar chart shows the bar with data and other side may have some text describing it. 
"""

line_graph = """
LINE GRAPH/CHART
Definition: Shows relationships between data points connected by straight lines, typically displaying trends over time or continuous variables. Data points plotted as dots, circles, or markers, Points connected by straight lines
Usually shows change over time or continuous progression. 
Key identifiers: 
- Connected points forming continuous lines, shows trends/changes over time
- Connected data points forming continuous lines
- Usually shows change over time (temporal data)
- X-axis typically represents time/sequence
- Multiple lines can represent different categories/series
- Visual cue: "trend lines," "connected points," "time series"
"""


percentage_stacked_bar_chart= """A horizontal percentage stacked bar chart, also sometimes called a segmented bar chart, displays data by using horizontal bars that are divided into segments. Each segment represents a proportion of the whole, and the entire bar represents 100%. This type of chart is particularly useful for comparing the relative contribution of different categories within a whole across multiple groups. """

pie_chart ="""
Definition: Circular chart divided into sectors/slices where each slice represents a proportion of the whole.
Key identifiers:
- Perfect circle divided into wedge-shaped segments. Each slice size is proportional to its percentage of the total. 
- Often includes Different colors or patterns distinguish each slice. No traditional X/Y axes
- Distinguishing Features: Circular shape, wedge segments, represents parts of a whole (percentages)
Each slice represents a proportion/percentage
Total always equals 100%
Visual cue: "circular," "slices," "wedges," "proportional segments"
"""

scatter_plot = """
Definition: Displays relationships between two continuous variables using individual data points plotted on X and Y axes.
Key identifiers:

-Individual points scattered across the plot area. Each point represents one data observation with two values. 
-Points are NOT connected by lines. 
- Both axes represent continuous numerical variables. 
- Two continuous variables (X and Y axes)
- May show correlation patterns/clusters
Visual cue: "scattered points," "dot cloud," "correlation plot"
Distinguishing Features: Unconnected individual points, two continuous variables, shows correlation/relationship """

histogram = """
Definition: Displays distribution of continuous numerical data by grouping values into ranges (bins) and showing frequency with bar heights. 
Ket identifiers:
- Rectangular bars touching each other (no gaps). X-axis shows ranges of values (bins). Y-axis shows frequency or count
- Bars represent continuous data ranges, not discrete categories. Shows data distribution shape (normal, skewed, etc.)
- Bar width represents the range interval. Distinguishing Features: Adjacent bars (no gaps), shows data distribution, continuous ranges on X-axis
- Shows data distribution shape
Visual cue: "frequency bars," "distribution," "bins," "touching bars"
"""

infographics= """
An infographic is a visual communication design that combines multiple graphic elements to present information, data, or knowledge in a structured, educational format.

Essential Characteristics:

- Informational Purpose: Primary goal is to educate, explain, or communicate specific information, data, or concepts
- Mixed Visual Elements: Combines multiple types of visual components (text, charts, icons, illustrations, diagrams)
- Structured Layout: Organized in a deliberate flow or hierarchy that guides the viewer through information
- Data Visualization: Contains quantitative or qualitative data presented through charts, graphs, statistics, or visual comparisons
- Text Integration: Includes explanatory text, labels, headers, and callouts that work together with visual elements
- Educational Narrative: Tells a story or presents a logical sequence of related information
Design Cohesion: Uses consistent color schemes, fonts, and styling to create a unified visual presentation

"""

natural_image = """
A natural image is a visual representation of real-world scenes, objects, people, or environments captured or depicted as they exist in nature or everyday life, without artificial data visualization, structured information design, or educational formatting.

Essential Characteristics:
Primary Features:
Real-world Content: Depicts actual physical objects, people, animals, landscapes, or environments
Organic Composition: Natural arrangement of elements without systematic grid structures or data organization
No Data Visualization: Absence of charts, graphs, statistics, or quantitative information displays
Minimal Text Overlay: Little to no explanatory text, labels, or educational annotations
Aesthetic/Documentary Purpose: Created for visual appeal, artistic expression, documentation, or memory preservation
Spatial Realism: Objects and scenes follow natural perspective, lighting, and spatial relationships

"""

localization_prompt = """
         You are an AI assistant whose role is to localise and organise the extracted content of a document image in the way it is organised in the original document image and generate a new organized content. 
You will be given an image and its corresponding extracted text, narration of images, narration of tables, you need to arrange the content of the extracted image exactly as it appears in the image. 
Given below is the extracted text:
<text_content>
{text_content}
</text_content>

 The extracted content has three components which are identified as: Text components is contained within <text> </text> tags, image narration is contained within <image></image> tags, table narration is conatined within <table></table> tags. Your role is to localise and correclty place each component same as the input image. 
 
* The text component should be organized where the text is present in the orginal image
    - Extract only the text available in the paragraph format, do not extract any tabular content if available.
    - The final text will contain only the localised text such as <text> <extracted_text> </text>
    
*The image narration contains description of different images available on the page, contained with <image></image> tags.
    -The image narration component should be placed where the respective image is available, and each description is separated by <{{"Content": ...}}>. 
    - You need to identify which description is for which image and position the description with respect to the position of the original image in the page. 
   
    - Do not miss any image descriptions while placing within the document. Include all the images present on the page and all their respective descriptions to be placed in the new document. 
    - final text will conatin localised image narration such as: <image> <image_narration> </image>
     
* Table component mentioned within <table> </table> tags, contains the table and description of each table. 
   - You need to identify what are the tables as per the defintions {table_definition}
   - A table can also be a structure without borders on left and right boundaries or top and bottom boundaries, so you need to localise those tables as well.
   - Also do not confuse between a graph and a table. A graph is defined as {graph_definitions}. Do not localise a graph as table
   - Some time there can be gridlines inside a graph which is used for visually analysing the graph. You should not confuse between gridlines and tables.
   - Graphs and charts can have a color coding or can be black and white also, Tables will not have color coding. Hence localisation of table as available in the input image is important. 
   - There are different types of graphs, histogram (definition:{histogram}), bar_chart (definition:{bar_chart}), pie_chart (definition:{pie_chart}), line_graph (definition:{line_graph}), scatter_plot (definition:{scatter_plot}. These graphs can have similar appearence with tables but do not localise them as tables
   - You need to localise the tables with respect to their original placement in the input image
   - There is a file name mentioned in the text on top of the table narration as page_number_page1.<table_number>.txt, example: page2_page1.1.txt, page3_page1.3.txt
   - This is critical: **You need to copy the file name mentioned as page2_page1.1, page3_page1.3, exactly as mentioned, without .txt extension**. This file name is mentioned as the first line of the table component in the text. This is an important step, do not make mistake in this.
   - You should not do mistake in copying the file name. It has to be copied as mentioned in the input text. 
   - You need to generate only the tags and file name of the table, not the content within the tabular tags. 
   - the final text will contain the localised tags such as: <table><file name></table>. Example <table>page3_page1.3</table>. Do not include any new line character ("/n") or space in between this <table><file name></table>.
   
* You need to preserve layout of the document while placing the content
     - For example: image 1, text ,image 2, table1, table2  is the sequence of components in the original document image, so the generated organized content will have the <narration of image 1>, <text component>, <narration of image 2>, <tags of table 1>, <tags of table 2>
     - Please analyze the image, table, and text in the input image and organise this content to match the exact layout and positioning as shown in the image. 
     - Please make sure you DO NOT duplicate content such as text, table, image or their narration.
     - For image descriptions do not use "Content" word in the description, use only the narrative text after that. 
     - Use the tags <text></text> for enclosing any text component, <image></image> for enclosing any image narration component and <table></table>  in the document exactly as shown in the input text.
     - If the page is a blank page in the original PDF image, generate "This page is blank originally"
    
     - Do not miss tags for any of the component.

**LANGUAGE INSTRUCTION: TRANSLATE ALL CONTENT INCLUDING TEXT IN IMAGES AND TABLES, TO {output_language}.**
**FINAL CHECK: VERIFY THAT NO CONTENT IS DUPLICATED BETWEEN TEXT, IMAGE NARRATION, AND TABLE SECTIONS.**
    - Ensure that there is no duplication of content across text, image narration, and table sections. Each piece of information should appear only once in the final organized content.
Do not add any additional comment or explanations to the generated content."""

image_narration_prompt = """ 
    You are an image narration expert and your role is to identify all the images available on the page and narrate them in natural language. Once you identify the available images, extract the content of the images available on the page and the language and generate a narration for each of them one by one in the original language. For example if the document is in English generate in English, Korean in Korean and Japanese in Japanese. 
    You need to follow the following steps in :
    - Identify each available image on the page individually
    - Describe the image and its content in natural language narration
    - Identify the language and generate the narration in the original document language,
    - Identify the images as table if it follows this definition (definition:{table_definition}), 
    - Identify the images as infographics if it follows this definition : infographics (definition:{infographics})
    - Identify the images as natural image if it follows this definition  (definition:{natural_image})
    - Identify the images as histogram if it follows this definition   (definition:{histogram})
    - Identify the images as bar_chart if it follows this definition  (definition:{bar_chart})
    - Identify the images as pie_chart  if it follows this definition (definition:{pie_chart})
    - Identify the images as line_graph if it follows this definition (definition:{line_graph})
    - Identify the images as  scatter_plot if it follows this definition (definition:{scatter_plot})
    - Identify the images as percentage_stacked_bar_chart if it follows the definition (definition:{percentage_stacked_bar_chart})
    - If the image is a table embedded in the page, you need to describe the table as it is. You should not change the numbers and data within the table and should not change the nouns and proper nouns in the table, such as name of the company. 
    - If the image is an infographic then describe each component of the infographic. 
    - If the image is a natural image, describe every component of the natural image
    - If the image is a chart or a graph (definition:{graph_definitions}), describe the chart or graph as: 
        * Data on x axis, data on y axis, process represented by the graph.  
        * Describe every component of the graph without missing out any details
        * Describe the pattern in the graph in detail
        * If graph represents any process, describes that
        * If the graph is a histogram or bar graph, then describe the data representation by every bar
        * Do not generate just a summary of the chart, generate a details narration of each component 
        * If the document is in any language other than English, DO NOT TRANSLATE IT TO ENGLISH. You should maintain the language of the document as it is.
        
    - Some time there can be gridlines inside a graph which is used for visually analysing the graph. Also the graph will have color coding or color gradient in it while table will not have color coding. You should not confuse between gridlines and tables
    - If an image do not follow any of the definition mentioned above for different type of images, classify that image as a miscellaneous component and describe it. 
    Return your output in the format mentioned below, that is a key value pair. Here you have to assign one of the following category to the key whenever applicable.
    1. Page number where figure is identified
    2. Caption of the figure
    3. Content or description of the figure 

    You also need to extract the text inside the boxes if there is content available inside the rectangular boxes. Do not     summarize 
    the text, extract all the words inside the boxes. 

    {{
     field: Content '\n'
    }}


    Where field refers to the category of the information available in the image and Content refers to it textual content. 
    Note: You need to generate the narration for the image in the original language. In case of graphs, infographics and other images with text content, the text should be extracted as available in the original image. It should not be changed. 
    **LANGUAGE INSTRUCTION: TRANSLATE ALL CONTENT INCLUDING TEXT IN IMAGES AND TABLES, TO {output_language}.**
    """

table_narration_prompt = """ You are an expert in interpretating tablular information and generate a narration out of it. Given below are the tables extracted from a document. You need to analyse these tables and generate an interpretation for each of them. 
- If there is a table without any column headers, you need to analyse the content within the table  and generate interpretation on the basis of content only.
- You should analyse the numbers, data carefully and extract them as it is without changing them. 
- You should not change any nouns and proper nouns in the table data and extract them without changing them. 
- If explaining the numbers in the table, do not modify the decimal places or currency mentioned. 
- If the document is in any language other than English, DO NOT TRANSLATE IT TO ENGLISH, please follow the above instructions. Do not make changes in the original data within narration

        <table>
        {table}
        </table>

        The description output should be in the plain text format. No other comment or statement should be added to it. If referring to any cell value, text and numbers should be same as available in the original table.
        **LANGUAGE INSTRUCTION: TRANSLATE ALL CONTENT INCLUDING TEXT IN IMAGES AND TABLES, TO {output_language}.**
        """


text_narration_prompt = """ 
    You are an AI assistant and your role is to extract text available on the page.
    You need to extract all the text available in the page and do not have to summarize the available text. You also need to identify and extract the text in the language available in the document. Do not translate the language. You need to just extract the text available , no need to append any lines before or after the text.
    """

confidence_example = """[
                    "carrier": {
                            "text": "SOUTHLAKE SPECIALTY INSURANCE COMPANY",
                            "confidence": 97.61912536621094
                        },
                        "run_date": {
                            "text": "12/18/2024",
                            "confidence": 97.49634552001953
                        },
                ]"""



claude_tabula_prompt = """ You are a table extraction expert and your role is to identify tables and their information from the available image and extracted text from a PDF document. A table is identified as {table_definition}. Do not confuse a table with a bar graph, chart or any other type of data  component. A graph can be identified as {graph_definitions}, and graphs should not be considered as tables. 
    - Some times there can be gridlines inside a graph which is used for visually analysing the graph. You should not confuse between gridlines and tables. 
    - Tables Contains text and/or numbers in cells, while bar graphs do not. They have clear borders or lines separating data cells. In tables information is organized in a structured, tabular format
    - Also the graph can have color coding or color gradient in it and they can also be black and white While table will never have color coding. 
    - In some cased horizontal bar charts can have same grid like structure as tables, but they can have color coding, values at x and y axis. You should not confuse between those type of bar charts and tables
    - Bar graphs can be solid colored, patterned, hatched, or outlined and uses visual length/height of bars to represent data values
    - Some graphs such as percentage_stacked_bar_chart (definition: {percentage_stacked_bar_chart}), bar_charts (definition:{bar_chart}) and histograms (definition:{histogram}), can look similar to tables. You need to differentiate between tables and these charts and identify only tables. 
    
    Please follow the below instructions for the same: Analyze the input image and text from PDF image from top to bottom and identify any tabular data structures. Find out all the tabular structures available in the input image. Analyse the input image carefully and do not miss any tables. There can be tables any where on the page, and can be surrounded by the text as well. Look carefully in all the parts of the page and identify all the tables. 
    - You need to be precise in the number of rows and columns available in the table. 
    - Do not over identify or under identify. This is a very critical piece of information, hence do not make errors in this.  Do not consider any additional row or column as a part of the table. Count the number or rows and columns very accurately.
    - Look for ALL distinct tabular structures, even if they appear similar and Do NOT merge similar-looking tables into one. 
    - Do not merge two continuous tables into one table, all different table's metadata should be distinct.

    The available Text content extracted from a PDF document is:
    {text_content}
    
    For each table found, provide:

    
    1. **Table Structure**: Rows (number of rows), columns (number of columns), headers. Check the structure of table very carefully
    Look for patterns that indicate tables and provide configuration for tabula extraction.
    2. Number of estimated rows would go as "rows": <estimated_rows>, Number of estimated columns will go as "columns": <estimated_columns>
    3. Number of tables found in the input image. Returned value will go in "tables_found": <count>
    4. "method" will take "lattice" or "stream" value: 
    -- lattice is used to parse tables that have demarcated lines between cells,
    -- stream is used to parse tables that have whitespaces between cells to simulate a table structure. Hence assign the method value by carefully analysing the table
    5. Do not consider the nearby text in the table as part of the table. A table is a single unit of cell values grouped in one place. Any nearby text is not a part of the table
    6. Check for the header carefully, if there are any headers available. If there is a header return "True", if there is no header, return "False"
    
    Format as JSON:
    {{
        "page_number": {page_number},
        "analysis_method": "text-based",
        "tables_found": <count>,
        "tables": [
            {{
                "table_id": 1,
                "title": "Detected table description",
                "structure": {{
                    "rows": <estimated_rows>,
                    "columns": <estimated_columns>,
                    "has_header": true/false
                }},
                "method": "lattice" or "stream",
                "confidence": 0.0-1.0
            }}
        ]
    }}
    
    If no tables detected:
    {{
        "page_number": {page_number},
        "analysis_method": "text-based",
        "tables_found": 0,
        "tables": [],
        "notes": "No tabular structures detected in text"
    }}
    """



table_validation_prompt = """ You are a document analysis assistant and table extraction validator, your role is to analyse the tables in the document and correct them. Your task is to compare an extracted table with the original PDF page and provide a corrected version. 
 
**EXTRACTED TABLE FROM: {source_info}**

{table_text}

**VALIDATION INSTRUCTIONS:**

1. **CAREFULLY EXAMINE** the PDF page and compare it with the extracted table above
2. **IDENTIFY ISSUES** in the extracted data:
   - Missing rows or columns: There can be rows or columns which are missing from the table, you need to restore them
   - Incorrect cell values could be numbers, text, dates: There can be missing cell values, you need to restore them
   - Wrong table structure or layout: The layout of the table can be multi-row spanning or multi-column spanning. You need to identify the correct layout of the original table and correct the layout of extracted table.
   - Merged cells not handled properly: Some cells could be merged to other cells, hence the cell values need to be placed appropriately  
   - Headers/footers missing or misplaced: You need to identify if there is a missing header or header merged with the table and fix it in the extracted table.
   - Row alignment issues: You need to identify if the multi-row spanning cells are aligned correctly with the other rows. You need to fix this in the extracted table
   - OCR errors in text recognition: You need to identify if there are any OCR errors in the extracted text. Fix them in the extracted table
   - Multi-row headers: You need to identify if there are headers which contain text in multiple rows. You need to fix the headers which takes multiple rows and correct the extracted table. 
   - Remove any extra row or column in the extracted table, if it is not available in the original table in the image
   - Do not add any row cell data into header if it is not available in the original table. Carefully check that cells within rows should remain in rows only not in the headers
   - If there is a row or column misplaced in the extracted table, correct it after carefully comparing with the original table in the image
   - Capture the number of rows and columns correctly from the original table. Do not merge different columns into single column. 
   - If there is a complex table, then carefully analyse the number of rows and columns and extract table correctly. 

3. **PROVIDE CORRECTIONS** for both data accuracy and table layout

**RESPOND IN THIS EXACT JSON FORMAT if the table if a simple table WITHOUT any multi-row or multi-column spanning rows:**
```json
{{
    
    "corrected_table": {{
        "headers": ["Column1", "Column2", "Column3"],
        "rows": [
            ["Row1Data1", "Row1Data2", "Row1Data3"],
            ["Row2Data1", "Row2Data2", "Row2Data3"]
        ]
    }}
    
}}
```
If the table is a complex table then below example format can be used for generating JSON. 

```json
{{
     "corrected_table": {{
       "headers": ["Item Description", "Qty", "Unit Price", "Line Total"],
      "rows": [
                ["Professional Consulting", "40", "$150.00", "$6,000.00"],
                 ["Software License (Annual)", "1", "$2,500.00", "$2,500.00"],
               ["", "", "Subtotal:", "$8,500.00"],
                ["", "", "Tax (8.5%):", "$722.50"],
                 ["", "", "TOTAL DUE:", "$9,222.50"]
         ]
     }}
 }}
 


```

Another example of complex table json:

```json
{{
"corrected_table":{{
  "headers": ["Item Description", "Qty", "Unit Price", "Line Total"],
  "rows": [
    ["Professional Consulting", "40", "$150.00", "$6,000.00"],
    ["", "", "Subtotal:", "$8,500.00"],
    ["", "", "Tax (8.5%):", "$722.50"],
    ["", "", "TOTAL DUE:", "$9,222.50"]
  ],
  "has_totals": true,
  "calculation_rows": [3, 4, 5]
}}
}}

```

**IMPORTANT GUIDELINES:**
- Focus on both text accuracy AND proper table structure
- Pay attention to numerical precision, date formats, and text accuracy
- Layout of the table should be similar to the original table available in the document image
- Do not add any extra row, header or column if not available in the original table in image
"""

translation_prompt = """
# Translation Task

## Your Role
You are a professional translator specializing in translating documents from their original language to {output_language} while preserving formatting, meaning, and technical terminology.

## Instructions:
1. Maintain the original document structure and formatting
2. Preserve tables, bullet points, and numbering
3. Keep technical terms accurate and contextually appropriate
4. Ensure any references to graphs and charts are described correctly

## Text Content to Translate:
{text_content}

## Output Format
Provide the complete translation in {output_language} while maintaining the document's original structure.
Do not include any explanations or notes about the translation - only return the translated text.
"""
