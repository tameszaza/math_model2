import os
import nbformat
from nbconvert import PythonExporter
import re

def clean_code(code):
    """
    Remove comments that start with # and reduce multiple newlines to a single newline.
    """
    # Remove comments
    lines = code.splitlines()
    filtered_lines = [line for line in lines if not re.match(r'^\s*#', line)]
    
    # Join the lines and reduce multiple newlines to a single newline
    cleaned_code = '\n'.join(filtered_lines)
    cleaned_code = re.sub(r'\n\s*\n', '\n', cleaned_code)
    
    return cleaned_code

def convert_ipynb_to_py(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return
    
    # Get a list of all .ipynb files in the folder
    ipynb_files = [f for f in os.listdir(folder_path) if f.endswith('.ipynb')]
    
    if not ipynb_files:
        print("No .ipynb files found in the folder.")
        return
    
    # Initialize the PythonExporter
    exporter = PythonExporter()
    
    for ipynb_file in ipynb_files:
        ipynb_path = os.path.join(folder_path, ipynb_file)
        py_filename = os.path.splitext(ipynb_file)[0] + '.py'
        py_path = os.path.join(folder_path, py_filename)
        
        # Read the notebook content
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Convert the notebook to a Python script
        (body, _) = exporter.from_notebook_node(notebook)
        
        # Clean the code
        clean_code_result = clean_code(body)
        
        # Write the cleaned Python script to a file
        with open(py_path, 'w', encoding='utf-8') as f:
            f.write(clean_code_result)
        
        print(f"Converted {ipynb_file} to {py_filename} without comments and with reduced newlines.")
        



# Example usage
convert_ipynb_to_py('jupyter')
