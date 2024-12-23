import pandas as pd
import os

# Helper function to clean and preprocess text
def preprocess_text(intput_file, output_file):
    """
    Reads and preprocesses text from a file.

    Args:
        file (str): Path to the text file.

    Returns:
        list: A list of cleaned lines from the file.
    """
    # Open the file and read its content
    with open(intput_file, 'r', encoding='utf-8') as viet_file:
        lines = viet_file.readlines()
    # Remove BOM characters and strip whitespace
    cleaned_lines = [line.replace('\ufeff', '').strip() for line in lines if line.strip()]
    # Save cleaned lines to the output file if specified
    output_folder = 'cleaned_data'
    os.makedirs(output_folder, exist_ok=True)
    path_file = os.path.join(output_folder, output_file)
    print(path_file)
    with open(path_file, 'w', encoding='utf-8') as output:
        output.write('\n'.join(cleaned_lines))
    print(f"Cleaned lines have been saved to: {output_file}")
    return cleaned_lines

# Function to save data to an Excel file
def save_to_excel(data, output_file):
    """
    Save a list of tuples to an Excel file.

    Args:
        data (list of tuples): The data to be saved, where each tuple contains
                               (Vietnamese, Chinese, Score).
        output_file (str): Path to the Excel file to save the data.
    """
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data, columns=["Vietnamese", "Chinese", "Score"])
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Excel file has been saved at: {output_file}")
