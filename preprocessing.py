# Helper function to clean and preprocess text
def preprocess_text(lines):
    # Remove BOM character and strip whitespace
    cleaned_lines = [line.replace('\ufeff', '').strip() for line in lines if line.strip()]
    return cleaned_lines

# Read content of the two files
with open('data/viet.txt', 'r', encoding='utf-8') as viet_file:
    viet_content = viet_file.readlines()

with open('data/hoa.txt', 'r', encoding='utf-8') as hoa_file:
    hoa_content = hoa_file.readlines()

# Preprocess both files
viet_cleaned = preprocess_text(viet_content)
hoa_cleaned = preprocess_text(hoa_content)

# Count sentences in both files
viet_sentence_count = len(viet_cleaned)
hoa_sentence_count = len(hoa_cleaned)


