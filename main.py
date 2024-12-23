from source.Sentence_alignment import SentenceAlignment
from source.preprocessing import preprocess_text, save_to_excel

if __name__ == "__main__":
    # Initialize the SentenceAlignment class
    aligner = SentenceAlignment()

    # Preprocess the two datasets
    viet_cleaned = preprocess_text('data/viet.txt', 'cleaned_viet.txt')  # Clean Vietnamese sentences
    hoa_cleaned = preprocess_text('data/hoa.txt', 'cleaned_hoa.txt')   # Clean Chinese sentences

    # Perform sentence alignment
    threshold = 0.0  # Minimum similarity threshold for alignment
    aligned_results = aligner.align_sentences(viet_cleaned, hoa_cleaned, threshold)

    # Save the alignment results to an Excel file
    excel_file = "alignment_results.xlsx"
    save_to_excel(aligned_results, excel_file)
