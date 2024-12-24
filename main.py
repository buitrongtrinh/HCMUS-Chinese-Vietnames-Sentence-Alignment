from source.Sentence_alignment import SentenceAlignment
from source.preprocessing import preprocess_text, save_to_excel
import sys

if __name__ == "__main__":
    # Initialize the Sentence Alignment Model
    print("Initialize the Sentence Alignment Model...")
    aligner = SentenceAlignment()

    # Preprocess the two datasets
    print("Preprocessing Vietnamese dataset...")
    viet_cleaned = preprocess_text("data/viet.txt")  # Clean Vietnamese sentences

    print("Preprocessing Chinese dataset...")
    hoa_cleaned = preprocess_text("data/hoa.txt")   # Clean Chinese sentences

    # Perform sentence alignment
    print("Encoding and aligning sentences...")
    threshold = 0.7  # Minimum similarity threshold for alignment
    all_batch_size = {'cpu':32, 'cuda': 256}
    aligned_results = aligner.align_sentences(viet_cleaned, hoa_cleaned, all_batch_size[aligner.device], threshold)
    print("Alignment completed!") if aligned_results else sys.exit("Alignment failed!!!")

    # Save the alignment results to an Excel file
    excel_file = f"result/kalignment_results.xlsx"
    save_to_excel(aligned_results, excel_file)
