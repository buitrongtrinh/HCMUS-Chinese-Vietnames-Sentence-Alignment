from source.Sentence_alignment import SentenceAlignment
from source.preprocessing import preprocess_text, save_to_excel
import sys
import argparse

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Sentence Alignment Model")
    parser.add_argument("--viet_file", type=str, required=True, help="Path to the Vietnamese dataset file")
    parser.add_argument("--hoa_file", type=str, required=True, help="Path to the Chinese dataset file")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum similarity threshold for alignment")
    args = parser.parse_args()

    # Initialize the Sentence Alignment Model
    print("Initialize the Sentence Alignment Model...")
    aligner = SentenceAlignment()

    # Preprocess the two datasets
    print("Preprocessing Vietnamese dataset...")
    viet_cleaned = preprocess_text(args.viet_file)  # Clean Vietnamese sentences

    print("Preprocessing Chinese dataset...")
    hoa_cleaned = preprocess_text(args.hoa_file)   # Clean Chinese sentences

    # Perform sentence alignment
    print("Encoding and aligning sentences...")
    all_batch_size = {'cpu': 32, 'cuda': 256}
    aligned_results = aligner.align_sentences(viet_cleaned, hoa_cleaned, all_batch_size[aligner.device], args.threshold)
    print("Alignment completed!") if aligned_results else sys.exit("Alignment failed!!!")

    # Save the alignment results to an Excel file
    excel_file = f"result/alignment_results.xlsx"
    save_to_excel(aligned_results, excel_file)
    print(f"Results saved to {excel_file}")
