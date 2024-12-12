import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from preprocessing import hoa_cleaned, viet_cleaned

# Load pre-trained LaBSE model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

def calculate_labse_score(sent1, sent2):
    inputs1 = tokenizer(sent1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(sent2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        embedding1 = model(**inputs1).last_hidden_state.mean(dim=1).cpu().numpy()
        embedding2 = model(**inputs2).last_hidden_state.mean(dim=1).cpu().numpy()

    return cosine_similarity(embedding1, embedding2)[0][0]

def gale_church_alignment(source_sentences, target_sentences):
    n = len(source_sentences)
    m = len(target_sentences)
    cost_matrix = np.zeros((n + 1, m + 1))
    cost_matrix[:, 0] = np.inf
    cost_matrix[0, :] = np.inf
    cost_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            source_len = len(source_sentences[i - 1])
            target_len = len(target_sentences[j - 1])
            length_diff = abs(source_len - target_len)
            cost_matrix[i, j] = cost_matrix[i - 1, j - 1] + length_diff

    aligned_pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        aligned_pairs.append((source_sentences[i - 1], target_sentences[j - 1]))
        i -= 1
        j -= 1

    return list(reversed(aligned_pairs))

def greedy_alignment(source_sentences, target_sentences):
    aligned_pairs = []
    used_target_indices = set()

    for src_sentence in source_sentences:
        best_score = -1
        best_match_idx = -1

        for tgt_idx, tgt_sentence in enumerate(target_sentences):
            if tgt_idx in used_target_indices:
                continue  # Skip already matched sentences

            score = calculate_labse_score(src_sentence, tgt_sentence)
            if score > best_score:
                best_score = score
                best_match_idx = tgt_idx

        if best_match_idx != -1:
            aligned_pairs.append((src_sentence, target_sentences[best_match_idx]))
            used_target_indices.add(best_match_idx)
        else:
            aligned_pairs.append((src_sentence, "Untranslated"))

    for tgt_idx, tgt_sentence in enumerate(target_sentences):
        if tgt_idx not in used_target_indices:
            aligned_pairs.append(("Untranslated", tgt_sentence))

    return aligned_pairs

def chunk_and_align(source_sentences, target_sentences, threshold=500):
    if len(source_sentences) <= threshold and len(target_sentences) <= threshold:
        return gale_church_alignment(source_sentences, target_sentences)

    mid_index_source = len(source_sentences) // 2
    mid_index_target = len(target_sentences) // 2

    left_chunk = chunk_and_align(source_sentences[:mid_index_source], target_sentences[:mid_index_target], threshold)
    right_chunk = chunk_and_align(source_sentences[mid_index_source:], target_sentences[mid_index_target:], threshold)

    return left_chunk + right_chunk

def find_path(aligned_pairs):
    """
    Evaluate alignment results and refine path.
    """
    scores = [calculate_labse_score(pair[0], pair[1]) if pair[1] != "Untranslated" else 0 for pair in aligned_pairs]
    return aligned_pairs  # Placeholder for more complex logic if needed

# Full pipeline
def align_sentences_pipeline(source_sentences, target_sentences, threshold=500):
    aligned_chunks = chunk_and_align(source_sentences, target_sentences, threshold)
    aligned_results = find_path(aligned_chunks)
    return aligned_results

aligned_results = align_sentences_pipeline(viet_cleaned, hoa_cleaned, threshold=200)

# Save results
output_file_path = "output_aligned.txt"
with open(output_file_path, "w", encoding="utf-8") as file:
    for viet, hoa in aligned_results:
        file.write(f"Tiếng Việt: {viet}\nTiếng Trung: {hoa}\n{'-' * 40}\n")

print(f"Aligned results saved to {output_file_path}")
