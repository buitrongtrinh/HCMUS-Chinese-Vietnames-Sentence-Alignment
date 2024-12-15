import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
from preprocessing import viet_cleaned,hoa_cleaned

# Load pre-trained LaBSE model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

def encode_sentences(sentences, tokenizer, model, batch_size=32):
    """
    Mã hóa danh sách câu thành vector embedding.
    """
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
            # Normalization
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def optimal_alignment(source_sentences, target_sentences, threshold):
    """
    Sử dụng thuật toán Hungarian để tối ưu căn chỉnh giữa các câu nguồn và câu đích.
    """
    # Tính embedding cho tất cả các câu
    source_embeddings = encode_sentences(source_sentences, tokenizer, model)
    target_embeddings = encode_sentences(target_sentences, tokenizer, model)

    # Tính ma trận độ tương đồng
    sim_matrix = cosine_similarity(source_embeddings, target_embeddings)

    # Chuyển độ tương đồng sang chi phí (Hungarian yêu cầu bài toán tối thiểu hóa)
    cost_matrix = -sim_matrix

    # Giải bài toán phân bổ tối ưu
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Xây dựng danh sách căn chỉnh
    aligned_pairs = []
    for i, j in zip(row_indices, col_indices):
        score = sim_matrix[i][j]
        if score >= threshold:
            aligned_pairs.append((source_sentences[i], target_sentences[j], score))
        else:
            aligned_pairs.append((source_sentences[i], "Untranslated", 0))

    # Xử lý các câu nguồn chưa được căn chỉnh (nếu số câu nguồn nhiều hơn câu đích)
    untranslated_indices = set(range(len(source_sentences))) - set(row_indices)
    for i in untranslated_indices:
        aligned_pairs.append((source_sentences[i], "Untranslated", 0))

    return aligned_pairs

# Chạy pipeline
threshold = 0.0
aligned_results = optimal_alignment(viet_cleaned, hoa_cleaned, threshold)

# Lưu kết quả căn chỉnh
output_file_path = "output_aligned.txt"
with open(output_file_path, "w", encoding="utf-8") as file:
    for result in aligned_results:
        viet, hoa, score = result
        file.write(f"Tiếng Việt: {viet}\nTiếng Trung: {hoa}\nScore: {score:.4f}\n{'-' * 40}\n")

print(f"Kết quả căn chỉnh đã được lưu tại {output_file_path}")
