import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm

class SentenceAlignment:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        """
        Initialize the SentenceAlignment class with a pre-trained model.
        The default model used is LaBSE (Language-agnostic BERT Sentence Embedding).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def encode_sentences(self, sentences, batch_size=32):
        """
        Encode a list of sentences into vector embeddings.

        Args:
            sentences (list): List of input sentences.
            batch_size (int): Number of sentences to process in each batch.

        Returns:
            np.ndarray: Encoded embeddings as a NumPy array.
        """
        embeddings = []
        with torch.no_grad():  # Disable gradient calculation for inference
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                # Extract CLS token embeddings for each sentence
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # Normalize the embeddings
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    def align_sentences(self, source_sentences, target_sentences, threshold=0.0):
        """
        Align sentences from the source and target lists using the Hungarian Algorithm.

        Args:
            source_sentences (list): List of source sentences.
            target_sentences (list): List of target sentences.
            threshold (float): Minimum similarity score to consider as aligned.

        Returns:
            list: A list of tuples (source_sentence, target_sentence, similarity_score).
        """
        # Encode sentences into embeddings
        source_embeddings = self.encode_sentences(source_sentences)
        target_embeddings = self.encode_sentences(target_sentences)

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(source_embeddings, target_embeddings)

        # Convert similarity scores to costs (for minimization in the Hungarian Algorithm)
        cost_matrix = -sim_matrix

        # Solve the optimal assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Build the list of aligned sentence pairs
        aligned_pairs = []
        for i, j in zip(row_indices, col_indices):
            score = sim_matrix[i][j]
            if score >= threshold:
                aligned_pairs.append((source_sentences[i], target_sentences[j], score))
            else:
                aligned_pairs.append((source_sentences[i], "Untranslated", 0))

        # Handle unmatched source sentences
        untranslated_indices = set(range(len(source_sentences))) - set(row_indices)
        for i in untranslated_indices:
            aligned_pairs.append((source_sentences[i], "Untranslated", 0))

        return aligned_pairs