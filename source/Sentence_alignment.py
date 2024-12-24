from contextlib import nullcontext
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torch.amp import autocast

class SentenceAlignment:
    def __init__(self, model_name="sentence-transformers/LaBSE", device=None):
        """
        Initialize the SentenceAlignment class with a pre-trained model.
        Default model: LaBSE (Language-agnostic BERT Sentence Embedding).

        Args:
            model_name (str): Name of the pre-trained model.
            device (str, optional): Device to run the computations ("cuda" or "cpu"). 
                                    Auto-detects if not specified.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # Default to GPU if available
        print(f"Using {self.device}...")
        self.model.to(self.device)  # Move model to the specified device

    def encode_sentences(self, sentences, batch_size, pbar=None):
        """
        Encode a list of sentences into vector embeddings.

        Args:
            sentences (list): List of input sentences.
            batch_size (int): Number of sentences to process in each batch.
            pbar (tqdm): Optional tqdm progress bar object.

        Returns:
            torch.Tensor: Encoded embeddings as a PyTorch tensor on GPU or CPU.
        """
        embeddings = []
        with torch.no_grad():  # Disable gradient computation for inference
            # Use autocast for GPU, or nullcontext for CPU
            autocast_context = autocast(device_type=self.device) if self.device == "cuda" else nullcontext()
            with autocast_context:
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i:i + batch_size]
                    # Tokenize the sentences
                    inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    # Forward pass through the model
                    outputs = self.model(**inputs)
                    # Extract CLS token embeddings
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                    # Normalize embeddings
                    batch_embeddings = batch_embeddings / torch.norm(batch_embeddings, dim=1, keepdim=True)
                    embeddings.append(batch_embeddings)
                    if pbar:
                        pbar.update(1)  # Update progress bar
        return torch.cat(embeddings, dim=0)  # Combine all embeddings into one tensor

    def align_sentences(self, source_sentences, target_sentences, batch_size=32, threshold=0.5):
        """
        Align sentences from the source and target lists using the Hungarian Algorithm.

        Args:
            source_sentences (list): List of source sentences.
            target_sentences (list): List of target sentences.
            batch_size (int): Number of sentences to process per batch during encoding.
            threshold (float): Minimum similarity score to consider a pair as aligned.

        Returns:
            list: A list of tuples (source_sentence, target_sentence, similarity_score).
        """
        # Initialize progress bar
        with tqdm(
            total=(len(source_sentences) + batch_size - 1) // batch_size * 2,
            desc="Processing Sentences",
            unit="batchs"
        ) as pbar:
            # Encode source and target sentences
            source_embeddings = self.encode_sentences(source_sentences, batch_size, pbar=pbar)
            target_embeddings = self.encode_sentences(target_sentences, batch_size, pbar=pbar)

        # Variables for alignment
        aligned_pairs = [None] * len(source_sentences)
        used_rows = set()  # Keep track of aligned rows
        used_cols = set()  # Keep track of aligned columns

        # Chunk size for memory efficiency
        chunk_size = 2000
        with tqdm(
            total=((len(source_sentences) + chunk_size - 1) // chunk_size) ** 2,
            desc="Processing Similarities",
            unit="batchs"
        ) as pbar1:
            for i in range(0, len(source_sentences), chunk_size):
                source_chunk = source_embeddings[i:i + chunk_size]
                for j in range(0, len(target_sentences), chunk_size):
                    target_chunk = target_embeddings[j:j + chunk_size]

                    # Compute cosine similarity matrix
                    sim_matrix = torch.mm(source_chunk, target_chunk.T)
                    # Convert similarity scores to costs
                    cost_matrix = -sim_matrix.cpu().numpy()
                    # Solve assignment using the Hungarian algorithm
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    # Process aligned pairs
                    for row, col in zip(row_indices, col_indices):
                        global_row = i + row
                        global_col = j + col
                        if global_row in used_rows or global_col in used_cols:
                            continue

                        score = sim_matrix[row, col].item()
                        if score >= threshold or (aligned_pairs[global_row] and score > aligned_pairs[global_row][2]):
                            aligned_pairs[global_row] = (source_sentences[global_row], target_sentences[global_col], score)
                            used_rows.add(global_row)
                            used_cols.add(global_col)
                    if pbar1:
                        pbar1.update(1)

        # Handle unused rows
        print("Handling the remainder....")
        unused_rows = list(set(range(len(source_sentences))) - used_rows)
        if unused_rows:
            unused_source = source_embeddings[unused_rows]
            unused_target = target_embeddings[unused_rows]

            sim_matrix = torch.mm(unused_source, unused_target.T)
            cost_matrix = -sim_matrix.cpu().numpy()
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_indices, col_indices):
                score = sim_matrix[i, j].item()
                if score >= threshold:
                    aligned_pairs[unused_rows[i]] = (
                        source_sentences[unused_rows[i]],
                        target_sentences[unused_rows[j]],
                        score,
                    )
                else:
                    aligned_pairs[unused_rows[i]] = (source_sentences[unused_rows[i]], "Untranslated", 0)

        return aligned_pairs
    