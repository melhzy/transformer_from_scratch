"""
Module 1: Token Embeddings
===========================

Token embeddings convert discrete tokens (word indices) into continuous vector
representations that the Transformer can process. This is the first step in
making text understandable to neural networks.

Key Concepts:
-------------
1. **Vocabulary**: The set of all possible tokens (words, subwords, characters)
2. **Embedding Dimension (d_model)**: Size of the vector representation
3. **Embedding Matrix**: Learnable lookup table of shape [vocab_size, d_model]
4. **Scaling**: Embeddings are scaled by sqrt(d_model) as per the paper

Why Embeddings?
---------------
- Neural networks can't process discrete symbols directly
- Embeddings capture semantic relationships (similar words → similar vectors)
- They're learned during training to optimize the task

Mathematical Formulation:
-------------------------
Given a token index i ∈ {0, 1, ..., vocab_size-1}:
    embedding(i) = E[i] * sqrt(d_model)

where E is the embedding matrix and the scaling factor helps balance
the magnitude with positional encodings.

Reference: "Attention Is All You Need", Section 3.4
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer
    
    Converts token indices to dense vector representations with learned embeddings.
    Includes the sqrt(d_model) scaling factor used in the original Transformer paper.
    
    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Dimension of the embedding vectors (typically 512)
        padding_idx (Optional[int]): If specified, entries at padding_idx do not
                                      contribute to the gradient (default: None)
    
    Shape:
        - Input: (batch_size, seq_len) - Long tensor of token indices
        - Output: (batch_size, seq_len, d_model) - Embedded sequences
    
    Examples:
        >>> vocab_size = 10000
        >>> d_model = 512
        >>> embedding = TokenEmbedding(vocab_size, d_model)
        >>> tokens = torch.randint(0, vocab_size, (2, 10))  # batch_size=2, seq_len=10
        >>> embedded = embedding(tokens)
        >>> print(embedded.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None
    ):
        super(TokenEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Embedding lookup table - this is learned during training
        # Shape: [vocab_size, d_model]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        
        # Scaling factor: sqrt(d_model)
        # This helps balance embedding magnitudes with positional encodings
        self.scale = math.sqrt(d_model)
        
        # Initialize embeddings with normal distribution
        # This is a common practice for better training dynamics
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding weights with Xavier/Glorot uniform initialization."""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.padding_idx is not None:
            # Zero out padding embeddings
            nn.init.constant_(self.embedding.weight[self.padding_idx], 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Convert token indices to scaled embeddings.
        
        Args:
            x (torch.Tensor): Token indices of shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Scaled embeddings of shape (batch_size, seq_len, d_model)
        """
        # Look up embeddings and scale by sqrt(d_model)
        # The scaling is crucial: it ensures embeddings and positional encodings
        # have similar magnitudes when added together
        return self.embedding(x) * self.scale
    
    def get_embedding_weight(self) -> torch.Tensor:
        """
        Returns the underlying embedding weight matrix.
        Useful for weight tying with output projection layer.
        
        Returns:
            torch.Tensor: Embedding weights of shape (vocab_size, d_model)
        """
        return self.embedding.weight


# ============================================================================
# Utility Functions for Understanding and Visualization
# ============================================================================

def compute_embedding_similarity(
    embedding_layer: TokenEmbedding,
    token_idx1: int,
    token_idx2: int
) -> float:
    """
    Compute cosine similarity between two token embeddings.
    
    This helps understand which tokens the model considers similar.
    Higher similarity (closer to 1) means more similar representations.
    
    Args:
        embedding_layer: The TokenEmbedding instance
        token_idx1: Index of first token
        token_idx2: Index of second token
    
    Returns:
        float: Cosine similarity between -1 and 1
    """
    with torch.no_grad():
        # Get embeddings for both tokens
        emb1 = embedding_layer(torch.tensor([[token_idx1]]))  # [1, 1, d_model]
        emb2 = embedding_layer(torch.tensor([[token_idx2]]))  # [1, 1, d_model]
        
        # Compute cosine similarity
        emb1 = emb1.squeeze()  # [d_model]
        emb2 = emb2.squeeze()  # [d_model]
        
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        )
        
        return similarity.item()


def get_nearest_neighbors(
    embedding_layer: TokenEmbedding,
    token_idx: int,
    top_k: int = 5
) -> list:
    """
    Find the k most similar tokens to a given token based on embedding similarity.
    
    This is useful for understanding what the model has learned about relationships
    between tokens (e.g., synonyms, related concepts).
    
    Args:
        embedding_layer: The TokenEmbedding instance
        token_idx: Index of the query token
        top_k: Number of nearest neighbors to return
    
    Returns:
        list: List of (token_idx, similarity_score) tuples
    """
    with torch.no_grad():
        # Get embedding for query token
        query_emb = embedding_layer(torch.tensor([[token_idx]]))  # [1, 1, d_model]
        query_emb = query_emb.squeeze()  # [d_model]
        
        # Get all embeddings
        all_embs = embedding_layer.get_embedding_weight()  # [vocab_size, d_model]
        
        # Compute similarities with all tokens
        similarities = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0),
            all_embs,
            dim=1
        )
        
        # Get top-k (excluding the query token itself)
        top_similarities, top_indices = torch.topk(similarities, k=top_k + 1)
        
        # Filter out the query token and return results
        results = []
        for idx, sim in zip(top_indices.tolist(), top_similarities.tolist()):
            if idx != token_idx:
                results.append((idx, sim))
            if len(results) == top_k:
                break
        
        return results


if __name__ == "__main__":
    # Example usage and demonstration
    print("=" * 70)
    print("Token Embedding Module - Example Usage")
    print("=" * 70)
    
    # Create a small vocabulary for demonstration
    vocab_size = 1000
    d_model = 512
    batch_size = 2
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension (d_model): {d_model}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Initialize embedding layer
    embedding = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    
    # Create sample token indices
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    print(f"\nInput tokens shape: {tokens.shape}")
    print(f"Sample tokens: {tokens[0, :5].tolist()}")
    
    # Get embeddings
    embedded = embedding(tokens)
    print(f"\nOutput embeddings shape: {embedded.shape}")
    print(f"Embedding scaling factor: {embedding.scale:.2f}")
    
    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embedded.mean().item():.4f}")
    print(f"  Std: {embedded.std().item():.4f}")
    print(f"  Min: {embedded.min().item():.4f}")
    print(f"  Max: {embedded.max().item():.4f}")
    
    # Demonstrate similarity computation
    token1, token2 = 42, 123
    similarity = compute_embedding_similarity(embedding, token1, token2)
    print(f"\nCosine similarity between token {token1} and {token2}: {similarity:.4f}")
    
    # Find nearest neighbors
    print(f"\nTop 5 nearest neighbors to token {token1}:")
    neighbors = get_nearest_neighbors(embedding, token1, top_k=5)
    for i, (neighbor_idx, sim) in enumerate(neighbors, 1):
        print(f"  {i}. Token {neighbor_idx}: similarity = {sim:.4f}")
    
    print("\n" + "=" * 70)
    print("Module 1 Complete! Next: Positional Encodings")
    print("=" * 70)
