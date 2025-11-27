"""
Module 3: Scaled Dot-Product Attention
=======================================

Attention is the core mechanism that allows the Transformer to focus on different
parts of the input sequence when processing each token. The "Attention Is All You
Need" paper introduces Scaled Dot-Product Attention as the fundamental building block.

Key Concepts:
-------------
1. **Query (Q)**: What we're looking for (current token representation)
2. **Key (K)**: What we can match against (all token representations)
3. **Value (V)**: The actual information we want to retrieve
4. **Attention Scores**: How much to focus on each position
5. **Scaling Factor**: Prevents softmax saturation for large d_k

The Intuition:
--------------
Think of attention as a soft lookup table:
- Query: Your search query
- Keys: Index entries in a database
- Values: The actual data stored
- Attention weights: How relevant each entry is to your query

The attention mechanism computes a weighted sum of values, where weights
are determined by the similarity between the query and keys.

Mathematical Formulation:
-------------------------
    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

where:
    - Q·K^T: Compute similarity scores (dot product)
    - sqrt(d_k): Scaling factor (prevents large values)
    - softmax: Normalize scores to probabilities
    - ·V: Weighted sum of values

Why Scaling?
------------
Without scaling, dot products grow large in magnitude for high dimensions,
pushing softmax into regions with tiny gradients. Scaling by sqrt(d_k)
counteracts this, keeping gradients healthy.

Reference: "Attention Is All You Need", Section 3.2.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Implements the core attention mechanism from "Attention Is All You Need".
    Computes attention weights based on query-key similarity and applies them
    to values to produce context-aware representations.
    
    Args:
        dropout (float): Dropout probability applied to attention weights (default: 0.1)
        scale (Optional[float]): Custom scaling factor. If None, uses sqrt(d_k)
    
    Shape:
        - Q: (batch_size, n_heads, seq_len_q, d_k)
        - K: (batch_size, n_heads, seq_len_k, d_k)
        - V: (batch_size, n_heads, seq_len_v, d_v) where seq_len_v == seq_len_k
        - mask: (batch_size, 1, seq_len_q, seq_len_k) or broadcastable
        - output: (batch_size, n_heads, seq_len_q, d_v)
        - attention_weights: (batch_size, n_heads, seq_len_q, seq_len_k)
    
    Examples:
        >>> batch_size, n_heads, seq_len, d_k = 2, 8, 10, 64
        >>> attention = ScaledDotProductAttention(dropout=0.1)
        >>> Q = torch.randn(batch_size, n_heads, seq_len, d_k)
        >>> K = torch.randn(batch_size, n_heads, seq_len, d_k)
        >>> V = torch.randn(batch_size, n_heads, seq_len, d_k)
        >>> output, weights = attention(Q, K, V)
        >>> print(output.shape)  # torch.Size([2, 8, 10, 64])
    """
    
    def __init__(self, dropout: float = 0.1, scale: Optional[float] = None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k)
            key (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k)
            value (torch.Tensor): Value tensor of shape (..., seq_len_v, d_v)
            mask (Optional[torch.Tensor]): Mask tensor to prevent attention to certain
                                           positions. Shape: (..., seq_len_q, seq_len_k)
                                           Use 0 for positions to mask, 1 for valid positions.
        
        Returns:
            tuple: (output, attention_weights)
                - output: Attention output of shape (..., seq_len_q, d_v)
                - attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
        """
        # Get dimension of keys (d_k)
        d_k = query.size(-1)
        
        # Compute scaling factor
        if self.scale is None:
            scale = math.sqrt(d_k)
        else:
            scale = self.scale
        
        # Step 1: Compute attention scores (Q·K^T)
        # query: (..., seq_len_q, d_k)
        # key^T: (..., d_k, seq_len_k)
        # scores: (..., seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale the scores
        scores = scores / scale
        
        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Where mask is 0, replace scores with large negative value
            # This makes softmax output ~0 for masked positions
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights
        # Softmax is applied over the last dimension (seq_len_k)
        # This ensures attention weights for each query sum to 1
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Apply dropout to attention weights
        # This is a form of regularization - randomly zero out some attention connections
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Apply attention weights to values
        # attention_weights: (..., seq_len_q, seq_len_k)
        # value: (..., seq_len_k, d_v)
        # output: (..., seq_len_q, d_v)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


# ============================================================================
# Utility Functions for Understanding and Visualization
# ============================================================================

def create_padding_mask(seq_len: int, padding_positions: list) -> torch.Tensor:
    """
    Create a padding mask for attention.
    
    Padding tokens should not contribute to attention. This function creates
    a mask where padding positions are marked as 0 (masked) and valid positions
    as 1 (not masked).
    
    Args:
        seq_len (int): Length of the sequence
        padding_positions (list): List of positions to mask (0-indexed)
    
    Returns:
        torch.Tensor: Mask of shape (1, 1, 1, seq_len) for broadcasting
    
    Example:
        >>> mask = create_padding_mask(10, [8, 9])  # Last 2 positions are padding
        >>> print(mask.squeeze())  # [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    """
    mask = torch.ones(1, 1, 1, seq_len)
    for pos in padding_positions:
        mask[0, 0, 0, pos] = 0
    return mask


def create_look_ahead_mask(seq_len: int) -> torch.Tensor:
    """
    Create a look-ahead (causal) mask for decoder self-attention.
    
    In the decoder, each position can only attend to earlier positions
    and itself. This prevents the model from "cheating" by looking at
    future tokens during training.
    
    Args:
        seq_len (int): Length of the sequence
    
    Returns:
        torch.Tensor: Lower triangular mask of shape (1, 1, seq_len, seq_len)
    
    Example:
        >>> mask = create_look_ahead_mask(4)
        >>> print(mask.squeeze())
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
    """
    # Create lower triangular matrix (1s below and on diagonal, 0s above)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


def visualize_attention_pattern(
    attention_weights: torch.Tensor,
    query_idx: int = 0,
    head_idx: int = 0
) -> torch.Tensor:
    """
    Extract attention pattern for a specific query position and attention head.
    
    This helps visualize which positions a given token attends to.
    
    Args:
        attention_weights (torch.Tensor): Attention weights of shape
                                          (batch_size, n_heads, seq_len_q, seq_len_k)
        query_idx (int): Index of the query position to visualize
        head_idx (int): Index of the attention head to visualize
    
    Returns:
        torch.Tensor: Attention distribution of shape (seq_len_k,)
    
    Example:
        >>> # After computing attention
        >>> pattern = visualize_attention_pattern(weights, query_idx=5, head_idx=0)
        >>> # pattern shows how much position 5 attends to each other position
    """
    if attention_weights.dim() != 4:
        raise ValueError(
            f"Expected 4D attention weights (batch, heads, seq_q, seq_k), "
            f"got shape {attention_weights.shape}"
        )
    
    # Extract pattern for first batch item, specific head, specific query
    pattern = attention_weights[0, head_idx, query_idx, :]
    return pattern


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distributions.
    
    Entropy measures how "spread out" the attention is:
    - Low entropy: Attention is focused on few positions (peaked distribution)
    - High entropy: Attention is spread across many positions (uniform distribution)
    
    This can reveal whether the model is making confident, specific decisions
    or considering many possibilities.
    
    Args:
        attention_weights (torch.Tensor): Attention weights of shape
                                          (batch_size, n_heads, seq_len_q, seq_len_k)
    
    Returns:
        torch.Tensor: Entropy values of shape (batch_size, n_heads, seq_len_q)
    
    Formula:
        H(p) = -Σ p_i * log(p_i)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    
    # Compute entropy: -sum(p * log(p))
    log_weights = torch.log(attention_weights + eps)
    entropy = -torch.sum(attention_weights * log_weights, dim=-1)
    
    return entropy


def self_attention_example():
    """
    Demonstrate self-attention with a simple example.
    
    Self-attention: Query, Key, and Value all come from the same sequence.
    This allows each token to gather information from all other tokens.
    """
    print("=" * 70)
    print("Self-Attention Example")
    print("=" * 70)
    
    # Configuration
    batch_size = 1
    n_heads = 1
    seq_len = 5
    d_k = 8
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Key/Query dimension: {d_k}")
    
    # Create simple Q, K, V (all from same source in self-attention)
    torch.manual_seed(42)
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = Q.clone()  # In self-attention, K comes from same sequence as Q
    V = Q.clone()  # In self-attention, V comes from same sequence as Q
    
    # Create attention module
    attention = ScaledDotProductAttention(dropout=0.0)
    
    # Compute attention
    output, weights = attention(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Show attention weights
    print(f"\nAttention weights (how much each position attends to others):")
    weights_2d = weights.squeeze().detach()
    print(weights_2d)
    
    print(f"\nNote: Each row sums to 1.0 (probability distribution)")
    row_sums = weights_2d.sum(dim=-1)
    print(f"Row sums: {row_sums}")
    
    # Compute entropy
    entropy = compute_attention_entropy(weights)
    print(f"\nAttention entropy (higher = more spread out):")
    print(f"{entropy.squeeze()}")
    
    return output, weights


def cross_attention_example():
    """
    Demonstrate cross-attention with encoder-decoder setup.
    
    Cross-attention: Query comes from one sequence (decoder), while
    Key and Value come from another sequence (encoder). This allows
    the decoder to attend to the encoder's output.
    """
    print("\n" + "=" * 70)
    print("Cross-Attention Example")
    print("=" * 70)
    
    # Configuration
    batch_size = 1
    n_heads = 1
    decoder_seq_len = 4
    encoder_seq_len = 6
    d_k = 8
    
    print(f"\nConfiguration:")
    print(f"  Decoder sequence length: {decoder_seq_len}")
    print(f"  Encoder sequence length: {encoder_seq_len}")
    print(f"  Key/Query dimension: {d_k}")
    
    # Create Q from decoder, K and V from encoder
    torch.manual_seed(42)
    Q = torch.randn(batch_size, n_heads, decoder_seq_len, d_k)
    K = torch.randn(batch_size, n_heads, encoder_seq_len, d_k)
    V = K.clone()
    
    # Create attention module
    attention = ScaledDotProductAttention(dropout=0.0)
    
    # Compute attention
    output, weights = attention(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"  (decoder attends to encoder: {decoder_seq_len} queries × {encoder_seq_len} keys)")
    
    # Show attention weights
    print(f"\nAttention weights:")
    print(f"  Rows = decoder positions (queries)")
    print(f"  Cols = encoder positions (keys)")
    weights_2d = weights.squeeze().detach()
    print(weights_2d)
    
    return output, weights


if __name__ == "__main__":
    print("=" * 70)
    print("Scaled Dot-Product Attention Module - Example Usage")
    print("=" * 70)
    
    # Run self-attention example
    self_output, self_weights = self_attention_example()
    
    # Run cross-attention example
    cross_output, cross_weights = cross_attention_example()
    
    # Demonstrate masking
    print("\n" + "=" * 70)
    print("Masking Example")
    print("=" * 70)
    
    batch_size = 1
    n_heads = 1
    seq_len = 5
    d_k = 8
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = Q.clone()
    V = Q.clone()
    
    # Create look-ahead mask (for decoder)
    mask = create_look_ahead_mask(seq_len)
    print(f"\nLook-ahead mask (causal mask for decoder):")
    print(mask.squeeze())
    print("(1 = can attend, 0 = masked/cannot attend)")
    
    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(Q, K, V, mask=mask)
    
    print(f"\nAttention weights with look-ahead mask:")
    print(weights.squeeze().detach())
    print("(Notice: each position can only attend to itself and earlier positions)")
    
    print("\n" + "=" * 70)
    print("Module 3 Complete! Next: Multi-Head Attention")
    print("=" * 70)
