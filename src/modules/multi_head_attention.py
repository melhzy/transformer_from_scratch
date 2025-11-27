"""
Module 4: Multi-Head Attention
===============================

While single-head attention is powerful, multi-head attention allows the model
to jointly attend to information from different representation subspaces at
different positions. This is like having multiple "attention perspectives"
simultaneously.

Key Concepts:
-------------
1. **Multiple Heads**: Run attention multiple times in parallel
2. **Subspace Projection**: Each head operates on a different learned projection
3. **Concatenation**: Combine all heads' outputs
4. **Final Projection**: Mix information from all heads

Why Multiple Heads?
-------------------
- Different heads can learn different relationships (syntax, semantics, etc.)
- Increases model capacity without drastically increasing computation
- Each head can focus on different aspects (local vs global, content vs position)
- Empirically shown to improve performance significantly

The Intuition:
--------------
Think of reading a text with different lenses:
- Head 1: Focus on grammar and syntax
- Head 2: Focus on semantic relationships
- Head 3: Focus on long-range dependencies
- Head 4: Focus on local context

Each head captures different patterns, and combining them gives a richer
representation.

Mathematical Formulation:
-------------------------
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where:
    head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    
    W_i^Q ∈ R^(d_model × d_k)    - Query projection for head i
    W_i^K ∈ R^(d_model × d_k)    - Key projection for head i
    W_i^V ∈ R^(d_model × d_v)    - Value projection for head i
    W^O ∈ R^(h·d_v × d_model)    - Output projection

Typically: d_k = d_v = d_model / h, so total dimension remains d_model

Reference: "Attention Is All You Need", Section 3.2.2
"""

import torch
import torch.nn as nn
from src.modules.attention import ScaledDotProductAttention
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Allows the model to jointly attend to information from different representation
    subspaces. Implements parallel attention heads that are concatenated and
    linearly transformed.
    
    Args:
        d_model (int): Dimension of the model (input/output dimension)
        n_heads (int): Number of parallel attention heads
        dropout (float): Dropout probability (default: 0.1)
        
    The dimensions are typically chosen such that:
        d_k = d_v = d_model / n_heads
    This ensures the total computational cost is similar to single-head attention
    with full dimensionality.
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
        - Attention weights: (batch_size, n_heads, seq_len_q, seq_len_k)
    
    Examples:
        >>> d_model, n_heads = 512, 8
        >>> mha = MultiHeadAttention(d_model, n_heads)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output, weights = mha(x, x, x)  # Self-attention
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super(MultiHeadAttention, self).__init__()
        
        # Validate that d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads  # Value dimension per head
        
        # Linear projections for Q, K, V
        # Instead of separate projections per head, we use one large projection
        # and split it into heads later (more efficient)
        self.W_q = nn.Linear(d_model, d_model)  # Projects to all heads at once
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Dropout for output
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k).
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose: (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine attention heads back into a single tensor.
        
        Args:
            x: Tensor of shape (batch_size, n_heads, seq_len, d_k)
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        
        # Transpose: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k)
        x = x.transpose(1, 2)
        
        # Reshape: (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model)
            mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
                                           or broadcastable. 1 for valid, 0 for masked.
        
        Returns:
            tuple: (output, attention_weights)
                - output: Shape (batch_size, seq_len_q, d_model)
                - attention_weights: Shape (batch_size, n_heads, seq_len_q, seq_len_k)
        
        Notes:
            - For self-attention: query = key = value (same sequence)
            - For cross-attention: query from decoder, key/value from encoder
        """
        batch_size = query.size(0)
        
        # Step 1: Linear projections
        # Project inputs to Q, K, V spaces
        # Shape: (batch_size, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Step 2: Split into multiple heads
        # Shape: (batch_size, n_heads, seq_len, d_k)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Step 3: Apply scaled dot-product attention
        # attn_output: (batch_size, n_heads, seq_len_q, d_k)
        # attention_weights: (batch_size, n_heads, seq_len_q, seq_len_k)
        attn_output, attention_weights = self.attention(Q, K, V, mask=mask)
        
        # Step 4: Concatenate heads
        # Shape: (batch_size, seq_len_q, d_model)
        concat_output = self._combine_heads(attn_output)
        
        # Step 5: Final linear projection
        # Shape: (batch_size, seq_len_q, d_model)
        output = self.W_o(concat_output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, attention_weights


# ============================================================================
# Utility Functions for Understanding and Visualization
# ============================================================================

def analyze_head_importance(attention_weights: torch.Tensor) -> dict:
    """
    Analyze which attention heads are most important based on attention patterns.
    
    Different metrics can indicate head importance:
    - Entropy: How focused/spread the attention is
    - Max attention: How much attention concentrates on single positions
    - Variance: How varied the attention patterns are
    
    Args:
        attention_weights: Shape (batch_size, n_heads, seq_len_q, seq_len_k)
    
    Returns:
        dict: Analysis results for each head
    """
    batch_size, n_heads, seq_len_q, seq_len_k = attention_weights.shape
    
    results = {}
    
    for head in range(n_heads):
        head_weights = attention_weights[:, head, :, :]  # (batch, seq_len_q, seq_len_k)
        
        # Compute entropy (averaged over queries and batch)
        eps = 1e-9
        log_weights = torch.log(head_weights + eps)
        entropy = -torch.sum(head_weights * log_weights, dim=-1)
        avg_entropy = entropy.mean().item()
        
        # Compute max attention (how much focuses on single position)
        max_attention = head_weights.max(dim=-1)[0].mean().item()
        
        # Compute variance
        variance = head_weights.var(dim=-1).mean().item()
        
        results[f"head_{head}"] = {
            "avg_entropy": avg_entropy,
            "avg_max_attention": max_attention,
            "avg_variance": variance
        }
    
    return results


def visualize_head_patterns(
    attention_weights: torch.Tensor,
    head_indices: list = None
) -> dict:
    """
    Extract attention patterns for specific heads for visualization.
    
    Args:
        attention_weights: Shape (batch_size, n_heads, seq_len_q, seq_len_k)
        head_indices: List of head indices to visualize (default: all heads)
    
    Returns:
        dict: Attention patterns for each specified head
    """
    if head_indices is None:
        head_indices = list(range(attention_weights.size(1)))
    
    patterns = {}
    for head_idx in head_indices:
        # Get attention for this head, first batch item
        # Shape: (seq_len_q, seq_len_k)
        pattern = attention_weights[0, head_idx, :, :].detach()
        patterns[f"head_{head_idx}"] = pattern
    
    return patterns


def compare_self_vs_cross_attention():
    """
    Demonstrate the difference between self-attention and cross-attention
    in multi-head attention.
    """
    print("=" * 70)
    print("Self-Attention vs Cross-Attention")
    print("=" * 70)
    
    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 10
    
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    
    # Self-attention: Q, K, V from same sequence
    print("\n1. Self-Attention (Q = K = V from same sequence)")
    x = torch.randn(batch_size, seq_len, d_model)
    self_output, self_weights = mha(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {self_output.shape}")
    print(f"   Attention weights shape: {self_weights.shape}")
    print(f"   Use case: Encoder self-attention, decoder self-attention")
    
    # Cross-attention: Q from one sequence, K, V from another
    print("\n2. Cross-Attention (Q from decoder, K=V from encoder)")
    decoder_seq = torch.randn(batch_size, 5, d_model)
    encoder_seq = torch.randn(batch_size, seq_len, d_model)
    cross_output, cross_weights = mha(decoder_seq, encoder_seq, encoder_seq)
    print(f"   Query (decoder) shape: {decoder_seq.shape}")
    print(f"   Key/Value (encoder) shape: {encoder_seq.shape}")
    print(f"   Output shape: {cross_output.shape}")
    print(f"   Attention weights shape: {cross_weights.shape}")
    print(f"   Use case: Decoder cross-attention to encoder")
    
    return self_output, cross_output


if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Head Attention Module - Example Usage")
    print("=" * 70)
    
    # Configuration
    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Dimension per head (d_k): {d_model // n_heads}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create multi-head attention module
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    
    # Count parameters
    n_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Self-attention example
    print("\n" + "-" * 70)
    print("Self-Attention Example")
    print("-" * 70)
    output, weights = mha(x, x, x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"  (batch_size, n_heads, seq_len_q, seq_len_k)")
    
    # Analyze heads
    print("\n" + "-" * 70)
    print("Head Analysis")
    print("-" * 70)
    
    head_analysis = analyze_head_importance(weights)
    print("\nPer-head statistics:")
    for head_name, stats in head_analysis.items():
        print(f"\n{head_name}:")
        print(f"  Average entropy: {stats['avg_entropy']:.4f}")
        print(f"  Average max attention: {stats['avg_max_attention']:.4f}")
        print(f"  Average variance: {stats['avg_variance']:.4f}")
    
    # Compare self vs cross attention
    print("\n" + "-" * 70)
    compare_self_vs_cross_attention()
    
    print("\n" + "=" * 70)
    print("Module 4 Complete! Next: Feed-Forward Network")
    print("=" * 70)
