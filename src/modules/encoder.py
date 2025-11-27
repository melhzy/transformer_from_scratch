"""
Module 6: Transformer Encoder
==============================

The encoder is the "understanding" part of the Transformer. It processes the
input sequence and creates rich contextual representations. Each encoder layer
combines self-attention (to understand relationships) with feed-forward networks
(to transform representations).

Key Concepts:
-------------
1. **Encoder Layer**: Building block with self-attention + FFN
2. **Layer Normalization**: Stabilizes training
3. **Residual Connections**: Enables gradient flow in deep networks
4. **Encoder Stack**: Multiple layers (typically 6) stacked together

Architecture of One Encoder Layer:
-----------------------------------
    Input
      |
      +--> Multi-Head Self-Attention
      |           |
      +--------Add & LayerNorm
                  |
                  +--> Feed-Forward Network
                  |           |
                  +--------Add & LayerNorm
                              |
                           Output

The "Add & LayerNorm" pattern is crucial:
- Add: Residual connection (x + Sublayer(x))
- LayerNorm: Normalizes across feature dimension

Why This Architecture?
----------------------
- Self-attention: Captures relationships between all positions
- FFN: Adds non-linearity and transformation capacity
- Residual connections: Prevent vanishing gradients
- Layer normalization: Stabilizes training dynamics

The Intuition:
--------------
Think of encoding as multiple passes of refinement:
- Layer 1: Captures basic patterns and local dependencies
- Layer 2: Builds on Layer 1, finds more complex relationships
- Layer 3-6: Progressively more abstract and global understanding

Each layer has the opportunity to attend to different patterns,
creating a hierarchical understanding of the input.

Reference: "Attention Is All You Need", Section 3.1
"""

import torch
import torch.nn as nn
from src.modules.multi_head_attention import MultiHeadAttention
from src.modules.feed_forward import PositionwiseFeedForward
from typing import Optional
import copy


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Consists of:
    1. Multi-head self-attention
    2. Add & LayerNorm
    3. Position-wise feed-forward network
    4. Add & LayerNorm
    
    Args:
        d_model (int): Dimension of the model
        n_heads (int): Number of attention heads
        d_ff (int): Dimension of feed-forward network
        dropout (float): Dropout probability (default: 0.1)
        activation (str): Activation function for FFN (default: 'relu')
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    
    Examples:
        >>> layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> output = layer(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Position-wise feed-forward network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (Optional[torch.Tensor]): Attention mask for padding
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Sub-layer 1: Multi-head self-attention
        # Pre-LN variant: LayerNorm before attention (more stable)
        # Post-LN variant: LayerNorm after attention (original paper)
        # We use Post-LN here as in the original paper
        
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, mask=mask)
        
        # Add & Norm (residual connection + layer normalization)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Sub-layer 2: Feed-forward network
        ff_output = self.feed_forward(x)
        
        # Add & Norm
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N Transformer Encoder Layers.
    
    The complete encoder consists of multiple identical layers stacked together.
    Each layer refines the representation, creating increasingly abstract
    and context-aware embeddings.
    
    Args:
        encoder_layer (EncoderLayer): A single encoder layer to be cloned
        n_layers (int): Number of encoder layers to stack
        
    Alternatively, you can specify architecture parameters:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        n_layers (int): Number of layers
        dropout (float): Dropout probability
        activation (str): Activation function
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    
    Examples:
        >>> encoder = TransformerEncoder(
        ...     d_model=512, n_heads=8, d_ff=2048,
        ...     n_layers=6, dropout=0.1
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = encoder(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        encoder_layer: Optional[EncoderLayer] = None,
        n_layers: int = 6,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super(TransformerEncoder, self).__init__()
        
        # If encoder_layer is provided, clone it n_layers times
        if encoder_layer is not None:
            self.layers = nn.ModuleList([
                copy.deepcopy(encoder_layer) for _ in range(n_layers)
            ])
            self.n_layers = n_layers
            self.d_model = encoder_layer.d_model
        # Otherwise, create layers from scratch
        elif all(x is not None for x in [d_model, n_heads, d_ff]):
            self.layers = nn.ModuleList([
                EncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(n_layers)
            ])
            self.n_layers = n_layers
            self.d_model = d_model
        else:
            raise ValueError(
                "Either provide encoder_layer or all of (d_model, n_heads, d_ff)"
            )
        
        # Final layer normalization (optional, used in some variants)
        self.norm = nn.LayerNorm(self.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (Optional[torch.Tensor]): Attention mask
        
        Returns:
            torch.Tensor: Encoded representations of shape (batch_size, seq_len, d_model)
        """
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        return x


# ============================================================================
# Utility Functions for Understanding and Visualization
# ============================================================================

def count_encoder_parameters(encoder: TransformerEncoder) -> dict:
    """
    Count parameters in the encoder.
    
    Args:
        encoder: TransformerEncoder instance
    
    Returns:
        dict: Parameter counts and breakdown
    """
    total = sum(p.numel() for p in encoder.parameters())
    
    # Count parameters per layer
    layer_params = sum(p.numel() for p in encoder.layers[0].parameters())
    
    return {
        "total": total,
        "per_layer": layer_params,
        "n_layers": encoder.n_layers,
        "final_norm": sum(p.numel() for p in encoder.norm.parameters()),
    }


def analyze_layer_outputs(
    encoder: TransformerEncoder,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Analyze the output of each encoder layer.
    
    This helps understand how representations evolve through the layers.
    
    Args:
        encoder: TransformerEncoder instance
        x: Input tensor
        mask: Optional attention mask
    
    Returns:
        dict: Statistics for each layer's output
    """
    layer_stats = {}
    
    with torch.no_grad():
        current = x
        
        for i, layer in enumerate(encoder.layers):
            current = layer(current, mask=mask)
            
            layer_stats[f"layer_{i}"] = {
                "mean": current.mean().item(),
                "std": current.std().item(),
                "min": current.min().item(),
                "max": current.max().item(),
                "norm": current.norm().item(),
            }
    
    return layer_stats


def visualize_representation_similarity(
    encoder: TransformerEncoder,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Compute similarity between input and output representations.
    
    This shows how much the encoder transforms the input.
    
    Args:
        encoder: TransformerEncoder instance
        x: Input tensor of shape (batch_size, seq_len, d_model)
    
    Returns:
        torch.Tensor: Cosine similarity for each position
    """
    with torch.no_grad():
        output = encoder(x)
        
        # Compute cosine similarity between input and output
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, x.size(-1))
        output_flat = output.view(-1, output.size(-1))
        
        similarity = torch.nn.functional.cosine_similarity(
            x_flat, output_flat, dim=1
        )
        
        return similarity.view(x.size(0), x.size(1))


if __name__ == "__main__":
    print("=" * 70)
    print("Transformer Encoder Module - Example Usage")
    print("=" * 70)
    
    # Configuration (matching the original Transformer paper)
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    dropout = 0.1
    batch_size = 2
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Feed-forward dimension (d_ff): {d_ff}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create encoder
    encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
        activation="relu"
    )
    
    # Count parameters
    param_info = count_encoder_parameters(encoder)
    print(f"\nParameters:")
    print(f"  Total: {param_info['total']:,}")
    print(f"  Per layer: {param_info['per_layer']:,}")
    print(f"  Number of layers: {param_info['n_layers']}")
    print(f"  Final norm: {param_info['final_norm']:,}")
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = encoder(x)
    print(f"Output shape: {output.shape}")
    
    # Analyze layer outputs
    print("\n" + "-" * 70)
    print("Layer-by-Layer Analysis")
    print("-" * 70)
    
    layer_stats = analyze_layer_outputs(encoder, x)
    
    for layer_name, stats in layer_stats.items():
        print(f"\n{layer_name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    # Visualize representation similarity
    print("\n" + "-" * 70)
    print("Input-Output Similarity")
    print("-" * 70)
    
    similarity = visualize_representation_similarity(encoder, x)
    print(f"\nCosine similarity between input and output:")
    print(f"  Mean: {similarity.mean().item():.4f}")
    print(f"  Std: {similarity.std().item():.4f}")
    print(f"  Range: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")
    print("\n(Lower similarity indicates more transformation)")
    
    # Demonstrate with padding mask
    print("\n" + "-" * 70)
    print("Using Padding Mask")
    print("-" * 70)
    
    # Create a simple padding mask (last 2 positions are padding)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, -2:] = 0  # Mask last 2 positions
    
    print(f"\nPadding mask shape: {mask.shape}")
    print(f"Mask pattern (first batch): {mask[0, 0, 0].tolist()}")
    print("  (1 = valid token, 0 = padding)")
    
    output_with_mask = encoder(x, mask=mask)
    print(f"\nOutput with mask shape: {output_with_mask.shape}")
    
    print("\n" + "=" * 70)
    print("Module 6 Complete! Next: Decoder Layer")
    print("=" * 70)
