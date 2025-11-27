"""
Module 7: Transformer Decoder
==============================

The decoder is the "generation" part of the Transformer. It generates output
sequences one token at a time, attending to both its own previous outputs
(via masked self-attention) and the encoder's output (via cross-attention).

Key Concepts:
-------------
1. **Masked Self-Attention**: Prevents looking at future tokens during generation
2. **Cross-Attention**: Attends to encoder output (encoder-decoder attention)
3. **Causal Mask**: Ensures autoregressive property (can only see past)
4. **Decoder Layer**: Self-attention + cross-attention + FFN
5. **Decoder Stack**: Multiple layers stacked together

Architecture of One Decoder Layer:
-----------------------------------
    Input
      |
      +--> Masked Multi-Head Self-Attention
      |           |
      +--------Add & LayerNorm
                  |
                  +--> Multi-Head Cross-Attention (to encoder)
                  |           |
                  +--------Add & LayerNorm
                              |
                              +--> Feed-Forward Network
                              |           |
                              +--------Add & LayerNorm
                                          |
                                       Output

Three Sub-layers:
1. Masked self-attention: Look at previously generated tokens
2. Cross-attention: Attend to encoder output
3. Feed-forward: Transform representations

Why Masked Self-Attention?
---------------------------
During training, we have the full target sequence, but we must prevent the
model from "cheating" by looking at future tokens. The mask ensures position i
can only attend to positions ≤ i.

During inference, we generate one token at a time, so the mask naturally emerges
from the autoregressive process.

The Intuition:
--------------
Think of the decoder as a conditional generator:
- Masked self-attention: "What have I generated so far?"
- Cross-attention: "What does the input (encoder) tell me?"
- Feed-forward: "How should I transform this information?"

Each layer refines the generation, considering both past context
and the source input.

Reference: "Attention Is All You Need", Section 3.1
"""

import torch
import torch.nn as nn
from src.modules.multi_head_attention import MultiHeadAttention
from src.modules.feed_forward import PositionwiseFeedForward
from typing import Optional
import copy


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & LayerNorm
    3. Multi-head cross-attention (to encoder)
    4. Add & LayerNorm
    5. Position-wise feed-forward network
    6. Add & LayerNorm
    
    Args:
        d_model (int): Dimension of the model
        n_heads (int): Number of attention heads
        d_ff (int): Dimension of feed-forward network
        dropout (float): Dropout probability (default: 0.1)
        activation (str): Activation function for FFN (default: 'relu')
    
    Shape:
        - tgt: (batch_size, tgt_seq_len, d_model) - Target sequence
        - memory: (batch_size, src_seq_len, d_model) - Encoder output
        - Output: (batch_size, tgt_seq_len, d_model)
    
    Examples:
        >>> layer = DecoderLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> tgt = torch.randn(2, 10, 512)
        >>> memory = torch.randn(2, 15, 512)
        >>> output = layer(tgt, memory)
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
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Masked multi-head self-attention (for target sequence)
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Multi-head cross-attention (decoder attends to encoder)
        self.cross_attention = MultiHeadAttention(
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
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            tgt (torch.Tensor): Target sequence of shape (batch_size, tgt_seq_len, d_model)
            memory (torch.Tensor): Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask (Optional[torch.Tensor]): Mask for target self-attention (causal mask)
            memory_mask (Optional[torch.Tensor]): Mask for encoder-decoder attention
        
        Returns:
            torch.Tensor: Output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Sub-layer 1: Masked self-attention
        # Target attends to itself with causal mask
        self_attn_output, _ = self.self_attention(
            query=tgt,
            key=tgt,
            value=tgt,
            mask=tgt_mask
        )
        
        # Add & Norm
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        
        # Sub-layer 2: Cross-attention to encoder
        # Query from decoder, Key and Value from encoder
        cross_attn_output, _ = self.cross_attention(
            query=tgt,
            key=memory,
            value=memory,
            mask=memory_mask
        )
        
        # Add & Norm
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        
        # Sub-layer 3: Feed-forward network
        ff_output = self.feed_forward(tgt)
        
        # Add & Norm
        tgt = self.norm3(tgt + self.dropout(ff_output))
        
        return tgt


class TransformerDecoder(nn.Module):
    """
    Stack of N Transformer Decoder Layers.
    
    The complete decoder consists of multiple identical layers stacked together.
    Each layer refines the generation by attending to both the target history
    and the encoder output.
    
    Args:
        decoder_layer (DecoderLayer): A single decoder layer to be cloned
        n_layers (int): Number of decoder layers to stack
        
    Alternatively, you can specify architecture parameters:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        n_layers (int): Number of layers
        dropout (float): Dropout probability
        activation (str): Activation function
    
    Shape:
        - tgt: (batch_size, tgt_seq_len, d_model)
        - memory: (batch_size, src_seq_len, d_model)
        - Output: (batch_size, tgt_seq_len, d_model)
    
    Examples:
        >>> decoder = TransformerDecoder(
        ...     d_model=512, n_heads=8, d_ff=2048,
        ...     n_layers=6, dropout=0.1
        ... )
        >>> tgt = torch.randn(2, 10, 512)
        >>> memory = torch.randn(2, 15, 512)
        >>> output = decoder(tgt, memory)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        decoder_layer: Optional[DecoderLayer] = None,
        n_layers: int = 6,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super(TransformerDecoder, self).__init__()
        
        # If decoder_layer is provided, clone it n_layers times
        if decoder_layer is not None:
            self.layers = nn.ModuleList([
                copy.deepcopy(decoder_layer) for _ in range(n_layers)
            ])
            self.n_layers = n_layers
            self.d_model = decoder_layer.d_model
        # Otherwise, create layers from scratch
        elif all(x is not None for x in [d_model, n_heads, d_ff]):
            self.layers = nn.ModuleList([
                DecoderLayer(
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
                "Either provide decoder_layer or all of (d_model, n_heads, d_ff)"
            )
        
        # Final layer normalization
        self.norm = nn.LayerNorm(self.d_model)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            tgt (torch.Tensor): Target sequence of shape (batch_size, tgt_seq_len, d_model)
            memory (torch.Tensor): Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask (Optional[torch.Tensor]): Causal mask for target
            memory_mask (Optional[torch.Tensor]): Mask for encoder output
        
        Returns:
            torch.Tensor: Decoded representations of shape (batch_size, tgt_seq_len, d_model)
        """
        # Pass through each decoder layer sequentially
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Apply final layer normalization
        tgt = self.norm(tgt)
        
        return tgt


# ============================================================================
# Utility Functions for Understanding and Visualization
# ============================================================================

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask for decoder self-attention.
    
    This ensures that position i can only attend to positions j where j <= i,
    preventing the model from looking at future tokens.
    
    Args:
        seq_len (int): Length of the sequence
        device (torch.device): Device to create the mask on
    
    Returns:
        torch.Tensor: Causal mask of shape (1, 1, seq_len, seq_len)
    
    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask.squeeze())
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_combined_mask(
    tgt_seq_len: int,
    tgt_padding: Optional[torch.Tensor] = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create a combined causal + padding mask for decoder.
    
    Combines:
    1. Causal mask (no looking ahead)
    2. Padding mask (no attending to padding tokens)
    
    Args:
        tgt_seq_len (int): Target sequence length
        tgt_padding (Optional[torch.Tensor]): Padding mask (1 = valid, 0 = padding)
                                              Shape: (batch_size, tgt_seq_len)
        device (torch.device): Device for the mask
    
    Returns:
        torch.Tensor: Combined mask of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
    """
    # Create causal mask
    causal_mask = create_causal_mask(tgt_seq_len, device=device)
    
    if tgt_padding is not None:
        # Expand padding mask to attention shape
        # From (batch_size, tgt_seq_len) to (batch_size, 1, 1, tgt_seq_len)
        padding_mask = tgt_padding.unsqueeze(1).unsqueeze(2)
        
        # Combine masks (both must be 1 for valid attention)
        combined_mask = causal_mask & padding_mask
        return combined_mask.float()
    else:
        return causal_mask


def count_decoder_parameters(decoder: TransformerDecoder) -> dict:
    """
    Count parameters in the decoder.
    
    Args:
        decoder: TransformerDecoder instance
    
    Returns:
        dict: Parameter counts and breakdown
    """
    total = sum(p.numel() for p in decoder.parameters())
    
    # Count parameters per layer
    layer_params = sum(p.numel() for p in decoder.layers[0].parameters())
    
    return {
        "total": total,
        "per_layer": layer_params,
        "n_layers": decoder.n_layers,
        "final_norm": sum(p.numel() for p in decoder.norm.parameters()),
    }


def visualize_cross_attention(
    decoder_layer: DecoderLayer,
    tgt: torch.Tensor,
    memory: torch.Tensor
) -> torch.Tensor:
    """
    Extract cross-attention weights from a decoder layer.
    
    This shows which encoder positions the decoder is attending to.
    
    Args:
        decoder_layer: A single DecoderLayer
        tgt: Target tensor
        memory: Encoder output
    
    Returns:
        torch.Tensor: Cross-attention weights
    """
    with torch.no_grad():
        # Forward through self-attention
        self_attn_output, _ = decoder_layer.self_attention(tgt, tgt, tgt)
        tgt_after_self = decoder_layer.norm1(tgt + self_attn_output)
        
        # Forward through cross-attention and extract weights
        _, cross_attn_weights = decoder_layer.cross_attention(
            query=tgt_after_self,
            key=memory,
            value=memory
        )
        
        return cross_attn_weights


if __name__ == "__main__":
    print("=" * 70)
    print("Transformer Decoder Module - Example Usage")
    print("=" * 70)
    
    # Configuration
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    dropout = 0.1
    batch_size = 2
    tgt_seq_len = 10
    src_seq_len = 15
    
    print(f"\nConfiguration:")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Feed-forward dimension (d_ff): {d_ff}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target sequence length: {tgt_seq_len}")
    print(f"  Source sequence length: {src_seq_len}")
    
    # Create decoder
    decoder = TransformerDecoder(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
        activation="relu"
    )
    
    # Count parameters
    param_info = count_decoder_parameters(decoder)
    print(f"\nParameters:")
    print(f"  Total: {param_info['total']:,}")
    print(f"  Per layer: {param_info['per_layer']:,}")
    print(f"  Number of layers: {param_info['n_layers']}")
    print(f"  (Note: Decoder has ~1.5x params vs encoder due to cross-attention)")
    
    # Create sample inputs
    tgt = torch.randn(batch_size, tgt_seq_len, d_model)
    memory = torch.randn(batch_size, src_seq_len, d_model)  # From encoder
    
    print(f"\nTarget input shape: {tgt.shape}")
    print(f"Encoder memory shape: {memory.shape}")
    
    # Create causal mask
    tgt_mask = create_causal_mask(tgt_seq_len)
    print(f"\nCausal mask shape: {tgt_mask.shape}")
    print("Causal mask pattern (ensures no looking ahead):")
    print(tgt_mask.squeeze()[:5, :5])  # Show first 5x5 for clarity
    
    # Forward pass
    output = decoder(tgt, memory, tgt_mask=tgt_mask)
    print(f"\nOutput shape: {output.shape}")
    
    # Demonstrate cross-attention visualization
    print("\n" + "-" * 70)
    print("Cross-Attention Analysis")
    print("-" * 70)
    
    decoder_layer = decoder.layers[0]
    cross_attn_weights = visualize_cross_attention(decoder_layer, tgt, memory)
    
    print(f"\nCross-attention weights shape: {cross_attn_weights.shape}")
    print(f"  (batch, n_heads, tgt_len, src_len)")
    print(f"\nFirst head, first target position attention distribution:")
    first_position_attention = cross_attn_weights[0, 0, 0, :]
    print(f"  Attention to source positions: {first_position_attention[:10].tolist()}")
    print(f"  Sum (should be ~1.0): {first_position_attention.sum().item():.4f}")
    
    # Demonstrate combined mask (causal + padding)
    print("\n" + "-" * 70)
    print("Combined Mask (Causal + Padding)")
    print("-" * 70)
    
    # Create padding mask (last 2 positions are padding)
    tgt_padding = torch.ones(batch_size, tgt_seq_len)
    tgt_padding[:, -2:] = 0
    
    combined_mask = create_combined_mask(
        tgt_seq_len,
        tgt_padding=tgt_padding
    )
    
    print(f"\nCombined mask shape: {combined_mask.shape}")
    print("First batch, showing both causal and padding effects:")
    print(combined_mask[0, 0, :, :])
    print("  (Last 2 rows show restricted attention due to padding)")
    
    print("\n" + "=" * 70)
    print("Module 7 Complete! Next: Full Transformer")
    print("=" * 70)
