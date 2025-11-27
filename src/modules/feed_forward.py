"""
Module 5: Position-wise Feed-Forward Network
=============================================

After attention mechanisms process the input, each position passes through
a position-wise feed-forward network (FFN) independently and identically.
This adds non-linearity and increases the model's capacity to learn complex patterns.

Key Concepts:
-------------
1. **Position-wise**: Applied independently to each position (no interaction between positions)
2. **Two Linear Layers**: First expands dimension, second projects back
3. **Non-linearity**: ReLU (or variants) between the layers
4. **Dimension Expansion**: Inner layer typically 4x larger than d_model

Why Feed-Forward Networks?
---------------------------
- Attention is linear (weighted sum), FFN adds non-linearity
- Allows the model to learn complex transformations
- Applied to each position independently (like a 1x1 convolution in CNNs)
- The expansion and projection creates a bottleneck architecture

The Intuition:
--------------
Think of it as a small neural network for each position:
- First layer: Expand to a higher-dimensional space (more expressive)
- Activation: Add non-linearity (enable complex patterns)
- Second layer: Project back to original dimension

This is like having a mini-MLP at each position that can learn to transform
the representations in arbitrary ways.

Mathematical Formulation:
-------------------------
    FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
    
    or more generally:
    FFN(x) = activation(x·W_1 + b_1)·W_2 + b_2

where:
    W_1 ∈ R^(d_model × d_ff)    - First layer weights
    W_2 ∈ R^(d_ff × d_model)    - Second layer weights
    d_ff = 4 × d_model (typically)
    activation = ReLU (originally), but GELU, SwiGLU are also used

Reference: "Attention Is All You Need", Section 3.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Applies two linear transformations with a non-linearity in between,
    independently to each position. This is equivalent to two 1x1 convolutions.
    
    Args:
        d_model (int): Dimension of input and output
        d_ff (int): Dimension of inner layer (typically 4 * d_model)
        dropout (float): Dropout probability (default: 0.1)
        activation (str): Activation function - 'relu', 'gelu', or 'swish' (default: 'relu')
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    
    Examples:
        >>> d_model, d_ff = 512, 2048
        >>> ffn = PositionwiseFeedForward(d_model, d_ff)
        >>> x = torch.randn(2, 10, 512)
        >>> output = ffn(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super(PositionwiseFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # First linear layer: d_model -> d_ff (expansion)
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: d_ff -> d_model (projection)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(p=dropout)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            "relu": F.relu,
            "gelu": F.gelu,
            "swish": lambda x: x * torch.sigmoid(x),  # SiLU/Swish
            "silu": F.silu,
        }
        
        if activation.lower() not in activations:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose from {list(activations.keys())}"
            )
        
        return activations[activation.lower()]
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        
        Process:
            1. Expand: x @ W1 + b1  ->  (batch, seq_len, d_ff)
            2. Activate: activation(·)
            3. Dropout: For regularization
            4. Project: · @ W2 + b2  ->  (batch, seq_len, d_model)
            5. Dropout: Final regularization
        """
        # First linear transformation + activation
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        
        # Second linear transformation
        # Shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear2(x)
        x = self.dropout_layer(x)
        
        return x


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) Feed-Forward Network.
    
    An alternative to the standard FFN that uses gating mechanisms.
    Popular variants include SwiGLU (used in LLaMA, PaLM) and GeGLU.
    These have shown improvements over standard FFN in some architectures.
    
    The gating mechanism allows the network to learn what information to pass through,
    similar to LSTM gates but applied to feed-forward layers.
    
    Args:
        d_model (int): Dimension of input and output
        d_ff (int): Dimension of inner layer
        dropout (float): Dropout probability (default: 0.1)
        activation (str): Activation for gate - 'swish' or 'gelu' (default: 'swish')
    
    Mathematical Formulation:
        GLU(x, W, V, b, c) = (xW + b) ⊗ activation(xV + c)
        
    where ⊗ is element-wise multiplication (gating).
    
    Reference: 
        - "GLU Variants Improve Transformer" (Shazeer, 2020)
        - Used in modern LLMs like LLaMA, PaLM
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "swish"
    ):
        super(GLUFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two parallel linear layers for gating
        self.linear_gate = nn.Linear(d_model, d_ff)  # Gate path
        self.linear_value = nn.Linear(d_model, d_ff)  # Value path
        
        # Output projection
        self.linear_out = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Activation function for gate
        if activation.lower() == "swish":
            self.activation = F.silu  # SiLU is the PyTorch name for Swish
        elif activation.lower() == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation for GLU: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for linear in [self.linear_gate, self.linear_value, self.linear_out]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GLU feed-forward network.
        
        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, d_model)
        """
        # Split into gate and value paths
        gate = self.activation(self.linear_gate(x))
        value = self.linear_value(x)
        
        # Element-wise multiplication (gating)
        x = gate * value
        x = self.dropout(x)
        
        # Output projection
        x = self.linear_out(x)
        x = self.dropout(x)
        
        return x


# ============================================================================
# Utility Functions for Understanding and Visualization
# ============================================================================

def count_parameters(model: nn.Module) -> dict:
    """
    Count the number of parameters in the feed-forward network.
    
    Args:
        model: The FFN module
    
    Returns:
        dict: Parameter counts and breakdown
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    breakdown = {}
    for name, param in model.named_parameters():
        breakdown[name] = param.numel()
    
    return {
        "total": total,
        "trainable": trainable,
        "breakdown": breakdown
    }


def analyze_activation_statistics(
    ffn: PositionwiseFeedForward,
    x: torch.Tensor
) -> dict:
    """
    Analyze activation statistics through the FFN.
    
    This helps understand how information flows through the network
    and can reveal issues like dead ReLUs or saturation.
    
    Args:
        ffn: The feed-forward network
        x: Input tensor
    
    Returns:
        dict: Statistics about activations at each layer
    """
    with torch.no_grad():
        # After first linear + activation
        h = ffn.linear1(x)
        h_activated = ffn.activation(h)
        
        # After second linear
        output = ffn.linear2(h_activated)
        
        stats = {
            "input": {
                "mean": x.mean().item(),
                "std": x.std().item(),
                "min": x.min().item(),
                "max": x.max().item(),
            },
            "hidden_before_activation": {
                "mean": h.mean().item(),
                "std": h.std().item(),
                "min": h.min().item(),
                "max": h.max().item(),
            },
            "hidden_after_activation": {
                "mean": h_activated.mean().item(),
                "std": h_activated.std().item(),
                "min": h_activated.min().item(),
                "max": h_activated.max().item(),
                "sparsity": (h_activated == 0).float().mean().item(),  # For ReLU
            },
            "output": {
                "mean": output.mean().item(),
                "std": output.std().item(),
                "min": output.min().item(),
                "max": output.max().item(),
            }
        }
        
        return stats


def compare_activations():
    """
    Compare different activation functions in FFN.
    """
    print("=" * 70)
    print("Comparing Activation Functions")
    print("=" * 70)
    
    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    activations = ["relu", "gelu", "swish"]
    
    for act_name in activations:
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0, activation=act_name)
        output = ffn(x)
        
        print(f"\n{act_name.upper()}:")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")


if __name__ == "__main__":
    print("=" * 70)
    print("Position-wise Feed-Forward Network Module - Example Usage")
    print("=" * 70)
    
    # Configuration
    d_model = 512
    d_ff = 2048  # 4x expansion
    batch_size = 2
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  Input/Output dimension (d_model): {d_model}")
    print(f"  Hidden dimension (d_ff): {d_ff}")
    print(f"  Expansion factor: {d_ff / d_model}x")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create FFN module
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.1, activation="relu")
    
    # Count parameters
    param_info = count_parameters(ffn)
    print(f"\nParameters:")
    print(f"  Total: {param_info['total']:,}")
    for name, count in param_info['breakdown'].items():
        print(f"  {name}: {count:,}")
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = ffn(x)
    print(f"Output shape: {output.shape}")
    
    # Analyze activations
    print("\n" + "-" * 70)
    print("Activation Statistics")
    print("-" * 70)
    
    stats = analyze_activation_statistics(ffn, x)
    
    for layer_name, layer_stats in stats.items():
        print(f"\n{layer_name}:")
        for stat_name, value in layer_stats.items():
            if stat_name == "sparsity":
                print(f"  {stat_name}: {value:.2%} (proportion of zeros)")
            else:
                print(f"  {stat_name}: {value:.4f}")
    
    # Compare activations
    print("\n" + "-" * 70)
    compare_activations()
    
    # Demonstrate GLU variant
    print("\n" + "-" * 70)
    print("GLU Feed-Forward Network (SwiGLU)")
    print("-" * 70)
    
    glu_ffn = GLUFeedForward(d_model, d_ff, dropout=0.1, activation="swish")
    glu_output = glu_ffn(x)
    
    print(f"\nGLU FFN output shape: {glu_output.shape}")
    glu_params = count_parameters(glu_ffn)
    print(f"GLU parameters: {glu_params['total']:,}")
    print(f"Standard FFN parameters: {param_info['total']:,}")
    print(f"Difference: GLU uses ~{glu_params['total'] / param_info['total']:.2f}x more parameters")
    print("  (due to parallel gate and value paths)")
    
    print("\n" + "=" * 70)
    print("Module 5 Complete! Next: Encoder Layer")
    print("=" * 70)
