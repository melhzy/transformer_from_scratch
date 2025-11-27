"""
Module 2: Positional Encodings
===============================

Transformers process all tokens in parallel (unlike RNNs which are sequential).
This means they have no inherent notion of token position or order. Positional
encodings inject position information into the input embeddings.

Key Concepts:
-------------
1. **Position-Awareness**: The model needs to know which token comes before/after
2. **Sinusoidal Functions**: Use sin/cos waves of different frequencies
3. **Fixed vs Learned**: Original paper uses fixed; some variants use learned
4. **Generalization**: Sinusoidal encodings can extrapolate to longer sequences

Why Sinusoidal Positional Encodings?
-------------------------------------
- Each position gets a unique encoding
- Different frequencies capture different scales of position
- Can interpolate/extrapolate to unseen sequence lengths
- No additional parameters to learn

Mathematical Formulation:
-------------------------
For position pos and dimension i:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
    - pos: position in the sequence (0, 1, 2, ...)
    - i: dimension index (0 to d_model/2)
    - Even dimensions use sine, odd dimensions use cosine

The wavelengths form a geometric progression from 2π to 10000·2π.

Reference: "Attention Is All You Need", Section 3.5
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sinusoidal functions.
    
    Adds position information to token embeddings using fixed sine and cosine
    functions of different frequencies. This allows the model to learn to
    attend to relative positions.
    
    Args:
        d_model (int): Dimension of the embeddings (must be even)
        max_len (int): Maximum sequence length to pre-compute (default: 5000)
        dropout (float): Dropout probability applied after adding PE (default: 0.1)
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    
    Examples:
        >>> d_model = 512
        >>> pos_encoding = PositionalEncoding(d_model)
        >>> embeddings = torch.randn(2, 10, 512)  # batch_size=2, seq_len=10
        >>> output = pos_encoding(embeddings)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super(PositionalEncoding, self).__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_len, d_model) to store positional encodings
        # We pre-compute all positional encodings up to max_len for efficiency
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices: [0, 1, 2, ..., max_len-1]
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the sinusoidal functions
        # This implements: 10000^(2i/d_model) for i in [0, d_model/2)
        # We use exp and log for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices in the embedding dimension
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices in the embedding dimension
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        # This allows broadcasting when adding to embeddings
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of model state)
        # Buffers are saved in state_dict but not updated by optimizer
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Embeddings with positional information, same shape as input
        """
        # Get sequence length from input
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}. "
                f"Increase max_len when creating PositionalEncoding."
            )
        
        # Add positional encodings to embeddings
        # pe[:, :seq_len] has shape (1, seq_len, d_model)
        # Broadcasting adds it to each batch
        x = x + self.pe[:, :seq_len]
        
        # Apply dropout for regularization
        return self.dropout(x)
    
    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get the positional encoding for a specific sequence length.
        Useful for visualization and analysis.
        
        Args:
            seq_len (int): Length of the sequence
        
        Returns:
            torch.Tensor: Positional encodings of shape (seq_len, d_model)
        """
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        return self.pe[0, :seq_len, :].clone()


# ============================================================================
# Utility Functions for Understanding and Visualization
# ============================================================================

def visualize_positional_encoding(d_model: int = 512, max_len: int = 100):
    """
    Create visualization data for positional encodings.
    
    This helps understand how different positions are encoded and how
    the sinusoidal patterns vary across dimensions.
    
    Args:
        d_model: Embedding dimension
        max_len: Number of positions to visualize
    
    Returns:
        torch.Tensor: Positional encodings of shape (max_len, d_model)
    """
    pe_layer = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)
    return pe_layer.get_positional_encoding(max_len)


def compute_relative_position_distance(
    pos_encoding: PositionalEncoding,
    pos1: int,
    pos2: int
) -> float:
    """
    Compute the distance between two positional encodings.
    
    This helps understand how the model might perceive relative positions.
    The sinusoidal encoding ensures that relative positions have consistent
    patterns regardless of absolute position.
    
    Args:
        pos_encoding: PositionalEncoding instance
        pos1: First position index
        pos2: Second position index
    
    Returns:
        float: Euclidean distance between the two positional encodings
    """
    pe1 = pos_encoding.get_positional_encoding(max(pos1, pos2) + 1)[pos1]
    pe2 = pos_encoding.get_positional_encoding(max(pos1, pos2) + 1)[pos2]
    
    distance = torch.dist(pe1, pe2, p=2)
    return distance.item()


def analyze_periodicity(d_model: int = 512, positions: int = 1000):
    """
    Analyze the periodic properties of positional encodings.
    
    Different dimensions have different wavelengths, from 2π (high frequency)
    to 10000·2π (low frequency). This creates a rich representation of position.
    
    Args:
        d_model: Embedding dimension
        positions: Number of positions to analyze
    
    Returns:
        dict: Analysis results including wavelengths per dimension
    """
    # Compute wavelengths for each dimension pair
    wavelengths = []
    for i in range(0, d_model, 2):
        div_term = 10000.0 ** (i / d_model)
        wavelength = 2 * math.pi * div_term
        wavelengths.append(wavelength)
    
    return {
        'min_wavelength': min(wavelengths),
        'max_wavelength': max(wavelengths),
        'wavelengths': wavelengths,
        'description': (
            f"Wavelengths range from {min(wavelengths):.2f} to {max(wavelengths):.2e}, "
            f"allowing the model to capture both fine-grained and coarse position information."
        )
    }


class LearnedPositionalEncoding(nn.Module):
    """
    Alternative: Learned Positional Encodings
    
    Instead of fixed sinusoidal functions, this variant learns position embeddings
    during training. This is used in some Transformer variants like BERT.
    
    Pros:
        - Can potentially learn task-specific position patterns
        - Simpler implementation
    
    Cons:
        - Cannot extrapolate to longer sequences than seen during training
        - Requires learning additional parameters
    
    Args:
        d_model (int): Dimension of the embeddings
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encodings to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Embeddings with positional information
        """
        batch_size, seq_len, _ = x.size()
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Look up position embeddings
        pos_embeddings = self.position_embeddings(positions)
        
        # Add to input embeddings
        x = x + pos_embeddings
        
        return self.dropout(x)


if __name__ == "__main__":
    # Example usage and demonstration
    print("=" * 70)
    print("Positional Encoding Module - Example Usage")
    print("=" * 70)
    
    # Configuration
    d_model = 512
    max_len = 100
    batch_size = 2
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  Embedding dimension (d_model): {d_model}")
    print(f"  Maximum sequence length: {max_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Current sequence length: {seq_len}")
    
    # Create positional encoding layer
    pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=0.1)
    
    # Create sample embeddings (from token embedding layer)
    embeddings = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput embeddings shape: {embeddings.shape}")
    
    # Add positional encodings
    output = pos_encoding(embeddings)
    print(f"Output shape: {output.shape}")
    
    # Analyze positional encodings
    print("\n" + "-" * 70)
    print("Positional Encoding Analysis")
    print("-" * 70)
    
    # Get raw positional encodings
    pe_matrix = pos_encoding.get_positional_encoding(seq_len)
    print(f"\nPositional encoding matrix shape: {pe_matrix.shape}")
    print(f"PE statistics:")
    print(f"  Mean: {pe_matrix.mean().item():.4f}")
    print(f"  Std: {pe_matrix.std().item():.4f}")
    print(f"  Min: {pe_matrix.min().item():.4f}")
    print(f"  Max: {pe_matrix.max().item():.4f}")
    
    # Compute distances between positions
    pos1, pos2, pos3 = 0, 1, 5
    dist_01 = compute_relative_position_distance(pos_encoding, pos1, pos2)
    dist_05 = compute_relative_position_distance(pos_encoding, pos1, pos3)
    print(f"\nRelative position distances:")
    print(f"  Distance between position {pos1} and {pos2}: {dist_01:.4f}")
    print(f"  Distance between position {pos1} and {pos3}: {dist_05:.4f}")
    print(f"  (Greater distance for positions farther apart)")
    
    # Analyze periodicity
    periodicity = analyze_periodicity(d_model, positions=max_len)
    print(f"\nPeriodicity analysis:")
    print(f"  Minimum wavelength: {periodicity['min_wavelength']:.2f}")
    print(f"  Maximum wavelength: {periodicity['max_wavelength']:.2e}")
    print(f"  Number of frequency bands: {len(periodicity['wavelengths'])}")
    print(f"\n  {periodicity['description']}")
    
    # Compare with learned positional encodings
    print("\n" + "-" * 70)
    print("Comparison: Sinusoidal vs Learned Positional Encodings")
    print("-" * 70)
    
    learned_pos_encoding = LearnedPositionalEncoding(d_model, max_len=max_len)
    learned_output = learned_pos_encoding(embeddings)
    
    print(f"\nSinusoidal PE: Fixed, generalizes to unseen lengths")
    print(f"Learned PE: Trainable, limited to max_len={max_len}")
    print(f"Both output shapes: {output.shape}")
    
    print("\n" + "=" * 70)
    print("Module 2 Complete! Next: Scaled Dot-Product Attention")
    print("=" * 70)
