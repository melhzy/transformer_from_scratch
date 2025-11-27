"""
Module 8: Complete Transformer Model
=====================================

This module brings together all components into the full Transformer architecture
for sequence-to-sequence tasks. The complete model consists of:

1. **Source & Target Embeddings**: Convert tokens to dense vectors
2. **Positional Encodings**: Add position information
3. **Encoder**: Process source sequence
4. **Decoder**: Generate target sequence
5. **Output Projection**: Map to vocabulary logits

The Architecture:
-----------------
    Source Sequence
         |
    [Embedding + Positional Encoding]
         |
    [Encoder Stack (N layers)]
         |
    Encoder Output (Memory)
         |         \
         |          \
    Target Sequence  |
         |           |
    [Embedding + PE] |
         |           |
    [Decoder Stack]--+  (Cross-attention to encoder)
         |
    [Linear + Softmax]
         |
    Output Probabilities

Key Design Decisions:
---------------------
1. **Weight Tying**: Share embeddings with output projection (reduces parameters)
2. **Dropout**: Applied throughout for regularization
3. **Masking**: Padding and causal masks for proper attention
4. **Initialization**: Careful initialization for stable training

Applications:
-------------
- Machine Translation (original use case)
- Text Summarization
- Question Answering
- Code Generation
- Any sequence-to-sequence task

DeepSeek-R1 Insights:
---------------------
Modern Transformers build on this foundation with:
- Larger models (billions of parameters)
- Better optimizations (Flash Attention, gradient checkpointing)
- Advanced training techniques (curriculum learning, mixture of experts)
- Emergent capabilities (reasoning, planning, tool use)

Reference: "Attention Is All You Need", Section 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.embeddings import TokenEmbedding
from src.modules.positional_encoding import PositionalEncoding
from src.modules.encoder import TransformerEncoder
from src.modules.decoder import TransformerDecoder, create_causal_mask
from typing import Optional
import math


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    Implements the full architecture from "Attention Is All You Need",
    combining encoder and decoder stacks with embeddings and output projection.
    
    Args:
        src_vocab_size (int): Size of source vocabulary
        tgt_vocab_size (int): Size of target vocabulary
        d_model (int): Dimension of the model (default: 512)
        n_heads (int): Number of attention heads (default: 8)
        n_encoder_layers (int): Number of encoder layers (default: 6)
        n_decoder_layers (int): Number of decoder layers (default: 6)
        d_ff (int): Dimension of feed-forward network (default: 2048)
        dropout (float): Dropout probability (default: 0.1)
        max_seq_len (int): Maximum sequence length (default: 5000)
        activation (str): Activation function (default: 'relu')
        tie_weights (bool): Whether to tie embeddings with output projection (default: True)
        pad_idx (Optional[int]): Padding token index (default: None)
    
    Shape:
        - src: (batch_size, src_seq_len) - Source token indices
        - tgt: (batch_size, tgt_seq_len) - Target token indices
        - output: (batch_size, tgt_seq_len, tgt_vocab_size) - Output logits
    
    Examples:
        >>> transformer = Transformer(
        ...     src_vocab_size=10000,
        ...     tgt_vocab_size=10000,
        ...     d_model=512,
        ...     n_heads=8,
        ...     n_encoder_layers=6,
        ...     n_decoder_layers=6,
        ...     d_ff=2048
        ... )
        >>> src = torch.randint(0, 10000, (2, 20))
        >>> tgt = torch.randint(0, 10000, (2, 15))
        >>> output = transformer(src, tgt)
        >>> print(output.shape)  # torch.Size([2, 15, 10000])
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        activation: str = "relu",
        tie_weights: bool = True,
        pad_idx: Optional[int] = None
    ):
        super(Transformer, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        
        # Source and target embeddings
        self.src_embedding = TokenEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            padding_idx=pad_idx
        )
        
        self.tgt_embedding = TokenEmbedding(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            padding_idx=pad_idx
        )
        
        # Positional encodings
        self.src_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        self.tgt_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_encoder_layers,
            dropout=dropout,
            activation=activation
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_decoder_layers,
            dropout=dropout,
            activation=activation
        )
        
        # Output projection (vocabulary logits)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Weight tying (share embeddings with output projection)
        # This reduces parameters and often improves performance
        if tie_weights:
            self.output_projection.weight = self.tgt_embedding.embedding.weight
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Create masks for source and target sequences.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
        
        Returns:
            tuple: (src_mask, tgt_mask, memory_mask)
                - src_mask: Padding mask for encoder
                - tgt_mask: Combined causal + padding mask for decoder self-attention
                - memory_mask: Padding mask for decoder cross-attention to encoder
        """
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        # Source padding mask
        if self.pad_idx is not None:
            # Shape: (batch_size, 1, 1, src_seq_len)
            src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
            memory_mask = src_mask  # Same mask for cross-attention
        else:
            src_mask = None
            memory_mask = None
        
        # Target causal mask
        tgt_causal_mask = create_causal_mask(tgt_seq_len, device=tgt.device)
        
        # Target padding mask
        if self.pad_idx is not None:
            tgt_padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
            # Combine causal and padding masks
            tgt_mask = tgt_causal_mask & tgt_padding_mask
        else:
            tgt_mask = tgt_causal_mask
        
        return src_mask, tgt_mask, memory_mask
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            src_mask: Optional source mask
        
        Returns:
            torch.Tensor: Encoder output of shape (batch_size, src_seq_len, d_model)
        """
        # Embed and add positional encoding
        src_embedded = self.src_embedding(src)
        src_embedded = self.src_pos_encoding(src_embedded)
        
        # Encode
        memory = self.encoder(src_embedded, mask=src_mask)
        
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder memory.
        
        Args:
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            memory: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Optional target mask (causal)
            memory_mask: Optional memory mask (padding)
        
        Returns:
            torch.Tensor: Decoder output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Embed and add positional encoding
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        
        # Decode
        output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        return output
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the complete Transformer.
        
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None or tgt_mask is None or memory_mask is None:
            src_mask, tgt_mask, memory_mask = self.create_masks(src, tgt)
        
        # Encode source
        memory = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        start_token: int = 1,
        end_token: int = 2,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate target sequence autoregressively.
        
        This implements greedy decoding by default, but supports
        temperature scaling, top-k, and nucleus (top-p) sampling.
        
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            max_len: Maximum generation length
            start_token: Start-of-sequence token ID
            end_token: End-of-sequence token ID
            temperature: Sampling temperature (1.0 = no change, <1 = more confident)
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling threshold
        
        Returns:
            torch.Tensor: Generated sequences of shape (batch_size, seq_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source once
        src_mask, _, memory_mask = self.create_masks(
            src,
            torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        )
        memory = self.encode(src, src_mask)
        
        # Initialize target with start token
        tgt = torch.full(
            (batch_size, 1),
            start_token,
            dtype=torch.long,
            device=device
        )
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for _ in range(max_len - 1):
            # Create target mask
            tgt_mask = create_causal_mask(tgt.size(1), device=device)
            
            # Decode
            decoder_output = self.decode(tgt, memory, tgt_mask, memory_mask)
            
            # Get logits for last position
            logits = self.output_projection(decoder_output[:, -1, :])
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1),
                    dim=-1
                )
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update finished sequences
            finished |= (next_token.squeeze(1) == end_token)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences finished
            if finished.all():
                break
        
        return tgt


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: Transformer) -> dict:
    """
    Count parameters in the Transformer model.
    
    Args:
        model: Transformer instance
    
    Returns:
        dict: Detailed parameter breakdown
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    breakdown = {
        "total": total,
        "trainable": trainable,
        "src_embedding": sum(p.numel() for p in model.src_embedding.parameters()),
        "tgt_embedding": sum(p.numel() for p in model.tgt_embedding.parameters()),
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "output_projection": sum(p.numel() for p in model.output_projection.parameters()),
    }
    
    return breakdown


def get_model_size_mb(model: Transformer) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: Transformer instance
    
    Returns:
        float: Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == "__main__":
    print("=" * 70)
    print("Complete Transformer Model - Example Usage")
    print("=" * 70)
    
    # Configuration (matching original Transformer paper)
    config = {
        "src_vocab_size": 10000,
        "tgt_vocab_size": 10000,
        "d_model": 512,
        "n_heads": 8,
        "n_encoder_layers": 6,
        "n_decoder_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_len": 5000,
        "pad_idx": 0
    }
    
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    transformer = Transformer(**config)
    
    # Parameter analysis
    print("\n" + "-" * 70)
    print("Model Statistics")
    print("-" * 70)
    
    param_breakdown = count_parameters(transformer)
    print(f"\nParameter counts:")
    for component, count in param_breakdown.items():
        if count > 0:
            print(f"  {component}: {count:,}")
    
    model_size = get_model_size_mb(transformer)
    print(f"\nModel size: {model_size:.2f} MB")
    
    # Example forward pass
    print("\n" + "-" * 70)
    print("Forward Pass Example")
    print("-" * 70)
    
    batch_size = 2
    src_seq_len = 20
    tgt_seq_len = 15
    
    # Create sample data
    src = torch.randint(1, config["src_vocab_size"], (batch_size, src_seq_len))
    tgt = torch.randint(1, config["tgt_vocab_size"], (batch_size, tgt_seq_len))
    
    # Add padding tokens
    src[:, -2:] = 0  # Last 2 tokens are padding
    tgt[:, -2:] = 0
    
    print(f"\nSource shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Source (first seq, first 10 tokens): {src[0, :10].tolist()}")
    print(f"Target (first seq, first 10 tokens): {tgt[0, :10].tolist()}")
    
    # Forward pass
    output = transformer(src, tgt)
    
    print(f"\nOutput logits shape: {output.shape}")
    print(f"  (batch_size, tgt_seq_len, tgt_vocab_size)")
    
    # Convert logits to probabilities
    probs = F.softmax(output, dim=-1)
    print(f"\nOutput probabilities shape: {probs.shape}")
    print(f"First position, top 5 token probabilities:")
    top5_probs, top5_indices = torch.topk(probs[0, 0], k=5)
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        print(f"  Token {idx}: {prob:.4f}")
    
    # Generation example
    print("\n" + "-" * 70)
    print("Autoregressive Generation Example")
    print("-" * 70)
    
    transformer.eval()
    with torch.no_grad():
        src_test = torch.randint(1, config["src_vocab_size"], (1, 10))
        generated = transformer.generate(
            src_test,
            max_len=20,
            start_token=1,
            end_token=2,
            temperature=1.0
        )
    
    print(f"\nGenerated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print("\n" + "=" * 70)
    print("Module 8 Complete! All core modules implemented!")
    print("=" * 70)
