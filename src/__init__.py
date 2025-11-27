"""
Transformer From Scratch
========================

A step-by-step implementation of the Transformer architecture from
"Attention Is All You Need" (Vaswani et al., 2017), enriched with
insights from modern developments like DeepSeek-R1.

This package provides modular, educational implementations of:
- Token embeddings
- Positional encodings
- Scaled dot-product attention
- Multi-head attention
- Position-wise feed-forward networks
- Encoder and decoder layers
- Complete Transformer architecture

Each module is designed for learning, with clear explanations and
visualizations to help understand how Transformers work.
"""

__version__ = "0.1.0"
__author__ = "Transformer Learning Project"

from src.modules.embeddings import TokenEmbedding
from src.modules.positional_encoding import PositionalEncoding
from src.modules.attention import ScaledDotProductAttention
from src.modules.multi_head_attention import MultiHeadAttention
from src.modules.feed_forward import PositionwiseFeedForward
from src.modules.encoder import EncoderLayer, TransformerEncoder
from src.modules.decoder import DecoderLayer, TransformerDecoder
from src.transformer import Transformer

__all__ = [
    "TokenEmbedding",
    "PositionalEncoding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionwiseFeedForward",
    "EncoderLayer",
    "TransformerEncoder",
    "DecoderLayer",
    "TransformerDecoder",
    "Transformer",
]
