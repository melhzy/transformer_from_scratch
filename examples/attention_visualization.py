"""
Example: Visualizing Attention Patterns

This script demonstrates how to visualize attention weights from a trained
or randomly initialized Transformer model. Useful for understanding what
patterns the model learns.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.transformer import Transformer


def plot_attention_weights(attention_weights, layer_idx=0, head_idx=0, figsize=(10, 8)):
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor (batch, n_heads, seq_len, seq_len)
        layer_idx: Which layer to visualize (if multiple layers)
        head_idx: Which attention head to visualize
        figsize: Figure size for the plot
    """
    # Extract attention for specific head
    attn = attention_weights[0, head_idx].detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        attn,
        cmap='viridis',
        cbar=True,
        square=True,
        xticklabels=range(attn.shape[1]),
        yticklabels=range(attn.shape[0]),
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(f'Attention Weights - Head {head_idx}')
    plt.tight_layout()
    
    return plt.gcf()


def plot_all_heads(attention_weights, n_heads=8, figsize=(16, 12)):
    """
    Plot attention weights for all heads in a grid.
    
    Args:
        attention_weights: Attention weights tensor
        n_heads: Number of attention heads
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    for head_idx in range(min(n_heads, 8)):
        attn = attention_weights[0, head_idx].detach().cpu().numpy()
        
        sns.heatmap(
            attn,
            ax=axes[head_idx],
            cmap='viridis',
            cbar=True,
            square=True,
            cbar_kws={'label': 'Weight'}
        )
        
        axes[head_idx].set_title(f'Head {head_idx}')
        axes[head_idx].set_xlabel('Key')
        axes[head_idx].set_ylabel('Query')
    
    plt.tight_layout()
    return fig


def visualize_transformer_attention():
    """
    Main function to visualize attention patterns in a Transformer.
    """
    print("=" * 70)
    print("Attention Pattern Visualization")
    print("=" * 70)
    
    # Configuration
    config = {
        "src_vocab_size": 1000,
        "tgt_vocab_size": 1000,
        "d_model": 512,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "d_ff": 2048,
        "dropout": 0.0,  # Disable dropout for visualization
        "pad_idx": 0
    }
    
    print("\nCreating Transformer model...")
    model = Transformer(**config)
    model.eval()
    
    # Create sample sequences
    batch_size = 1
    src_seq_len = 12
    tgt_seq_len = 10
    
    print(f"Source sequence length: {src_seq_len}")
    print(f"Target sequence length: {tgt_seq_len}")
    
    src = torch.randint(1, 100, (batch_size, src_seq_len))
    tgt = torch.randint(1, 100, (batch_size, tgt_seq_len))
    
    # Forward pass through encoder
    print("\n" + "-" * 70)
    print("Encoder Self-Attention")
    print("-" * 70)
    
    with torch.no_grad():
        src_embedded = model.src_embedding(src)
        src_embedded = model.src_pos_encoding(src_embedded)
        
        # Get attention from first encoder layer
        encoder_layer = model.encoder.layers[0]
        _, encoder_attn_weights = encoder_layer.self_attention(
            src_embedded, src_embedded, src_embedded
        )
    
    print(f"Encoder attention weights shape: {encoder_attn_weights.shape}")
    print(f"  (batch_size, n_heads, seq_len, seq_len)")
    
    # Plot encoder attention
    fig1 = plot_attention_weights(
        encoder_attn_weights,
        head_idx=0,
        figsize=(10, 8)
    )
    plt.savefig('encoder_attention_head0.png', dpi=150, bbox_inches='tight')
    print("\nSaved: encoder_attention_head0.png")
    
    # Plot all encoder heads
    fig2 = plot_all_heads(encoder_attn_weights, n_heads=8)
    plt.savefig('encoder_attention_all_heads.png', dpi=150, bbox_inches='tight')
    print("Saved: encoder_attention_all_heads.png")
    
    # Forward pass through decoder
    print("\n" + "-" * 70)
    print("Decoder Self-Attention (with Causal Mask)")
    print("-" * 70)
    
    with torch.no_grad():
        memory = model.encode(src)
        tgt_embedded = model.tgt_embedding(tgt)
        tgt_embedded = model.tgt_pos_encoding(tgt_embedded)
        
        # Create causal mask
        from src.modules.decoder import create_causal_mask
        tgt_mask = create_causal_mask(tgt_seq_len)
        
        # Get attention from first decoder layer
        decoder_layer = model.decoder.layers[0]
        _, decoder_self_attn_weights = decoder_layer.self_attention(
            tgt_embedded, tgt_embedded, tgt_embedded, mask=tgt_mask
        )
    
    print(f"Decoder self-attention weights shape: {decoder_self_attn_weights.shape}")
    
    # Plot decoder self-attention
    fig3 = plot_attention_weights(
        decoder_self_attn_weights,
        head_idx=0,
        figsize=(10, 8)
    )
    plt.savefig('decoder_self_attention_head0.png', dpi=150, bbox_inches='tight')
    print("\nSaved: decoder_self_attention_head0.png")
    print("  (Note the triangular pattern due to causal masking)")
    
    # Cross-attention (decoder to encoder)
    print("\n" + "-" * 70)
    print("Decoder Cross-Attention (Decoder → Encoder)")
    print("-" * 70)
    
    with torch.no_grad():
        # After self-attention
        tgt_after_self = decoder_layer.norm1(
            tgt_embedded + decoder_layer.dropout(
                decoder_layer.self_attention(
                    tgt_embedded, tgt_embedded, tgt_embedded, mask=tgt_mask
                )[0]
            )
        )
        
        # Cross-attention
        _, cross_attn_weights = decoder_layer.cross_attention(
            tgt_after_self, memory, memory
        )
    
    print(f"Cross-attention weights shape: {cross_attn_weights.shape}")
    print(f"  (batch_size, n_heads, tgt_len, src_len)")
    
    # Plot cross-attention
    fig4 = plot_attention_weights(
        cross_attn_weights,
        head_idx=0,
        figsize=(10, 8)
    )
    plt.savefig('decoder_cross_attention_head0.png', dpi=150, bbox_inches='tight')
    print("\nSaved: decoder_cross_attention_head0.png")
    print("  (Shows which source positions each target position attends to)")
    
    print("\n" + "=" * 70)
    print("Attention visualization complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - encoder_attention_head0.png")
    print("  - encoder_attention_all_heads.png")
    print("  - decoder_self_attention_head0.png")
    print("  - decoder_cross_attention_head0.png")


if __name__ == "__main__":
    visualize_transformer_attention()
    
    # Keep plots open
    print("\nClose the plot windows to exit...")
    plt.show()
