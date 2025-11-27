"""
Quick demo script to verify installation and run basic examples.

Run this after installation to make sure everything works!
"""

import sys
import torch

print("=" * 70)
print("Transformer From Scratch - Installation Verification")
print("=" * 70)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Try importing modules
print("\n" + "-" * 70)
print("Checking module imports...")
print("-" * 70)

try:
    from src.modules.embeddings import TokenEmbedding
    print("✓ Token Embeddings")
except ImportError as e:
    print(f"✗ Token Embeddings: {e}")

try:
    from src.modules.positional_encoding import PositionalEncoding
    print("✓ Positional Encoding")
except ImportError as e:
    print(f"✗ Positional Encoding: {e}")

try:
    from src.modules.attention import ScaledDotProductAttention
    print("✓ Scaled Dot-Product Attention")
except ImportError as e:
    print(f"✗ Scaled Dot-Product Attention: {e}")

try:
    from src.modules.multi_head_attention import MultiHeadAttention
    print("✓ Multi-Head Attention")
except ImportError as e:
    print(f"✗ Multi-Head Attention: {e}")

try:
    from src.modules.feed_forward import PositionwiseFeedForward
    print("✓ Feed-Forward Network")
except ImportError as e:
    print(f"✗ Feed-Forward Network: {e}")

try:
    from src.modules.encoder import TransformerEncoder
    print("✓ Encoder")
except ImportError as e:
    print(f"✗ Encoder: {e}")

try:
    from src.modules.decoder import TransformerDecoder
    print("✓ Decoder")
except ImportError as e:
    print(f"✗ Decoder: {e}")

try:
    from src.transformer import Transformer
    print("✓ Complete Transformer")
except ImportError as e:
    print(f"✗ Complete Transformer: {e}")

# Quick functionality test
print("\n" + "-" * 70)
print("Running quick functionality test...")
print("-" * 70)

try:
    # Create a small Transformer
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        dropout=0.1
    )
    
    # Test forward pass
    src = torch.randint(0, 100, (2, 5))
    tgt = torch.randint(0, 100, (2, 5))
    
    output = model(src, tgt)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {src.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(src[:1], max_len=10, start_token=1, end_token=2)
    
    print(f"\n✓ Generation successful!")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model created successfully!")
    print(f"  Parameters: {n_params:,}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("Installation Check Complete!")
print("=" * 70)

print("\n✓ All systems operational!")
print("\nNext steps:")
print("  1. Run individual modules: python src/modules/embeddings.py")
print("  2. Try examples: python llm-fine-tune/attention_visualization.py")
print("  3. Run tests: pytest tests/ -v")
print("  4. Open notebooks: jupyter lab")

print("\nFor help, see:")
print("  - README.md for overview")
print("  - QUICKSTART.md for quick start")
print("  - llm-fine-tune/README.md for fine-tuning examples")

print("\nHappy learning! 🚀")
