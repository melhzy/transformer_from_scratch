# 🧠 Transformer From Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Learn Transformers step by step** — A comprehensive educational repository implementing the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), enriched with modern insights from DeepSeek-R1 and other recent advances.

## 🎯 What You'll Learn

This repository takes you on a journey from basic building blocks to a complete working Transformer:

1. **Token Embeddings** — Converting discrete tokens to continuous representations
2. **Positional Encodings** — Injecting sequence order information
3. **Scaled Dot-Product Attention** — The core attention mechanism
4. **Multi-Head Attention** — Parallel attention perspectives
5. **Feed-Forward Networks** — Position-wise transformations
6. **Encoder Layers** — Understanding the input
7. **Decoder Layers** — Generating outputs
8. **Complete Transformer** — Putting it all together

Each module includes:
- ✅ Clear mathematical explanations
- ✅ Detailed code comments
- ✅ Visualization utilities
- ✅ Example usage and demonstrations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Running Your First Transformer

```python
import torch
from src.transformer import Transformer

# Create a Transformer model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048
)

# Example forward pass
src = torch.randint(0, 10000, (2, 20))  # Batch of 2, sequence length 20
tgt = torch.randint(0, 10000, (2, 15))  # Batch of 2, sequence length 15

output = model(src, tgt)  # Shape: (2, 15, 10000)
print(f"Output logits shape: {output.shape}")

# Generate sequences
generated = model.generate(src, max_len=50, temperature=0.8)
print(f"Generated sequence: {generated[0].tolist()}")
```

## 📚 Learning Path

### Quick Start: Google Colab (Recommended!)

**No installation required!** Run all tutorials directly in your browser:

**Transformer Foundations (Start Here):**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/transformer-foundation/01_embeddings_and_positional_encoding.ipynb) Tutorial 1: Embeddings & Positional Encoding
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/transformer-foundation/02_scaled_dot_product_attention.ipynb) Tutorial 2: Scaled Dot-Product Attention
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/transformer-foundation/03_multi_head_attention.ipynb) Tutorial 3: Multi-Head Attention
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/transformer-foundation/04_feed_forward_networks.ipynb) Tutorial 4: Feed-Forward Networks
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/transformer-foundation/05_encoder_decoder.ipynb) Tutorial 5: Encoder & Decoder
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/transformer-foundation/06_complete_transformer.ipynb) Tutorial 6: Complete Transformer
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/transformer-foundation/07_emergent_reasoning.ipynb) Tutorial 7: Emergent Reasoning

**LLM Fine-Tuning (Advanced):**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/01_introduction_to_fine_tuning.ipynb) Tutorial 1: Introduction to Fine-Tuning
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/02_lora_implementation.ipynb) Tutorial 2: LoRA Implementation
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/03_data_preparation.ipynb) Tutorial 3: Data Preparation
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/04_instruction_tuning.ipynb) Tutorial 4: Instruction Tuning
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/05_evaluation_metrics.ipynb) Tutorial 5: Evaluation Metrics

Click any badge above to open that tutorial directly in Google Colab and start learning immediately!

---

### For Beginners

Start with the modules in order, running the example code in each file:

```bash
# Run individual modules to see examples
python src/modules/embeddings.py
python src/modules/positional_encoding.py
python src/modules/attention.py
python src/modules/multi_head_attention.py
python src/modules/feed_forward.py
python src/modules/encoder.py
python src/modules/decoder.py
python src/transformer.py
```

### For Visual Learners

Explore the Jupyter notebooks in the `transformer-foundation/` directory:

1. **01_embeddings_and_positional_encoding.ipynb** — Visualize embeddings and position encodings
2. **02_scaled_dot_product_attention.ipynb** — Core attention mechanism
3. **03_multi_head_attention.ipynb** — Interactive attention visualizations
4. **04_feed_forward_networks.ipynb** — FFN and layer normalization
5. **05_encoder_decoder.ipynb** — Encoder and decoder architecture
6. **06_complete_transformer.ipynb** — Build and train a Transformer
7. **07_emergent_reasoning.ipynb** — Explore advanced capabilities with DeepSeek

```bash
# Launch Jupyter Lab
jupyter lab
```

### For Practitioners

Check out the fine-tuning examples in `llm-fine-tune/`:

- `train_translation.py` — Machine translation example
- `text_generation.py` — Autoregressive text generation
- `attention_visualization.py` — Visualize attention patterns

## 🏗️ Project Structure

```
transformer_from_scratch/
│
├── src/                           # Main source code
│   ├── modules/                   # Individual components
│   │   ├── embeddings.py         # Token embeddings
│   │   ├── positional_encoding.py # Positional encodings
│   │   ├── attention.py          # Scaled dot-product attention
│   │   ├── multi_head_attention.py # Multi-head attention
│   │   ├── feed_forward.py       # Feed-forward networks
│   │   ├── encoder.py            # Encoder layers
│   │   └── decoder.py            # Decoder layers
│   ├── transformer.py            # Complete Transformer model
│   └── __init__.py
│
├── transformer-foundation/        # Jupyter notebooks (tutorials)
│   ├── 01_embeddings_and_positional_encoding.ipynb
│   ├── 02_scaled_dot_product_attention.ipynb
│   ├── 03_multi_head_attention.ipynb
│   ├── 04_feed_forward_networks.ipynb
│   ├── 05_encoder_decoder.ipynb
│   ├── 06_complete_transformer.ipynb
│   ├── 07_emergent_reasoning.ipynb
│   └── TUTORIALS_OVERVIEW.md
│
├── tests/                         # Unit tests
│   ├── test_embeddings.py
│   ├── test_attention.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_transformer.py
│
├── llm-fine-tune/                 # LLM fine-tuning examples
│   ├── train_translation.py
│   ├── text_generation.py
│   └── attention_visualization.py
│
├── papers/                        # Reference papers
│   └── (Add "Attention Is All You Need" PDF here)
│
├── pyproject.toml                # Project configuration
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 🔬 Key Features

### 1. Educational Focus

Every line of code is documented with:
- Clear variable names
- Inline comments explaining the "why"
- Mathematical formulations in docstrings
- References to the original paper

### 2. Modular Design

Each component can be understood and used independently:

```python
from src.modules.attention import ScaledDotProductAttention
from src.modules.multi_head_attention import MultiHeadAttention
from src.modules.positional_encoding import PositionalEncoding

# Use individual components
attention = ScaledDotProductAttention(dropout=0.1)
Q, K, V = torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64)
output, weights = attention(Q, K, V)
```

### 3. Modern Extensions

While staying true to the original paper, we include modern variants:

- **GLU Feed-Forward Networks** (SwiGLU, used in LLaMA)
- **Pre-LayerNorm** vs **Post-LayerNorm**
- **Learned Positional Encodings** (BERT-style)
- **Advanced Generation** (top-k, nucleus sampling)

### 4. Visualization Tools

Built-in utilities to understand model behavior:

```python
from src.modules.attention import compute_attention_entropy
from src.modules.embeddings import get_nearest_neighbors

# Analyze attention patterns
entropy = compute_attention_entropy(attention_weights)

# Find similar tokens
neighbors = get_nearest_neighbors(embedding_layer, token_idx=42, top_k=5)
```

## 📖 Architecture Details

### Model Configuration (Original Paper)

```python
d_model = 512          # Model dimension
n_heads = 8            # Number of attention heads
d_ff = 2048            # Feed-forward dimension (4x expansion)
n_layers = 6           # Both encoder and decoder
dropout = 0.1          # Dropout rate
max_seq_len = 5000     # Maximum sequence length
```

### Parameter Count

For the default configuration:
- **Total Parameters**: ~65M (depends on vocabulary size)
- **Encoder**: ~37M parameters
- **Decoder**: ~56M parameters (includes cross-attention)
- **Embeddings**: ~10M parameters (for 20K vocabulary)

## 🎓 DeepSeek-R1 Insights

Modern large language models like DeepSeek-R1 build upon the Transformer foundation with:

### Architectural Improvements
- **Scaling**: Billions of parameters vs millions
- **Sparse Attention**: More efficient for long sequences
- **Mixture of Experts (MoE)**: Conditional computation
- **Better Normalizations**: RMSNorm instead of LayerNorm

### Training Techniques
- **Large Batch Training**: 1000s of examples per batch
- **Curriculum Learning**: Progressive difficulty
- **Advanced Optimizers**: AdamW with learning rate scheduling
- **Gradient Checkpointing**: Memory efficiency

### Emergent Capabilities
- **Chain-of-Thought Reasoning**: Step-by-step problem solving
- **Few-Shot Learning**: Learning from examples
- **Tool Use**: Integrating with external systems
- **Multimodal Understanding**: Beyond just text

This repository focuses on the core architecture, providing a foundation for understanding these advanced techniques.

## 🧪 Testing

Run the test suite to verify all components:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_attention.py -v
```

## 📊 Notebooks

### 1. Embeddings and Positional Encoding

Explore how tokens are converted to continuous representations and how position information is injected:

- Visualize embedding spaces
- Plot positional encoding patterns
- Understand wavelength hierarchies

### 2. Attention Mechanism

Interactive visualizations of the attention mechanism:

- Self-attention patterns
- Cross-attention in encoder-decoder
- Effect of masking
- Attention head diversity

### 3. Complete Transformer

Build and train a complete Transformer:

- Data preparation and batching
- Training loop with loss computation
- Evaluation and generation
- Attention visualization during inference

### 4. Emergent Reasoning

Explore how Transformers can exhibit reasoning capabilities:

- Pattern recognition and completion
- Arithmetic and logical operations
- Compositional generalization
- In-context learning examples

## 🤝 Contributing

Contributions are welcome! Whether it's:

- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New features or examples
- 🎨 Visualization enhancements

Please open an issue or submit a pull request.

## 📚 Resources

### Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

### Additional Learning Materials

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) by Harvard NLP
- [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)

### Related Projects

- [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [Transformers by Hugging Face](https://github.com/huggingface/transformers)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Vaswani et al.** for the original Transformer architecture
- **The PyTorch team** for an excellent deep learning framework
- **The open-source community** for countless educational resources
- **DeepSeek** for pushing the boundaries of what's possible with Transformers

## 📬 Contact

Questions? Suggestions? Feel free to:
- Open an issue on GitHub
- Start a discussion in the Discussions tab
- Contribute to the wiki

---

**Happy Learning! 🚀**

*"Attention Is All You Need" — and now you'll understand why!*