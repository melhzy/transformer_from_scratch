# Quick Start Guide

Get up and running with the Transformer From Scratch project in minutes!

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch
```

### 2. Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## First Steps

### Run a Simple Example

```python
# Create a file: test_transformer.py
import torch
from src.transformer import Transformer

# Create a small Transformer
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=128,
    n_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_ff=512
)

# Create sample data
src = torch.randint(0, 1000, (2, 10))  # Batch of 2, length 10
tgt = torch.randint(0, 1000, (2, 8))   # Batch of 2, length 8

# Forward pass
output = model(src, tgt)
print(f"Output shape: {output.shape}")  # (2, 8, 1000)
```

Run it:
```bash
python test_transformer.py
```

### Explore Individual Modules

Each module can be run independently to see examples:

```bash
# Token Embeddings
python src/modules/embeddings.py

# Positional Encoding
python src/modules/positional_encoding.py

# Attention Mechanism
python src/modules/attention.py

# Multi-Head Attention
python src/modules/multi_head_attention.py

# Feed-Forward Network
python src/modules/feed_forward.py

# Encoder
python src/modules/encoder.py

# Decoder
python src/modules/decoder.py

# Full Transformer
python src/transformer.py
```

### Run Examples

```bash
# Visualize attention patterns
python llm-fine-tune/attention_visualization.py

# Try text generation
python llm-fine-tune/text_generation.py
```

### Open Jupyter Notebooks

```bash
# Launch Jupyter Lab
jupyter lab

# Navigate to transformer-foundation/ directory
# Open: 01_embeddings_and_positional_encoding.ipynb
```

## Learning Path

### For Complete Beginners

1. **Start with README.md** - Get overview of architecture
2. **Run individual modules** - See each component in action
3. **Read module docstrings** - Understand the theory
4. **Modify parameters** - Experiment and learn
5. **Open notebooks** - Interactive learning

### For Intermediate Users

1. **Study the code** - Read through each module
2. **Run tests** - See comprehensive usage examples
3. **Modify architecture** - Try different configurations
4. **Visualize attention** - Understand what model learns
5. **Train on toy tasks** - (Coming soon)

### For Advanced Users

1. **Implement extensions** - Add new features
2. **Optimize performance** - Profile and improve
3. **Contribute** - Share improvements
4. **Apply to real tasks** - Use in your projects

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_transformer.py -v
```

## Common Issues

### ImportError: No module named 'src'

**Solution:** Install the package in development mode:
```bash
pip install -e .
```

Or add to Python path:
```python
import sys
sys.path.append('path/to/transformer_from_scratch')
```

### CUDA Out of Memory

**Solution:** Use smaller model or CPU:
```python
model = Transformer(
    d_model=128,  # Smaller
    n_encoder_layers=2,  # Fewer layers
    ...
)
```

### PyTorch Not Installed

**Solution:** Install PyTorch:
```bash
# CPU version
pip install torch

# Or GPU version (check pytorch.org for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

### Learn More

- 📖 Read the [full README](README.md)
- 📓 Try the [Jupyter notebooks](transformer-foundation/)
- 🔬 Run the [fine-tuning examples](llm-fine-tune/)
- 🧪 Study the [tests](tests/)

### Experiment

- Modify model sizes
- Try different activation functions
- Visualize attention patterns
- Add new features

### Contribute

- Fix bugs
- Add examples
- Improve documentation
- Share your learnings

## Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper

### Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### Communities
- [PyTorch Discourse](https://discuss.pytorch.org/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)

## Getting Help

- 🐛 **Found a bug?** Open an issue on GitHub
- ❓ **Have a question?** Start a discussion
- 💡 **Have an idea?** Submit a feature request
- 🤝 **Want to contribute?** Check CONTRIBUTING.md

---

**Happy Learning! 🚀**

*Building Transformers from scratch is the best way to understand them!*
