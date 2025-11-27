# LLM Fine-Tuning Examples

This directory contains example scripts for fine-tuning and working with Transformer models, including visualization tools and generation strategies.

## Available Examples

### 1. Attention Visualization (`attention_visualization.py`)

Visualize attention patterns in encoder and decoder layers:

```bash
python llm-fine-tune/attention_visualization.py
```

**What it does:**
- Creates attention heatmaps for encoder self-attention
- Visualizes decoder masked self-attention (causal mask)
- Shows decoder cross-attention to encoder
- Generates PNG files with visualizations

**Output files:**
- `encoder_attention_head0.png` - Single head encoder attention
- `encoder_attention_all_heads.png` - All encoder heads
- `decoder_self_attention_head0.png` - Decoder self-attention with causal mask
- `decoder_cross_attention_head0.png` - Decoder-to-encoder attention

### 2. Text Generation (`text_generation.py`)

Demonstrate different decoding strategies:

```bash
python llm-fine-tune/text_generation.py
```

**What it does:**
- Shows greedy decoding (always pick most likely token)
- Demonstrates sampling with different temperatures
- Illustrates top-k sampling
- Compares deterministic vs stochastic generation

**Key concepts:**
- **Greedy decoding**: Most confident but potentially repetitive
- **Temperature sampling**: Controls randomness (low = confident, high = diverse)
- **Top-k sampling**: Restricts to k most likely tokens

### 3. Train Translation Model (Coming Soon)

Example of training a Transformer for machine translation:

```bash
python llm-fine-tune/train_translation.py
```

**Features (planned):**
- Data preparation and tokenization
- Training loop with loss computation
- Learning rate scheduling
- Model checkpointing
- Evaluation on validation set

## Running the Examples

### Prerequisites

Make sure you have installed the package:

```bash
pip install -e .
```

Or install dependencies:

```bash
pip install -r requirements.txt
```

### From Repository Root

```bash
# Run from the project root directory
python llm-fine-tune/attention_visualization.py
python llm-fine-tune/text_generation.py
```

### Tips for Learning

1. **Start with visualization**: Run `attention_visualization.py` to see attention patterns
2. **Understand generation**: Try `text_generation.py` with different parameters
3. **Read the code**: Each example is heavily commented
4. **Experiment**: Modify parameters and see what happens
5. **Compare outputs**: Try different temperatures, top-k values, etc.

## Customization

### Modify Model Configuration

You can easily adjust the model size in each example:

```python
config = {
    "src_vocab_size": 10000,
    "tgt_vocab_size": 10000,
    "d_model": 512,        # Try: 256, 768, 1024
    "n_heads": 8,          # Try: 4, 12, 16
    "n_encoder_layers": 6, # Try: 3, 12
    "n_decoder_layers": 6,
    "d_ff": 2048,          # Try: 1024, 4096
    "dropout": 0.1
}
```

### Add Your Own Examples

Create a new Python file in this directory following the pattern:

```python
"""
Example: Your Custom Use Case

Brief description of what this example demonstrates.
"""

import torch
from src.transformer import Transformer

def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()
```

## Further Reading

- See [transformer-foundation/](../transformer-foundation/) for interactive Jupyter notebooks
- Check [tests/](../tests/) for unit tests showing module usage
- Read the [main README](../README.md) for architecture details

## Troubleshooting

### Out of Memory

If you encounter OOM errors, try:
- Reducing `d_model` (e.g., from 512 to 256)
- Reducing number of layers
- Using smaller batch sizes

### Slow Execution

For faster experimentation:
- Use smaller models (fewer layers, smaller d_model)
- Reduce sequence lengths
- Use CPU for small models (GPU overhead can be significant)

## Contributing

Have an interesting example? Please contribute!

1. Create your example script
2. Add documentation
3. Test thoroughly
4. Submit a pull request
