# LLM Fine-Tuning Examples 🚀

Comprehensive tutorials and examples for fine-tuning Large Language Models using modern techniques like LoRA, QLoRA, and instruction tuning.

**🎯 Goal**: Learn practical fine-tuning techniques used by Unsloth AI and the open-source community.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/)

---

## 📚 Tutorial Series

### Tutorial 1: Introduction to Fine-Tuning
**File**: [01_introduction_to_fine_tuning.ipynb](01_introduction_to_fine_tuning.ipynb)  
**Time**: ~45 minutes

**What You'll Learn**:
- What is fine-tuning and why it matters
- Different strategies: Full, PEFT, LoRA, QLoRA
- Parameter efficiency comparison
- When to use each approach
- LoRA mathematical foundations

**Prerequisites**:
- Complete [transformer-foundation/06_complete_transformer.ipynb](../transformer-foundation/06_complete_transformer.ipynb)
- Basic understanding of PyTorch

---

### Tutorial 2: LoRA Implementation (Coming Soon)
**File**: `02_lora_implementation.ipynb`  
**Time**: ~60 minutes

**What You'll Learn**:
- Implement LoRA from scratch
- Apply LoRA to attention layers
- Merge and unmerge weights
- Compare with full fine-tuning
- Memory profiling

---

### Tutorial 3: Data Preparation (Coming Soon)
**File**: `03_data_preparation.ipynb`  
**Time**: ~50 minutes

**What You'll Learn**:
- Dataset formats for fine-tuning
- Instruction-following format
- Tokenization strategies
- Data collation and batching
- Quality control

---

### Tutorial 4: Instruction Tuning (Coming Soon)
**File**: `04_instruction_tuning.ipynb`  
**Time**: ~75 minutes

**What You'll Learn**:
- Fine-tune for instruction following
- Training loop implementation
- Learning rate scheduling
- Checkpoint management
- Monitoring training

---

### Tutorial 5: Evaluation Metrics (Coming Soon)
**File**: `05_evaluation_metrics.ipynb`  
**Time**: ~45 minutes

**What You'll Learn**:
- Evaluate fine-tuned models
- Perplexity, ROUGE, BLEU scores
- Human evaluation setup
- Compare before/after
- A/B testing strategies

---

## 🛠️ Available Examples

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

## 📖 Learning Path

### Recommended Order:

1. **Start with Foundations**: Complete [transformer-foundation/](../transformer-foundation/) tutorials 1-6
2. **Understand Architecture**: Study [src/](../src/) implementation
3. **Read Papers**: Review [papers/](../papers/) - especially DeepSeek-R1
4. **Begin Fine-Tuning**: Work through tutorials 1-5 in this directory
5. **Practice**: Try examples with your own data

### Prerequisites:

- ✅ Complete transformer-foundation tutorials
- ✅ Understand attention mechanisms
- ✅ Familiar with PyTorch
- ✅ Basic knowledge of gradient descent

---

## 🔗 Key References

### Internal Resources:
- **[transformer-foundation/](../transformer-foundation/)** - Core Transformer tutorials
- **[src/](../src/)** - Our Transformer implementation
- **[papers/](../papers/)** - Research papers (Attention is All You Need, DeepSeek-R1)
- **[tests/](../tests/)** - Unit tests showing module usage

### External Inspirations:
- **Unsloth AI** - Optimized fine-tuning (referenced in tutorials)
- **Hugging Face PEFT** - Parameter-efficient fine-tuning library
- **QLoRA Paper** - Efficient quantized fine-tuning

---

## 🎯 What Makes This Different?

✨ **From Scratch Understanding**: Built on our custom Transformer implementation  
🔗 **Connected Learning**: References transformer-foundation tutorials  
📊 **Visual Learning**: Heavy visualization of concepts  
🎓 **Research-Backed**: References papers in ../papers/  
💡 **Practical Focus**: Real-world fine-tuning techniques  
🚀 **Colab-Ready**: Run everything in browser

---

## Further Reading

- See [transformer-foundation/](../transformer-foundation/) for Transformer fundamentals
- Check [tests/](../tests/) for unit tests showing module usage
- Read the [main README](../README.md) for architecture details
- Study [papers/](../papers/) for research background

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
