# LLM Fine-Tuning Examples 🚀

Comprehensive tutorials and examples for fine-tuning Large Language Models using modern techniques like LoRA, QLoRA, and instruction tuning.

**🎯 Goal**: Learn practical fine-tuning techniques used by Unsloth AI and the open-source community.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/)

---

## 📚 Tutorial Series

### Tutorial 1: Introduction to Fine-Tuning ✅
**File**: [01_introduction_to_fine_tuning.ipynb](01_introduction_to_fine_tuning.ipynb)  
**Time**: ~45 minutes  
**Status**: Complete

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

### Tutorial 2: LoRA Implementation ✅
**File**: [02_lora_implementation.ipynb](02_lora_implementation.ipynb)  
**Time**: ~60 minutes  
**Status**: Complete

**What You'll Learn**:
- Implement LoRA from scratch
- Apply LoRA to multi-head attention
- Build complete LoRA encoder
- Weight merging for efficient inference
- Memory profiling and speed comparisons
- Training example with LoRA parameters

**Prerequisites**:
- Tutorial 1
- [transformer-foundation/03_multi_head_attention.ipynb](../transformer-foundation/03_multi_head_attention.ipynb)
- [transformer-foundation/04_feed_forward_networks.ipynb](../transformer-foundation/04_feed_forward_networks.ipynb)

---

### Tutorial 3: Data Preparation ✅
**File**: [03_data_preparation.ipynb](03_data_preparation.ipynb)  
**Time**: ~60 minutes  
**Status**: Complete

**What You'll Learn**:
- Dataset formats (prompt-completion, instruction, chat)
- Simple tokenizer implementation
- Dataset classes for different formats
- Data collation with padding
- Quality control and filtering
- Data visualization and statistics

**Prerequisites**:
- Tutorials 1-2
- [transformer-foundation/01_embeddings_and_positional_encoding.ipynb](../transformer-foundation/01_embeddings_and_positional_encoding.ipynb)

---

### Tutorial 4: Instruction Tuning ✅
**File**: [04_instruction_tuning.ipynb](04_instruction_tuning.ipynb)  
**Time**: ~75 minutes  
**Status**: Complete

**What You'll Learn**:
- Complete instruction tuning pipeline
- Training with LoRA adapters
- Learning rate scheduling (warmup + cosine)
- Gradient accumulation and clipping
- Text generation with fine-tuned model
- Checkpoint saving and loading (LoRA-only)

**Prerequisites**:
- Tutorials 1-3
- [transformer-foundation/06_complete_transformer.ipynb](../transformer-foundation/06_complete_transformer.ipynb)

---

### Tutorial 5: Evaluation Metrics ✅
**File**: [05_evaluation_metrics.ipynb](05_evaluation_metrics.ipynb)  
**Time**: ~60 minutes  
**Status**: Complete

**What You'll Learn**:
- Perplexity for language modeling
- BLEU scores for translation/generation
- ROUGE scores for summarization
- Exact Match and F1 for QA tasks
- Comprehensive evaluation suite
- Before/after comparison visualization

**Prerequisites**:
- Tutorials 1-4

---

##  Learning Path

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

## 💡 Tips for Success

1. **Follow the order**: Tutorials 1→2→3→4→5 build on each other
2. **Run in Colab**: All notebooks work out-of-the-box in Google Colab
3. **Experiment**: Modify parameters and see results
4. **Cross-reference**: Use links to transformer-foundation for deeper understanding
5. **Check papers/**: Read research papers for theoretical background

## 🔧 Troubleshooting

### Out of Memory in Colab

- Use smaller models (reduce `d_model`, `n_layers`)
- Reduce batch size
- Enable gradient checkpointing (Tutorial 4)
- Use QLoRA instead of LoRA (Tutorial 1)

### Slow Training

- Check GPU availability: `torch.cuda.is_available()`
- Reduce sequence length
- Use gradient accumulation instead of larger batches
- Enable mixed precision training

### Import Errors

- Make sure repo is cloned: Colab setup cells handle this automatically
- Check Python version: Requires Python 3.8+
- Install missing packages: `!pip install <package>`

## 📚 Further Reading

- **Foundations**: [transformer-foundation/](../transformer-foundation/) for architecture deep-dive
- **Implementation**: [src/](../src/) for production code
- **Research**: [papers/](../papers/) for theoretical background
- **Testing**: [tests/](../tests/) for usage examples

## 🤝 Contributing

Found an issue or want to add content?

1. Fork the repository
2. Create a feature branch
3. Make your changes (tutorials, examples, docs)
4. Test thoroughly
5. Submit a pull request

## 📦 Archive

Older standalone Python examples have been moved to [archive/](archive/) as their functionality is now covered comprehensively in the tutorial notebooks.
