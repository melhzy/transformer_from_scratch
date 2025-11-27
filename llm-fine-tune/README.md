# LLM Fine-Tuning Examples 🚀

Comprehensive **educational** tutorials for understanding fine-tuning from first principles. Learn how LoRA, QLoRA, and instruction tuning work by implementing them from scratch.

**🎯 Goal**: Build foundational understanding before using production libraries like [Unsloth AI](https://github.com/unslothai/unsloth).

**🔥 Philosophy**: 
- **Learn by Building**: Implement LoRA from scratch to truly understand it
- **Theory to Practice**: Mathematical foundations → Code → Real examples
- **Then Optimize**: After understanding, use [Unsloth](https://github.com/unslothai/notebooks) for production (2x faster, 30% less memory)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/transformer_from_scratch/blob/main/llm-fine-tune/)

> **Note**: These tutorials are **educational** (implement from scratch). For **production** fine-tuning, see [Unsloth AI's 100+ optimized notebooks](https://github.com/unslothai/notebooks) after completing these fundamentals.

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

### Complete Journey: Theory → Implementation → Production

**Phase 1: Foundation (Start Here)**
1. **Transformer Basics**: Complete [transformer-foundation/](../transformer-foundation/) tutorials 1-6
2. **Architecture Deep-Dive**: Study [src/](../src/) implementation
3. **Research Papers**: Review [papers/](../papers/) - Attention is All You Need, DeepSeek-R1

**Phase 2: Fine-Tuning Fundamentals (This Directory)**
4. **Tutorial 1**: Understand fine-tuning strategies and LoRA theory
5. **Tutorial 2**: Implement LoRA from scratch in PyTorch
6. **Tutorial 3**: Master data preparation and tokenization
7. **Tutorial 4**: Build complete training pipelines
8. **Tutorial 5**: Learn evaluation metrics and validation

**Phase 3: Production (Next Steps)**
9. **Unsloth Library**: Install and explore [unslothai/unsloth](https://github.com/unslothai/unsloth)
10. **Production Notebooks**: Use [100+ Unsloth examples](https://github.com/unslothai/notebooks) for real models
11. **Optimization**: Apply 2x speed improvements and 30% memory savings
12. **Deploy**: Build production pipelines with your fine-tuned models

### Why This Order?

✅ **Understand First**: Know *why* LoRA works before using black-box libraries  
✅ **Debug Better**: When Unsloth fails, you'll know how to fix it  
✅ **Optimize Smarter**: Understand trade-offs between speed and accuracy  
✅ **Innovate**: Create your own techniques based on solid foundations

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

### External Resources & Next Steps:

**For Production Use (After Completing These Tutorials)**:
- **[Unsloth AI](https://github.com/unslothai/unsloth)** - 2x faster fine-tuning with optimized kernels
- **[Unsloth Notebooks](https://github.com/unslothai/notebooks)** - 100+ production-ready fine-tuning examples
- **[Unsloth Documentation](https://docs.unsloth.ai/)** - Complete guides for real-world projects

**For Deeper Learning**:
- **[Hugging Face PEFT](https://github.com/huggingface/peft)** - Parameter-efficient fine-tuning library
- **[LoRA Paper](https://arxiv.org/abs/2106.09685)** - Original LoRA research
- **[QLoRA Paper](https://arxiv.org/abs/2305.14314)** - Efficient quantized fine-tuning

---

## 🎯 What Makes This Different?

### Educational vs. Production Approach

**Our Tutorials (Educational Foundation)**:
✨ **From Scratch Understanding**: Implement LoRA manually to understand how it works  
🔗 **Connected Learning**: Build on our custom Transformer implementation  
📊 **Visual Learning**: Heavy visualization of concepts and mathematics  
🎓 **Research-Backed**: Deep-dive into papers and theory  
💡 **No External Libraries**: Pure PyTorch implementation for learning  

**Unsloth AI (Production Ready)**:
🚀 **Optimized Library**: Use [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training  
⚡ **Memory Efficient**: 30% less VRAM usage with optimized kernels  
🎯 **Model-Specific**: Pre-configured for Llama, Gemma, Mistral, Qwen, etc.  
📦 **Ready-to-Deploy**: Production pipelines with [100+ notebooks](https://github.com/unslothai/notebooks)  

**Recommended Learning Path**:
1. **Start Here** (Educational): Understand LoRA theory and implementation from scratch
2. **Then Use Unsloth** (Production): Apply knowledge with optimized library for real projects

This tutorial series is **inspired by** and **complementary to** Unsloth AI's approach, focusing on foundational understanding before production usage.

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
