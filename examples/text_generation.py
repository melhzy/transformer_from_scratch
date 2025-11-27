"""
Example: Simple Text Generation with Transformer

This script demonstrates how to use the Transformer for autoregressive
text generation with different sampling strategies.
"""

import torch
import torch.nn.functional as F
from src.transformer import Transformer


def create_toy_dataset():
    """
    Create a simple toy dataset for demonstration.
    
    In practice, you'd use a real tokenizer and dataset.
    """
    # Simple vocabulary
    vocab = {
        '<PAD>': 0,
        '<SOS>': 1,  # Start of sequence
        '<EOS>': 2,  # End of sequence
        'the': 3,
        'cat': 4,
        'dog': 5,
        'sat': 6,
        'ran': 7,
        'on': 8,
        'in': 9,
        'mat': 10,
        'park': 11,
    }
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    return vocab, id_to_token


def decode_sequence(token_ids, id_to_token):
    """
    Convert token IDs back to text.
    
    Args:
        token_ids: List or tensor of token IDs
        id_to_token: Dictionary mapping IDs to tokens
    
    Returns:
        str: Decoded text
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    tokens = []
    for token_id in token_ids:
        if token_id == 2:  # EOS
            break
        if token_id not in [0, 1]:  # Skip PAD and SOS
            tokens.append(id_to_token.get(token_id, '<UNK>'))
    
    return ' '.join(tokens)


def generate_with_greedy(model, src, max_len=20):
    """
    Generate sequence using greedy decoding (always pick most likely token).
    
    Args:
        model: Transformer model
        src: Source sequence
        max_len: Maximum generation length
    
    Returns:
        torch.Tensor: Generated sequence
    """
    model.eval()
    device = src.device
    batch_size = src.size(0)
    
    # Encode source
    with torch.no_grad():
        memory = model.encode(src)
        
        # Start with SOS token
        tgt = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Get predictions
            decoder_output = model.decode(tgt, memory)
            logits = model.output_projection(decoder_output[:, -1, :])
            
            # Greedy: take argmax
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if EOS generated
            if (next_token == 2).all():
                break
    
    return tgt


def generate_with_sampling(model, src, max_len=20, temperature=1.0, top_k=None):
    """
    Generate sequence using sampling (stochastic decoding).
    
    Args:
        model: Transformer model
        src: Source sequence
        max_len: Maximum generation length
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k tokens
    
    Returns:
        torch.Tensor: Generated sequence
    """
    model.eval()
    device = src.device
    batch_size = src.size(0)
    
    with torch.no_grad():
        memory = model.encode(src)
        tgt = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            decoder_output = model.decode(tgt, memory)
            logits = model.output_projection(decoder_output[:, -1, :])
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if (next_token == 2).all():
                break
    
    return tgt


def main():
    """
    Main function demonstrating text generation.
    """
    print("=" * 70)
    print("Text Generation with Transformer")
    print("=" * 70)
    
    # Create toy vocabulary
    vocab, id_to_token = create_toy_dataset()
    vocab_size = len(vocab)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Vocabulary: {list(vocab.keys())}")
    
    # Create model
    config = {
        "src_vocab_size": vocab_size,
        "tgt_vocab_size": vocab_size,
        "d_model": 128,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "pad_idx": 0
    }
    
    print("\nCreating Transformer model...")
    model = Transformer(**config)
    model.eval()
    
    # Create source sequence: "the cat sat"
    src_tokens = [vocab['<SOS>'], vocab['the'], vocab['cat'], vocab['sat'], vocab['<EOS>']]
    src = torch.tensor([src_tokens])
    
    print(f"\nSource sequence: {decode_sequence(src_tokens, id_to_token)}")
    
    # Generate with greedy decoding
    print("\n" + "-" * 70)
    print("Greedy Decoding")
    print("-" * 70)
    
    generated_greedy = generate_with_greedy(model, src, max_len=15)
    print(f"Generated (greedy): {decode_sequence(generated_greedy[0], id_to_token)}")
    
    # Generate with sampling (temperature = 1.0)
    print("\n" + "-" * 70)
    print("Sampling (temperature = 1.0)")
    print("-" * 70)
    
    for i in range(3):
        generated = generate_with_sampling(model, src, max_len=15, temperature=1.0)
        print(f"Sample {i+1}: {decode_sequence(generated[0], id_to_token)}")
    
    # Generate with sampling (temperature = 0.5, more confident)
    print("\n" + "-" * 70)
    print("Sampling (temperature = 0.5, more confident)")
    print("-" * 70)
    
    for i in range(3):
        generated = generate_with_sampling(model, src, max_len=15, temperature=0.5)
        print(f"Sample {i+1}: {decode_sequence(generated[0], id_to_token)}")
    
    # Generate with sampling (temperature = 1.5, more diverse)
    print("\n" + "-" * 70)
    print("Sampling (temperature = 1.5, more diverse)")
    print("-" * 70)
    
    for i in range(3):
        generated = generate_with_sampling(model, src, max_len=15, temperature=1.5)
        print(f"Sample {i+1}: {decode_sequence(generated[0], id_to_token)}")
    
    # Generate with top-k sampling
    print("\n" + "-" * 70)
    print("Top-k Sampling (k=3)")
    print("-" * 70)
    
    for i in range(3):
        generated = generate_with_sampling(model, src, max_len=15, temperature=1.0, top_k=3)
        print(f"Sample {i+1}: {decode_sequence(generated[0], id_to_token)}")
    
    print("\n" + "=" * 70)
    print("Generation complete!")
    print("=" * 70)
    
    print("\nNote: The model is randomly initialized, so outputs are random.")
    print("With proper training, the model would generate coherent text!")


if __name__ == "__main__":
    main()
