"""
Unit tests for the complete Transformer model.
"""

import pytest
import torch
import torch.nn as nn
from src.transformer import Transformer, count_parameters, get_model_size_mb


class TestTransformer:
    """Test suite for the complete Transformer model."""
    
    @pytest.fixture
    def transformer_config(self):
        """Standard configuration for testing."""
        return {
            "src_vocab_size": 1000,
            "tgt_vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "d_ff": 512,
            "dropout": 0.1,
            "max_seq_len": 100,
            "pad_idx": 0
        }
    
    @pytest.fixture
    def transformer(self, transformer_config):
        """Create a Transformer model for testing."""
        return Transformer(**transformer_config)
    
    def test_transformer_initialization(self, transformer, transformer_config):
        """Test that Transformer initializes correctly."""
        assert transformer.src_vocab_size == transformer_config["src_vocab_size"]
        assert transformer.tgt_vocab_size == transformer_config["tgt_vocab_size"]
        assert transformer.d_model == transformer_config["d_model"]
        assert transformer.n_heads == transformer_config["n_heads"]
    
    def test_forward_pass(self, transformer):
        """Test forward pass produces correct output shape."""
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8
        
        src = torch.randint(1, 100, (batch_size, src_seq_len))
        tgt = torch.randint(1, 100, (batch_size, tgt_seq_len))
        
        output = transformer(src, tgt)
        
        assert output.shape == (batch_size, tgt_seq_len, transformer.tgt_vocab_size)
    
    def test_encoder_only(self, transformer):
        """Test encoding without decoder."""
        batch_size = 2
        src_seq_len = 10
        
        src = torch.randint(1, 100, (batch_size, src_seq_len))
        memory = transformer.encode(src)
        
        assert memory.shape == (batch_size, src_seq_len, transformer.d_model)
    
    def test_decoder_with_memory(self, transformer):
        """Test decoder with pre-computed encoder memory."""
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8
        
        src = torch.randint(1, 100, (batch_size, src_seq_len))
        tgt = torch.randint(1, 100, (batch_size, tgt_seq_len))
        
        memory = transformer.encode(src)
        decoder_output = transformer.decode(tgt, memory)
        
        assert decoder_output.shape == (batch_size, tgt_seq_len, transformer.d_model)
    
    def test_mask_creation(self, transformer):
        """Test mask creation for padding."""
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8
        
        src = torch.randint(1, 100, (batch_size, src_seq_len))
        tgt = torch.randint(1, 100, (batch_size, tgt_seq_len))
        
        # Add padding tokens
        src[:, -2:] = 0
        tgt[:, -2:] = 0
        
        src_mask, tgt_mask, memory_mask = transformer.create_masks(src, tgt)
        
        assert src_mask is not None
        assert tgt_mask is not None
        assert memory_mask is not None
        
        # Check that padding positions are masked
        assert src_mask[0, 0, 0, -1] == 0
        assert tgt_mask[0, 0, -1, -1] == 0
    
    def test_generation(self, transformer):
        """Test autoregressive generation."""
        batch_size = 1
        src_seq_len = 10
        
        src = torch.randint(1, 100, (batch_size, src_seq_len))
        
        transformer.eval()
        with torch.no_grad():
            generated = transformer.generate(
                src,
                max_len=20,
                start_token=1,
                end_token=2,
                temperature=1.0
            )
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 20
        assert generated[0, 0] == 1  # Starts with start token
    
    def test_weight_tying(self):
        """Test that weight tying works correctly."""
        config = {
            "src_vocab_size": 1000,
            "tgt_vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "d_ff": 512,
            "tie_weights": True
        }
        
        transformer = Transformer(**config)
        
        # Check that embeddings and output projection share weights
        assert transformer.output_projection.weight is transformer.tgt_embedding.embedding.weight
    
    def test_no_weight_tying(self):
        """Test Transformer without weight tying."""
        config = {
            "src_vocab_size": 1000,
            "tgt_vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "d_ff": 512,
            "tie_weights": False
        }
        
        transformer = Transformer(**config)
        
        # Check that embeddings and output projection have separate weights
        assert transformer.output_projection.weight is not transformer.tgt_embedding.embedding.weight
    
    def test_parameter_count(self, transformer):
        """Test parameter counting utility."""
        param_info = count_parameters(transformer)
        
        assert "total" in param_info
        assert "trainable" in param_info
        assert param_info["total"] > 0
        assert param_info["trainable"] == param_info["total"]
    
    def test_model_size_calculation(self, transformer):
        """Test model size calculation."""
        size_mb = get_model_size_mb(transformer)
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
    
    def test_different_vocab_sizes(self):
        """Test with different source and target vocabulary sizes."""
        config = {
            "src_vocab_size": 1000,
            "tgt_vocab_size": 2000,
            "d_model": 128,
            "n_heads": 4,
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "d_ff": 512
        }
        
        transformer = Transformer(**config)
        
        batch_size = 2
        src = torch.randint(1, 1000, (batch_size, 10))
        tgt = torch.randint(1, 2000, (batch_size, 8))
        
        output = transformer(src, tgt)
        
        assert output.shape == (batch_size, 8, 2000)
    
    def test_gradient_flow(self, transformer):
        """Test that gradients flow through the model."""
        batch_size = 2
        src = torch.randint(1, 100, (batch_size, 10))
        tgt = torch.randint(1, 100, (batch_size, 8))
        
        output = transformer(src, tgt)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for key parameters
        assert transformer.encoder.layers[0].self_attention.W_q.weight.grad is not None
        assert transformer.decoder.layers[0].self_attention.W_q.weight.grad is not None
    
    def test_eval_mode(self, transformer):
        """Test switching between train and eval modes."""
        transformer.train()
        assert transformer.training
        
        transformer.eval()
        assert not transformer.training


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
