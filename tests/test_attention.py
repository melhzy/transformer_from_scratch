"""
Unit tests for attention mechanisms.
"""

import pytest
import torch
from src.modules.attention import (
    ScaledDotProductAttention,
    create_padding_mask,
    create_look_ahead_mask,
    compute_attention_entropy
)
from src.modules.multi_head_attention import MultiHeadAttention


class TestScaledDotProductAttention:
    """Test suite for ScaledDotProductAttention."""
    
    @pytest.fixture
    def attention(self):
        """Create a ScaledDotProductAttention instance for testing."""
        return ScaledDotProductAttention(dropout=0.1)
    
    def test_forward_shape(self, attention):
        """Test that forward pass produces correct output shapes."""
        batch_size = 2
        n_heads = 4
        seq_len = 10
        d_k = 64
        
        Q = torch.randn(batch_size, n_heads, seq_len, d_k)
        K = torch.randn(batch_size, n_heads, seq_len, d_k)
        V = torch.randn(batch_size, n_heads, seq_len, d_k)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, n_heads, seq_len, d_k)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self, attention):
        """Test that attention weights sum to 1 across key dimension."""
        Q = torch.randn(1, 1, 5, 64)
        K = torch.randn(1, 1, 5, 64)
        V = torch.randn(1, 1, 5, 64)
        
        attention.dropout.p = 0.0  # Disable dropout for this test
        _, weights = attention(Q, K, V)
        
        # Sum over key dimension (last dimension)
        sums = weights.sum(dim=-1)
        
        # Should sum to 1.0 for each query
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    
    def test_with_mask(self, attention):
        """Test attention with masking."""
        Q = torch.randn(1, 1, 5, 64)
        K = torch.randn(1, 1, 5, 64)
        V = torch.randn(1, 1, 5, 64)
        
        # Create a mask that masks last 2 positions
        mask = torch.ones(1, 1, 5, 5)
        mask[:, :, :, -2:] = 0
        
        attention.dropout.p = 0.0
        _, weights = attention(Q, K, V, mask=mask)
        
        # Masked positions should have near-zero attention
        assert torch.allclose(weights[:, :, :, -2:], torch.zeros_like(weights[:, :, :, -2:]), atol=1e-6)
    
    def test_causal_mask(self, attention):
        """Test attention with causal (look-ahead) mask."""
        seq_len = 5
        Q = torch.randn(1, 1, seq_len, 64)
        K = torch.randn(1, 1, seq_len, 64)
        V = torch.randn(1, 1, seq_len, 64)
        
        # Create causal mask
        mask = create_look_ahead_mask(seq_len)
        
        attention.dropout.p = 0.0
        _, weights = attention(Q, K, V, mask=mask)
        
        # Upper triangle (future positions) should be zero
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(weights[0, 0, i, j], torch.tensor(0.0), atol=1e-6)
    
    def test_custom_scaling(self):
        """Test attention with custom scaling factor."""
        attention = ScaledDotProductAttention(scale=10.0)
        
        Q = torch.randn(1, 1, 5, 64)
        K = torch.randn(1, 1, 5, 64)
        V = torch.randn(1, 1, 5, 64)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (1, 1, 5, 64)
        assert weights.shape == (1, 1, 5, 5)
    
    def test_cross_attention(self, attention):
        """Test cross-attention with different sequence lengths."""
        batch_size = 2
        n_heads = 4
        seq_len_q = 8
        seq_len_k = 12
        d_k = 64
        
        Q = torch.randn(batch_size, n_heads, seq_len_q, d_k)
        K = torch.randn(batch_size, n_heads, seq_len_k, d_k)
        V = torch.randn(batch_size, n_heads, seq_len_k, d_k)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, n_heads, seq_len_q, d_k)
        assert weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention."""
    
    @pytest.fixture
    def mha(self):
        """Create a MultiHeadAttention instance for testing."""
        return MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)
    
    def test_initialization(self, mha):
        """Test that MHA initializes correctly."""
        assert mha.d_model == 512
        assert mha.n_heads == 8
        assert mha.d_k == 64  # 512 / 8
        assert mha.d_v == 64
    
    def test_invalid_d_model(self):
        """Test that error is raised for invalid d_model."""
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=513, n_heads=8)  # Not divisible by n_heads
    
    def test_forward_shape(self, mha):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        
        x = torch.randn(batch_size, seq_len, 512)
        
        output, weights = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, 512)
        assert weights.shape == (batch_size, 8, seq_len, seq_len)
    
    def test_self_attention(self, mha):
        """Test self-attention (Q=K=V)."""
        x = torch.randn(2, 10, 512)
        
        output, weights = mha(x, x, x)
        
        assert output.shape == x.shape
    
    def test_cross_attention(self, mha):
        """Test cross-attention (Q from one sequence, K,V from another)."""
        query = torch.randn(2, 8, 512)
        key_value = torch.randn(2, 12, 512)
        
        output, weights = mha(query, key_value, key_value)
        
        assert output.shape == (2, 8, 512)
        assert weights.shape == (2, 8, 8, 12)
    
    def test_with_mask(self, mha):
        """Test MHA with attention mask."""
        batch_size = 2
        seq_len = 10
        
        x = torch.randn(batch_size, seq_len, 512)
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, :, -2:] = 0  # Mask last 2 positions
        
        output, weights = mha(x, x, x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, 512)
    
    def test_gradient_flow(self, mha):
        """Test that gradients flow through MHA."""
        x = torch.randn(2, 10, 512, requires_grad=True)
        
        output, _ = mha(x, x, x)
        loss = output.sum()
        loss.backward()
        
        # Check that all parameters have gradients
        assert mha.W_q.weight.grad is not None
        assert mha.W_k.weight.grad is not None
        assert mha.W_v.weight.grad is not None
        assert mha.W_o.weight.grad is not None


class TestAttentionUtilities:
    """Test suite for attention utility functions."""
    
    def test_create_padding_mask(self):
        """Test creating padding mask."""
        mask = create_padding_mask(seq_len=10, padding_positions=[8, 9])
        
        assert mask.shape == (1, 1, 1, 10)
        assert mask[0, 0, 0, 8] == 0
        assert mask[0, 0, 0, 9] == 0
        assert mask[0, 0, 0, 7] == 1
    
    def test_create_look_ahead_mask(self):
        """Test creating causal mask."""
        seq_len = 4
        mask = create_look_ahead_mask(seq_len)
        
        assert mask.shape == (1, 1, seq_len, seq_len)
        
        # Check lower triangular pattern
        expected = torch.tensor([[
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.]
        ]])
        
        assert torch.allclose(mask.squeeze(), expected.squeeze())
    
    def test_compute_attention_entropy(self):
        """Test computing attention entropy."""
        # Create uniform attention (high entropy)
        uniform_weights = torch.ones(2, 4, 10, 10) / 10
        uniform_entropy = compute_attention_entropy(uniform_weights)
        
        # Create peaked attention (low entropy)
        peaked_weights = torch.zeros(2, 4, 10, 10)
        peaked_weights[:, :, :, 0] = 1.0  # All attention on first position
        peaked_entropy = compute_attention_entropy(peaked_weights)
        
        # Uniform should have higher entropy than peaked
        assert uniform_entropy.mean() > peaked_entropy.mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
