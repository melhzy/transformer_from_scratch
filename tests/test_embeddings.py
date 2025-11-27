"""
Unit tests for embeddings and positional encoding modules.
"""

import pytest
import torch
import math
from src.modules.embeddings import TokenEmbedding, compute_embedding_similarity, get_nearest_neighbors
from src.modules.positional_encoding import PositionalEncoding, LearnedPositionalEncoding


class TestTokenEmbedding:
    """Test suite for TokenEmbedding."""
    
    @pytest.fixture
    def embedding(self):
        """Create a TokenEmbedding instance for testing."""
        return TokenEmbedding(vocab_size=1000, d_model=128, padding_idx=0)
    
    def test_initialization(self, embedding):
        """Test that embedding initializes correctly."""
        assert embedding.vocab_size == 1000
        assert embedding.d_model == 128
        assert embedding.padding_idx == 0
        assert embedding.scale == math.sqrt(128)
    
    def test_forward_shape(self, embedding):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        x = torch.randint(0, 1000, (batch_size, seq_len))
        
        output = embedding(x)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_scaling(self, embedding):
        """Test that embeddings are properly scaled."""
        x = torch.tensor([[1, 2, 3]])
        output = embedding(x)
        
        # Get raw embedding without scaling
        raw_emb = embedding.embedding(x)
        
        # Check that output is scaled version
        expected = raw_emb * embedding.scale
        assert torch.allclose(output, expected)
    
    def test_padding_idx(self):
        """Test that padding index embeddings are zero."""
        embedding = TokenEmbedding(vocab_size=100, d_model=64, padding_idx=0)
        
        # Padding embedding should be close to zero
        padding_emb = embedding.embedding.weight[0]
        assert torch.allclose(padding_emb, torch.zeros_like(padding_emb))
    
    def test_get_embedding_weight(self, embedding):
        """Test getting raw embedding weights."""
        weights = embedding.get_embedding_weight()
        
        assert weights.shape == (1000, 128)
        assert weights is embedding.embedding.weight
    
    def test_embedding_similarity(self, embedding):
        """Test computing similarity between embeddings."""
        similarity = compute_embedding_similarity(embedding, 10, 20)
        
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1
    
    def test_nearest_neighbors(self, embedding):
        """Test finding nearest neighbors."""
        neighbors = get_nearest_neighbors(embedding, token_idx=10, top_k=5)
        
        assert len(neighbors) == 5
        for token_idx, similarity in neighbors:
            assert isinstance(token_idx, int)
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1


class TestPositionalEncoding:
    """Test suite for PositionalEncoding."""
    
    @pytest.fixture
    def pos_encoding(self):
        """Create a PositionalEncoding instance for testing."""
        return PositionalEncoding(d_model=128, max_len=100, dropout=0.1)
    
    def test_initialization(self, pos_encoding):
        """Test that positional encoding initializes correctly."""
        assert pos_encoding.d_model == 128
        assert pos_encoding.max_len == 100
        assert pos_encoding.pe.shape == (1, 100, 128)
    
    def test_forward_shape(self, pos_encoding):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 128)
        
        output = pos_encoding(x)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_positional_encoding_values(self):
        """Test that positional encodings follow sine/cosine pattern."""
        pos_encoding = PositionalEncoding(d_model=128, max_len=100, dropout=0.0)
        
        pe = pos_encoding.pe[0]  # Get raw positional encodings
        
        # Check that even dimensions use sine
        # Check that odd dimensions use cosine
        # These should have different patterns
        even_dim = pe[:, 0]
        odd_dim = pe[:, 1]
        
        # They should not be identical
        assert not torch.allclose(even_dim, odd_dim)
    
    def test_sequence_length_error(self, pos_encoding):
        """Test that error is raised for sequences longer than max_len."""
        x = torch.randn(2, 150, 128)  # Longer than max_len=100
        
        with pytest.raises(ValueError):
            pos_encoding(x)
    
    def test_get_positional_encoding(self, pos_encoding):
        """Test getting positional encodings for specific length."""
        seq_len = 20
        pe = pos_encoding.get_positional_encoding(seq_len)
        
        assert pe.shape == (seq_len, 128)
    
    def test_different_positions_different_encodings(self, pos_encoding):
        """Test that different positions have different encodings."""
        pe = pos_encoding.get_positional_encoding(10)
        
        # Each position should have a unique encoding
        for i in range(9):
            for j in range(i + 1, 10):
                assert not torch.allclose(pe[i], pe[j])
    
    def test_dropout_in_training(self):
        """Test that dropout is applied in training mode."""
        pos_encoding = PositionalEncoding(d_model=128, max_len=100, dropout=0.5)
        pos_encoding.train()
        
        x = torch.randn(2, 10, 128)
        
        # Run multiple times and check for variation (due to dropout)
        output1 = pos_encoding(x)
        output2 = pos_encoding(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)
    
    def test_no_dropout_in_eval(self):
        """Test that dropout is not applied in eval mode."""
        pos_encoding = PositionalEncoding(d_model=128, max_len=100, dropout=0.5)
        pos_encoding.eval()
        
        x = torch.randn(2, 10, 128)
        
        # Run multiple times and check for consistency
        output1 = pos_encoding(x)
        output2 = pos_encoding(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestLearnedPositionalEncoding:
    """Test suite for LearnedPositionalEncoding."""
    
    @pytest.fixture
    def learned_pos_encoding(self):
        """Create a LearnedPositionalEncoding instance for testing."""
        return LearnedPositionalEncoding(d_model=128, max_len=100, dropout=0.1)
    
    def test_initialization(self, learned_pos_encoding):
        """Test that learned positional encoding initializes correctly."""
        assert learned_pos_encoding.d_model == 128
        assert learned_pos_encoding.max_len == 100
        assert learned_pos_encoding.position_embeddings.num_embeddings == 100
        assert learned_pos_encoding.position_embeddings.embedding_dim == 128
    
    def test_forward_shape(self, learned_pos_encoding):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 128)
        
        output = learned_pos_encoding(x)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_gradients_flow(self, learned_pos_encoding):
        """Test that gradients flow through learned embeddings."""
        x = torch.randn(2, 10, 128, requires_grad=True)
        output = learned_pos_encoding(x)
        loss = output.sum()
        loss.backward()
        
        # Check that position embeddings have gradients
        assert learned_pos_encoding.position_embeddings.weight.grad is not None
    
    def test_sequence_length_error(self, learned_pos_encoding):
        """Test error for sequences longer than max_len."""
        x = torch.randn(2, 150, 128)  # Longer than max_len=100
        
        with pytest.raises(ValueError):
            learned_pos_encoding(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
