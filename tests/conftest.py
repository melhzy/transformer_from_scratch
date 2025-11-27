"""
Test configuration and shared fixtures.
"""

import pytest
import torch
import random
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility in tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def device():
    """Get the device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Standard sequence length for tests."""
    return 10


@pytest.fixture
def d_model():
    """Standard model dimension for tests."""
    return 128


@pytest.fixture
def vocab_size():
    """Standard vocabulary size for tests."""
    return 1000
