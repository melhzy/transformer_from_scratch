# Contributing to Transformer From Scratch

Thank you for your interest in contributing! This project aims to be an educational resource, so we welcome contributions that improve clarity, correctness, and usability.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title
   - Detailed description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, OS, etc.)

### Suggesting Features

We love new ideas! Please:

1. Open an issue with the "enhancement" label
2. Describe the feature and its benefits
3. Provide examples if possible
4. Discuss implementation approaches

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add comments and docstrings
   - Include tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check code style
   black src/ tests/ llm-fine-tune/
   flake8 src/ tests/ llm-fine-tune/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test updates
   - `refactor:` Code refactoring
   - `style:` Code style changes

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide clear description
   - Reference related issues
   - Explain your changes
   - Add screenshots if relevant

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for formatting
- Maximum line length: 100 characters
- Use type hints where appropriate

### Documentation Style

- Clear, concise explanations
- Math formulas in docstrings where relevant
- Examples in docstrings
- Comments explaining "why", not just "what"

Example:

```python
def scaled_attention(Q, K, V, scale):
    """
    Compute scaled dot-product attention.
    
    Mathematical formulation:
        Attention(Q, K, V) = softmax(Q·K^T / scale) · V
    
    Args:
        Q: Query tensor of shape (batch, seq_len, d_k)
        K: Key tensor of shape (batch, seq_len, d_k)
        V: Value tensor of shape (batch, seq_len, d_v)
        scale: Scaling factor (typically sqrt(d_k))
    
    Returns:
        Attention output of shape (batch, seq_len, d_v)
    
    Example:
        >>> Q = torch.randn(2, 10, 64)
        >>> K = torch.randn(2, 10, 64)
        >>> V = torch.randn(2, 10, 64)
        >>> output = scaled_attention(Q, K, V, scale=8.0)
    """
    # Implementation...
```

## Testing Guidelines

### Writing Tests

- Test files should match source files: `src/module.py` → `tests/test_module.py`
- Use descriptive test names
- Test edge cases
- Use fixtures for common setups

Example:

```python
def test_attention_weights_sum_to_one():
    """Test that attention weights sum to 1 across key dimension."""
    attention = ScaledDotProductAttention()
    Q, K, V = create_test_tensors()
    
    _, weights = attention(Q, K, V)
    sums = weights.sum(dim=-1)
    
    assert torch.allclose(sums, torch.ones_like(sums))
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_transformer.py

# Specific test function
pytest tests/test_transformer.py::test_forward_pass

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Documentation Guidelines

### Adding to README

If you add a significant feature:

1. Update relevant sections in README.md
2. Add usage examples
3. Update feature list

### Creating Examples

When adding example scripts:

1. Add to `examples/` directory
2. Include detailed comments
3. Update `examples/README.md`
4. Make it educational

### Writing Notebooks

For Jupyter notebooks:

1. Clear section structure
2. Mix explanation with code
3. Include visualizations
4. Make it interactive

## Types of Contributions We Welcome

### Code

- Bug fixes
- New features
- Performance improvements
- Better error handling
- Code refactoring

### Documentation

- Fixing typos
- Clarifying explanations
- Adding examples
- Improving docstrings
- Creating tutorials

### Tests

- Adding test cases
- Improving test coverage
- Testing edge cases
- Performance benchmarks

### Examples

- New use cases
- Visualization improvements
- Real-world applications
- Comparison studies

## Review Process

1. **Automated checks** - Code style, tests, etc.
2. **Manual review** - Code quality, documentation
3. **Discussion** - Feedback and suggestions
4. **Approval** - Merge when ready

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Provide constructive feedback
- Assume good intentions
- Keep discussions on-topic

## Questions?

- Open a discussion on GitHub
- Tag maintainers in issues
- Ask in community forums

---

Thank you for contributing to making Transformers more accessible to everyone! 🎓
