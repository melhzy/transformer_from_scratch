"""
Test runner for all unit tests.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    # Run all tests with verbose output and coverage
    pytest.main([
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
