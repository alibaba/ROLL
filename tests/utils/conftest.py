"""
Standalone pytest configuration for testing without torch dependency.

This conftest.py allows running unit tests in the utils directory
without requiring torch to be installed.
"""

import sys
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "distributed: mark test as requiring distributed environment")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and available dependencies."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_distributed = pytest.mark.skip(reason="Distributed environment not available")
    
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    for item in items:
        if not torch_available:
            for marker in item.iter_markers(name="gpu"):
                item.add_marker(pytest.mark.skip(reason="torch not available"))
