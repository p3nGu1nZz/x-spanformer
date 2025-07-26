import warnings
import pytest

# Suppress RuntimeWarning about unawaited coroutines from pdf2jsonl module
# This can occur when pytest imports modules containing async functions
warnings.filterwarnings("ignore", message="coroutine.*was never awaited", category=RuntimeWarning)

def pytest_addoption(parser):
    """Add custom command line option for enabling worker tests."""
    parser.addoption(
        "--workers", 
        action="store_true", 
        default=False, 
        help="Enable parallel worker tests (skipped by default for CI/CD)"
    )

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "workers: mark test as requiring --workers flag to run"
    )

def pytest_collection_modifyitems(config, items):
    """Skip worker tests unless --workers flag is provided."""
    if config.getoption("--workers"):
        # Workers flag provided, run all tests
        return
    
    skip_workers = pytest.mark.skip(reason="need --workers option to run")
    for item in items:
        if "workers" in item.keywords:
            item.add_marker(skip_workers)