# Testing Guide

This document provides comprehensive information about the X-Spanformer testing framework, covering test organization, mathematical validation, and development workflows.

## Overview

X-Spanformer's test suite validates both the mathematical correctness of algorithms described in our paper (Sections 3.1 and 3.2) and the practical implementation across different components. The tests are organized into focused categories that align with the architectural components.

## Test Architecture

### Organizational Principles

1. **Mathematical Fidelity** - Core algorithms (EM convergence, forward-backward consistency, Xavier initialization) are rigorously tested
2. **Integration Validation** - End-to-end pipelines tested with both synthetic and real-world data
3. **Schema Consistency** - Pydantic models ensure data integrity across pipeline boundaries
4. **Modular Testing** - Component isolation enables focused debugging and validation

### Directory Structure

```
tests/
├── pipelines/          # Data processing pipeline validation
├── embedding/          # Section 3.2 embedding analysis utilities  
├── schema/             # Pydantic schema and data validation
├── agents/             # AI agent and content processing
├── core/               # Core utilities and configuration
└── conftest.py         # Shared pytest fixtures
```

## Algorithm Validation

### Section 3.1: Vocabulary Induction

**File:** `tests/pipelines/test_pipelines_jsonl2vocab.py`

Key mathematical properties validated:
- **EM Convergence**: Perplexity monotonic decrease across iterations
- **Viterbi Consistency**: Forward-backward algorithm mathematical consistency
- **Pruning Correctness**: OOV rate and perplexity thresholds properly applied
- **Whitespace Separation**: Strict separation between whitespace and content tokens

```python
def test_em_convergence():
    """Validate that EM iterations reduce perplexity monotonically."""
    # Implementation validates mathematical convergence properties
    
def test_viterbi_forward_backward_consistency():
    """Ensure forward-backward probabilities sum correctly."""
    # Validates Section 3.1 mathematical formulation
```

### Section 3.2: Seed Embeddings & Span Generation

**File:** `tests/pipelines/test_pipelines_vocab2embedding.py`

Mathematical validation includes:
- **Forward-Backward Soft Probabilities**: HMM-adapted algorithm correctness
- **Xavier Initialization**: Vocabulary-aware scaling validation  
- **Multi-Scale Convolutions**: Dilated convolution receptive field verification
- **Span Filtering**: Alignment, compositional potential, whitespace coherence

```python
class TestMathematicalCorrectness:
    def test_forward_backward_normalization(self):
        """Validate soft probability normalization."""
        # Ensures ∑_i P_{t,i} ≤ 1 for all positions t
        
    def test_xavier_initialization_scaling(self):
        """Test vocabulary-aware Xavier initialization."""
        # Validates frequency-scaled Gaussian initialization
        
    def test_compositional_probability_computation(self):
        """Test span compositional potential scoring."""
        # Validates vocabulary-informed filtering
```

### Integration Testing

**File:** `tests/embedding/test_pipeline.py`

Comprehensive end-to-end validation:
- Pipeline initialization and configuration loading
- Vocabulary loading and soft probability computation  
- Sequence processing with quality metrics
- Embedding analysis and span pattern validation

## Data Validation

### Schema Testing

**Directory:** `tests/schema/`

Validates Pydantic models ensuring:
- **Type Safety**: All fields properly typed and validated
- **Edge Cases**: Boundary conditions and invalid data handling
- **Serialization**: JSON round-trip consistency
- **Integration**: Schema compatibility across pipeline stages

### Synthetic Data Generation

Test utilities create controlled datasets:
- **Vocabulary Generation**: Realistic frequency distributions
- **Sequence Creation**: Diverse length and complexity patterns
- **Ground Truth**: Known-correct segmentations for validation

## Development Workflows

### Running Tests

```bash
# Complete test suite
python -m pytest tests/ -v

# Mathematical algorithm validation
python -m pytest tests/pipelines/test_pipelines_vocab2embedding.py::TestMathematicalCorrectness -v

# Embedding pipeline integration
python -m pytest tests/embedding/test_pipeline.py -v

# Schema validation
python -m pytest tests/schema/ -v

# Coverage reporting
python -m pytest tests/ --cov=x_spanformer --cov-report=html
```

### Performance Testing

```bash
# Pipeline performance benchmarks
python -m pytest tests/pipelines/ -v --benchmark-only

# Memory usage profiling
python -m pytest tests/embedding/ --memray
```

### Debugging Test Failures

1. **Mathematical Issues**: Check algorithm implementations against paper formulations
2. **Integration Failures**: Validate data flow between pipeline components
3. **Schema Errors**: Ensure Pydantic model compatibility
4. **Performance Degradation**: Profile with synthetic data scaling

## Test Data Management

### Fixtures and Utilities

- **Synthetic Vocabularies**: Controlled frequency distributions
- **Test Sequences**: Varied complexity and length patterns
- **Configuration Management**: YAML config loading and validation
- **Temporary File Handling**: Automatic cleanup and isolation

### Ground Truth Validation

Key validation datasets:
- **Mathematical Properties**: Known convergence behaviors
- **Algorithm Outputs**: Expected span distributions and quality metrics
- **Integration Results**: End-to-end pipeline consistency

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- **Pull Requests**: Full test suite validation
- **Main Branch**: Comprehensive testing with coverage reporting
- **Release Tags**: Performance benchmarking and compatibility validation

### Test Matrix

- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Ubuntu, macOS, Windows
- **Dependencies**: Core vs. optional (visualization, ML libraries)

## Contributing to Tests

### Adding New Tests

1. **Mathematical Validation**: Reference paper sections and formulations
2. **Integration Testing**: Cover new pipeline components and data flows  
3. **Edge Case Coverage**: Boundary conditions and error scenarios
4. **Performance Benchmarking**: Resource usage and scaling characteristics

### Test Quality Standards

- **Reproducibility**: Deterministic random seeds and controlled environments
- **Clarity**: Clear test names and comprehensive docstrings
- **Coverage**: Both positive and negative test cases
- **Performance**: Reasonable execution times for development workflows

This testing framework ensures X-Spanformer maintains mathematical fidelity to our paper while providing robust, production-ready implementations.
