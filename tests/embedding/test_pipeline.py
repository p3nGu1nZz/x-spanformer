#!/usr/bin/env python3
"""
Test script for vocab2embedding pipeline and embedding utilities.
Creates sample data and runs the pipeline to verify everything works.
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys


def create_sample_vocab(vocab_path: Path) -> None:
    """Create a sample vocabulary file for testing."""
    # Sample vocabulary with pieces and probabilities
    vocab_data = [
        {"piece": "the", "probability": 0.15, "frequency": 150},
        {"piece": "quick", "probability": 0.02, "frequency": 20},
        {"piece": "brown", "probability": 0.01, "frequency": 10},
        {"piece": "fox", "probability": 0.005, "frequency": 5},
        {"piece": " ", "probability": 0.2, "frequency": 200},
        {"piece": ".", "probability": 0.05, "frequency": 50},
        {"piece": "t", "probability": 0.08, "frequency": 80},
        {"piece": "h", "probability": 0.06, "frequency": 60},
        {"piece": "e", "probability": 0.12, "frequency": 120},
        {"piece": "qu", "probability": 0.015, "frequency": 15},
        {"piece": "br", "probability": 0.008, "frequency": 8},
        {"piece": "ow", "probability": 0.007, "frequency": 7},
        {"piece": "n", "probability": 0.07, "frequency": 70},
        {"piece": "f", "probability": 0.04, "frequency": 40},
        {"piece": "o", "probability": 0.09, "frequency": 90},
        {"piece": "x", "probability": 0.003, "frequency": 3},
    ]
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for entry in vocab_data:
            f.write(json.dumps(entry) + '\n')


def create_sample_sequences(sequences_path: Path) -> None:
    """Create sample input sequences for testing."""
    sequences = [
        {"content": "the quick brown fox"},
        {"content": "the fox"},
        {"content": "quick brown"},
        {"content": "the quick fox."},
        {"content": "brown fox."},
    ]
    
    with open(sequences_path, 'w', encoding='utf-8') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')


@pytest.fixture
def temp_test_files():
    """Create temporary test files for pipeline testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        vocab_path = temp_path / "vocab.jsonl"
        sequences_path = temp_path / "sequences.jsonl"
        output_dir = temp_path / "output"
        
        # Create sample data
        create_sample_vocab(vocab_path)
        create_sample_sequences(sequences_path)
        
        yield {
            'temp_path': temp_path,
            'vocab_path': vocab_path,
            'sequences_path': sequences_path,
            'output_dir': output_dir
        }


def test_vocab2embedding_pipeline_initialization(temp_test_files):
    """Test that the vocab2embedding pipeline can be initialized."""
    from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline
    
    config_path = Path(__file__).resolve().parents[2] / "config" / "pipelines" / "vocab2embedding.yaml"
    pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
    
    assert pipeline is not None
    assert hasattr(pipeline, 'load_vocabulary')
    assert hasattr(pipeline, 'process_sequence')


def test_vocab2embedding_pipeline_vocabulary_loading(temp_test_files):
    """Test vocabulary loading functionality."""
    from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline
    
    config_path = Path(__file__).resolve().parents[2] / "config" / "pipelines" / "vocab2embedding.yaml"
    pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
    
    # Load vocabulary
    pipeline.load_vocabulary(str(temp_test_files['vocab_path']))
    
    # Check that vocabulary was loaded (verify method doesn't raise exception)
    # The actual vocabulary attribute checking depends on pipeline implementation


def test_vocab2embedding_pipeline_sequence_processing(temp_test_files):
    """Test sequence processing with the pipeline."""
    from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline
    
    config_path = Path(__file__).resolve().parents[2] / "config" / "pipelines" / "vocab2embedding.yaml"
    pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
    
    # Load vocabulary
    pipeline.load_vocabulary(str(temp_test_files['vocab_path']))
    
    # Process a test sequence
    test_sequence = "the quick brown fox"
    result = pipeline.process_sequence(test_sequence)
    
    # Validate result structure
    assert 'soft_probabilities' in result
    assert 'seed_embeddings' in result
    assert 'contextual_embeddings' in result
    assert 'num_candidates' in result
    assert 'span_candidates' in result
    
    # Validate shapes and types
    assert result['soft_probabilities'].shape[0] > 0
    assert result['seed_embeddings'].shape[0] > 0
    assert result['contextual_embeddings'].shape[0] > 0
    assert result['num_candidates'] > 0
    assert len(result['span_candidates']) > 0


def test_embedding_quality_analysis(temp_test_files):
    """Test embedding quality analysis utilities."""
    from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline
    from x_spanformer.embedding.embedding_utils import analyze_embedding_quality
    
    config_path = Path(__file__).resolve().parents[2] / "config" / "pipelines" / "vocab2embedding.yaml"
    pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
    
    # Load vocabulary and process sequence
    pipeline.load_vocabulary(str(temp_test_files['vocab_path']))
    result = pipeline.process_sequence("the quick brown fox")
    
    # Analyze embedding quality
    quality = analyze_embedding_quality(result['contextual_embeddings'])
    
    # Validate quality metrics
    assert 'mean_embedding_norm' in quality
    assert 'dimension_variance_ratio' in quality
    assert quality['mean_embedding_norm'] > 0
    assert quality['dimension_variance_ratio'] >= 0


def test_span_analysis(temp_test_files):
    """Test span analysis functionality."""
    from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline
    from x_spanformer.embedding.span_analysis import SpanAnalyzer
    
    config_path = Path(__file__).resolve().parents[2] / "config" / "pipelines" / "vocab2embedding.yaml"
    pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
    
    # Load vocabulary and process sequence
    test_sequence = "the quick brown fox"
    pipeline.load_vocabulary(str(temp_test_files['vocab_path']))
    result = pipeline.process_sequence(test_sequence)
    
    # Create span analyzer
    analyzer = SpanAnalyzer(test_sequence, result['span_candidates'])
    
    # Test coverage analysis
    coverage = analyzer.compute_coverage_statistics()
    assert 'coverage_density' in coverage
    assert 'average_coverage_depth' in coverage
    assert 0 <= coverage['coverage_density'] <= 1
    assert coverage['average_coverage_depth'] >= 0
    
    # Test span length analysis
    lengths = analyzer.analyze_span_lengths()
    if 'mean_length' in lengths:
        assert lengths['mean_length'] > 0
        assert lengths['min_length'] > 0
        assert lengths['max_length'] >= lengths['min_length']


def test_complete_pipeline_integration(temp_test_files):
    """Test complete pipeline integration with all components."""
    from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline
    from x_spanformer.embedding.embedding_utils import analyze_embedding_quality
    from x_spanformer.embedding.span_analysis import SpanAnalyzer
    
    config_path = Path(__file__).resolve().parents[2] / "config" / "pipelines" / "vocab2embedding.yaml"
    
    # Initialize pipeline
    pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
    
    # Load vocabulary
    pipeline.load_vocabulary(str(temp_test_files['vocab_path']))
    
    # Process sequence
    test_sequence = "the quick brown fox"
    result = pipeline.process_sequence(test_sequence)
    
    # Test embedding analysis
    quality = analyze_embedding_quality(result['contextual_embeddings'])
    assert quality['mean_embedding_norm'] > 0
    
    # Test span analysis
    analyzer = SpanAnalyzer(test_sequence, result['span_candidates'])
    coverage = analyzer.compute_coverage_statistics()
    assert coverage['coverage_density'] > 0
    
    # Verify we have reasonable number of candidates
    assert result['num_candidates'] > 10  # Should have reasonable span coverage
    assert len(result['span_candidates']) == result['num_candidates']


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v"])
