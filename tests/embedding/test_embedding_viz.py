#!/usr/bin/env python3
"""
Test suite for embedding_viz.py

Comprehensive tests for visualization functionality including plot generation,
dimensionality reduction, and visualization utilities.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from x_spanformer.embedding.embedding_viz import (
    plot_soft_probabilities,
    plot_embedding_space,
    plot_candidate_distribution,
    create_span_heatmap,
    plot_embedding_comparison,
    create_comprehensive_visualization
)


class TestPlotSoftProbabilities:
    """Test soft probabilities visualization."""
    
    @pytest.fixture
    def sample_soft_probs(self):
        """Create sample soft probability data."""
        return np.random.rand(25, 100)  # 25 positions, 100 vocab pieces
    
    @pytest.fixture
    def sample_vocab_pieces(self):
        """Create sample vocabulary pieces."""
        return [f"piece_{i}" for i in range(100)]
    
    def test_plot_soft_probabilities_basic(self, sample_soft_probs):
        """Test basic soft probabilities plotting."""
        sequence = "the quick brown fox"
        
        fig = plot_soft_probabilities(sample_soft_probs, sequence)
        
        assert fig is not None
        assert len(fig.axes) >= 2  # Should have multiple subplots
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_soft_probabilities_with_vocab(self, sample_soft_probs, sample_vocab_pieces):
        """Test plotting with vocabulary pieces for labeling."""
        sequence = "test sequence"
        
        fig = plot_soft_probabilities(sample_soft_probs, sequence, 
                                    vocab_pieces=sample_vocab_pieces)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_soft_probabilities_save_path(self, sample_soft_probs, tmp_path):
        """Test saving plot to file."""
        sequence = "test"
        save_path = tmp_path / "soft_probs.png"
        
        fig = plot_soft_probabilities(sample_soft_probs, sequence, 
                                    save_path=str(save_path))
        
        assert fig is not None
        assert save_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_soft_probabilities_parameters(self, sample_soft_probs):
        """Test plotting with different parameters."""
        sequence = "test sequence with many words"
        
        fig = plot_soft_probabilities(sample_soft_probs, sequence,
                                    max_pieces=10, max_positions=15)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_soft_probabilities_edge_cases(self):
        """Test plotting with edge case inputs."""
        # Very small probability matrix
        small_probs = np.random.rand(2, 3)
        sequence = "ab"
        
        fig = plot_soft_probabilities(small_probs, sequence)
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # Single position
        single_pos = np.random.rand(1, 10)
        sequence = "a"
        
        fig = plot_soft_probabilities(single_pos, sequence)
        assert fig is not None
        
        plt.close(fig)


class TestPlotEmbeddingSpace:
    """Test embedding space visualization."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embedding data."""
        return np.random.rand(30, 256)
    
    @pytest.fixture
    def sample_span_candidates(self):
        """Create sample span candidates."""
        return [(0, 5), (10, 15), (20, 25)]
    
    def test_plot_embedding_space_pca(self, sample_embeddings):
        """Test PCA visualization of embedding space."""
        fig = plot_embedding_space(sample_embeddings, method='pca')
        
        assert fig is not None
        assert len(fig.axes) >= 1  # Main plot + potentially colorbar
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_embedding_space_tsne(self, sample_embeddings):
        """Test t-SNE visualization of embedding space."""
        fig = plot_embedding_space(sample_embeddings, method='tsne')
        
        assert fig is not None
        assert len(fig.axes) >= 1  # Main plot + potentially colorbar
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_embedding_space_with_spans(self, sample_embeddings, sample_span_candidates):
        """Test embedding visualization with span highlighting."""
        sequence = "the quick brown fox jumps"
        
        fig = plot_embedding_space(sample_embeddings, 
                                 span_candidates=sample_span_candidates,
                                 sequence=sequence)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_embedding_space_invalid_method(self, sample_embeddings):
        """Test embedding visualization with invalid reduction method."""
        with pytest.raises(ValueError, match="Unknown reduction method"):
            plot_embedding_space(sample_embeddings, method='invalid')
    
    def test_plot_embedding_space_save_path(self, sample_embeddings, tmp_path):
        """Test saving embedding space plot."""
        save_path = tmp_path / "embedding_space.png"
        
        fig = plot_embedding_space(sample_embeddings, save_path=str(save_path))
        
        assert fig is not None
        assert save_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_embedding_space_small_dataset(self):
        """Test with small dataset (edge case for t-SNE)."""
        small_embeddings = np.random.rand(5, 64)
        
        # Should handle small datasets gracefully
        fig = plot_embedding_space(small_embeddings, method='pca')
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotCandidateDistribution:
    """Test span candidate distribution visualization."""
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample span candidates."""
        return [(0, 3), (2, 5), (4, 8), (6, 10), (9, 15)]
    
    def test_plot_candidate_distribution_basic(self, sample_candidates):
        """Test basic candidate distribution plotting."""
        sequence_length = 19  # Length of "the quick brown fox"
        
        fig = plot_candidate_distribution(sample_candidates, sequence_length)
        
        assert fig is not None
        assert len(fig.axes) >= 2  # Should have multiple subplots
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_candidate_distribution_save_path(self, sample_candidates, tmp_path):
        """Test saving candidate distribution plot."""
        sequence_length = 13  # Length of "test sequence"
        save_path = tmp_path / "candidates.png"
        
        fig = plot_candidate_distribution(sample_candidates, sequence_length,
                                        save_path=str(save_path))
        
        assert fig is not None
        assert save_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_candidate_distribution_empty_candidates(self):
        """Test plotting with empty candidate list."""
        sequence_length = 4  # Length of "test"
        candidates = []
        
        with pytest.raises(ValueError, match="No span candidates"):
            fig = plot_candidate_distribution(candidates, sequence_length)
    
    def test_plot_candidate_distribution_single_candidate(self):
        """Test plotting with single candidate."""
        sequence_length = 4  # Length of "test"
        candidates = [(0, 4)]
        
        fig = plot_candidate_distribution(candidates, sequence_length)
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestCreateSpanHeatmap:
    """Test span heatmap creation."""
    
    @pytest.fixture
    def sample_embeddings_small(self):
        """Create small embedding matrix for heatmap testing."""
        return np.random.rand(15, 32)
    
    @pytest.fixture
    def sample_spans_small(self):
        """Create sample spans for small sequence."""
        return [(0, 3), (5, 8), (10, 13)]
    
    def test_create_span_heatmap_basic(self, sample_embeddings_small, sample_spans_small):
        """Test basic span heatmap creation."""
        sequence = "test sequence ok"
        
        fig = create_span_heatmap(sequence, sample_spans_small)
        
        assert fig is not None
        assert len(fig.axes) >= 1  # Should have multiple panels
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_create_span_heatmap_save_path(self, sample_embeddings_small, sample_spans_small, tmp_path):
        """Test saving span heatmap."""
        sequence = "test sequence"
        save_path = tmp_path / "heatmap.png"
        
        fig = create_span_heatmap(sequence, sample_spans_small,
                                save_path=str(save_path))
        
        assert fig is not None
        assert save_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_create_span_heatmap_no_spans(self, sample_embeddings_small):
        """Test heatmap creation with no spans."""
        sequence = "test"
        spans = []
        
        with pytest.raises(ValueError, match="No span candidates"):
            create_span_heatmap(sequence, spans)


class TestPlotEmbeddingComparison:
    """Test embedding comparison visualization."""
    
    @pytest.fixture
    def sample_embedding_results(self):
        """Create sample embedding results for comparison."""
        return {
            'seed_embeddings': np.random.rand(20, 128),
            'contextual_embeddings': np.random.rand(20, 128),
            'metadata': {
                'sequence': 'test comparison sequence',
                'span_candidates': [(0, 4), (5, 9), (10, 18)]
            }
        }
    
    def test_plot_embedding_comparison_basic(self, sample_embedding_results):
        """Test basic embedding comparison plotting."""
        contextual = sample_embedding_results['contextual_embeddings']
        seed = np.random.rand(*contextual.shape)  # Same shape as contextual
        
        fig = plot_embedding_comparison(seed, contextual)
        
        assert fig is not None
        assert len(fig.axes) >= 2  # Should have multiple comparison plots
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_embedding_comparison_save_path(self, sample_embedding_results, tmp_path):
        """Test saving embedding comparison plot."""
        save_path = tmp_path / "comparison.png"
        contextual = sample_embedding_results['contextual_embeddings']
        seed = np.random.rand(*contextual.shape)
        
        fig = plot_embedding_comparison(seed, contextual, save_path=str(save_path))
        
        assert fig is not None
        assert save_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_embedding_comparison_missing_data(self):
        """Test comparison with missing embedding data."""
        # Test with mismatched shapes or missing data
        seed_embeddings = np.random.rand(10, 64)
        # Create a contextual_embeddings with wrong shape to test error handling
        contextual_embeddings = np.random.rand(5, 32)  # Different shape
        
        with pytest.raises((ValueError, TypeError, AttributeError)):
            fig = plot_embedding_comparison(seed_embeddings, contextual_embeddings)


class TestCreateComprehensiveVisualization:
    """Test comprehensive visualization creation."""
    
    @pytest.fixture
    def comprehensive_results(self):
        """Create comprehensive results for visualization."""
        return {
            'soft_probabilities': np.random.rand(25, 50),
            'seed_embeddings': np.random.rand(25, 128),
            'contextual_embeddings': np.random.rand(25, 128),
            'metadata': {
                'sequence': 'the quick brown fox jumps over lazy dog',
                'span_candidates': [(0, 3), (4, 9), (10, 15), (16, 19), 
                                  (20, 25), (26, 30), (31, 35), (36, 39)],
                'vocab_pieces': [f'piece_{i}' for i in range(50)]
            }
        }
    
    def test_create_comprehensive_visualization_basic(self, comprehensive_results, tmp_path):
        """Test basic comprehensive visualization."""
        import json
        
        # Create a temporary result directory with proper structure
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        
        seq_id = 1
        contextual = comprehensive_results['contextual_embeddings']
        np.save(result_dir / f"contextual_embeddings_{seq_id:06d}.npy", contextual)
        
        metadata = {
            'sequence': 'comprehensive test',
            'span_candidates': [(0, 5), (6, 10), (11, 15)],
            'num_candidates': 3
        }
        with open(result_dir / f"embedding_{seq_id:06d}.json", 'w') as f:
            json.dump(metadata, f)
        
        figures = create_comprehensive_visualization(result_dir, seq_id)
        assert isinstance(figures, dict)
        assert len(figures) > 0
    
    def test_create_comprehensive_visualization_save_path(self, comprehensive_results, tmp_path):
        """Test saving comprehensive visualization."""
        # Mock the load_embedding_results function from embedding_utils
        with patch('x_spanformer.embedding.embedding_utils.load_embedding_results') as mock_load:
            mock_load.return_value = comprehensive_results
            
            result_dir = tmp_path / "results"
            result_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            figures = create_comprehensive_visualization(result_dir, sequence_id=0, output_dir=output_dir)
            
            assert isinstance(figures, dict)
            # Check that some files were created
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) > 0
    
    def test_create_comprehensive_visualization_save_individual(self, comprehensive_results, tmp_path):
        """Test saving individual component plots."""
        # Mock the load_embedding_results function
        with patch('x_spanformer.embedding.embedding_utils.load_embedding_results') as mock_load:
            mock_load.return_value = comprehensive_results
            
            result_dir = tmp_path / "results"
            result_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            figures = create_comprehensive_visualization(result_dir, sequence_id=0, output_dir=output_dir)
            
            # Should have multiple figure types
            assert len(figures) >= 2
            
            # Should have created multiple PNG files
            png_files = list(output_dir.glob("*.png"))
            assert len(png_files) >= 2
    
    def test_create_comprehensive_visualization_minimal_data(self):
        """Test comprehensive visualization with minimal data."""
        # Create minimal mock data
        minimal_results = {
            'metadata': {
                'sequence': 'hello',
                'span_candidates': [(0, 2), (1, 4)]
            },
            'soft_probabilities': None,
            'seed_embeddings': None,
            'contextual_embeddings': None
        }
        
        with patch('x_spanformer.embedding.embedding_utils.load_embedding_results') as mock_load:
            mock_load.return_value = minimal_results
            
            figures = create_comprehensive_visualization("/fake/path", sequence_id=0)
            
            # Should still create some figures even with minimal data
            assert isinstance(figures, dict)
            assert len(figures) >= 1  # At least candidate distribution should work


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_invalid_embedding_shapes(self):
        """Test handling of invalid embedding shapes."""
        # Wrong number of dimensions
        invalid_embeddings = np.random.rand(10)  # Should be 2D
        
        with pytest.raises((ValueError, IndexError)):
            plot_embedding_space(invalid_embeddings)
    
    def test_mismatched_data_sizes(self):
        """Test handling of mismatched data sizes."""
        # Test mismatched seed and contextual embeddings
        seed_embeddings = np.random.rand(10, 64)  # 10 positions
        contextual_embeddings = np.random.rand(15, 64)  # 15 positions (mismatch)
        
        # The function should raise an error due to mismatched sizes
        # matplotlib will raise ValueError when array dimensions don't match for plotting
        with pytest.raises((IndexError, ValueError)):
            plot_embedding_comparison(seed_embeddings, contextual_embeddings)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Test empty span candidates
        with pytest.raises(ValueError):
            plot_candidate_distribution([], sequence_length=10)
            
        # Test empty span heatmap
        with pytest.raises(ValueError):
            create_span_heatmap("hello", [])


class TestVisualizationUtilities:
    """Test visualization utility functions and parameters."""
    
    def test_color_schemes(self, tmp_path):
        """Test different color schemes in visualizations."""
        embeddings = np.random.rand(20, 64)
        
        # Test that different color schemes work
        fig1 = plot_embedding_space(embeddings, method='pca')
        assert fig1 is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig1)
    
    def test_figure_size_parameters(self):
        """Test that figure size parameters work correctly."""
        embeddings = np.random.rand(10, 32)
        
        # Test that different plotting functions produce different figure sizes
        fig1 = plot_embedding_space(embeddings, method='pca')
        fig2 = plot_soft_probabilities(np.random.rand(10, 50), "test sequence")
        
        assert fig1 is not None
        assert fig2 is not None
        
        # Figures should have different sizes based on their function
        size1 = fig1.get_size_inches()
        size2 = fig2.get_size_inches()
        
        # Both should have reasonable sizes
        assert size1[0] > 0 and size1[1] > 0
        assert size2[0] > 0 and size2[1] > 0
        
        import matplotlib.pyplot as plt
        plt.close(fig1)
        plt.close(fig2)


class TestVisualizationPerformance:
    """Test visualization performance with larger datasets."""
    
    def test_large_embedding_visualization(self):
        """Test visualization with large embedding matrices."""
        large_embeddings = np.random.rand(500, 128)
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        
        fig = plot_embedding_space(large_embeddings, method='pca')
        
        end_time = time.time()
        
        assert fig is not None
        assert end_time - start_time < 30  # Should complete within 30 seconds
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_many_spans_visualization(self):
        """Test visualization with many span candidates."""
        sequence = "this is a test sequence with many words and spans"
        sequence_length = len(sequence)
        
        # Generate many span candidates
        spans = []
        for start in range(sequence_length - 1):
            for length in range(1, min(6, sequence_length - start)):
                spans.append((start, start + length))
        
        # Should handle large number of spans gracefully
        fig = plot_candidate_distribution(spans, sequence_length)
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
        
        # Test span heatmap with many spans (should truncate)
        fig2 = create_span_heatmap(sequence, spans, max_spans=10)
        assert fig2 is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        plt.close(fig2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
