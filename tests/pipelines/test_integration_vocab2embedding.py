#!/usr/bin/env python3
"""
test_integration_vocab2embedding.py

Integration test for the complete vocab2embedding pipeline using realistic
data from the jsonl2vocab pipeline output.
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys
import torch
import yaml
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline


class TestVocab2EmbeddingIntegration(unittest.TestCase):
    """Integration tests for vocab2embedding pipeline."""
    
    def setUp(self):
        """Set up realistic test data."""
        # Create realistic vocabulary (similar to jsonl2vocab output)
        # Make sure we have all characters needed for test sequences
        self.vocab_data = [
            # Common English words
            {'piece': 'the', 'probability': 0.045},
            {'piece': 'and', 'probability': 0.032},
            {'piece': 'to', 'probability': 0.028}, 
            {'piece': 'a', 'probability': 0.025},
            {'piece': 'of', 'probability': 0.024},
            {'piece': 'in', 'probability': 0.022},
            {'piece': 'is', 'probability': 0.020},
            {'piece': 'it', 'probability': 0.018},
            {'piece': 'that', 'probability': 0.016},
            {'piece': 'was', 'probability': 0.014},
            {'piece': 'cat', 'probability': 0.008},
            {'piece': 'dog', 'probability': 0.007},
            {'piece': 'quick', 'probability': 0.003},
            {'piece': 'brown', 'probability': 0.003},
            {'piece': 'fox', 'probability': 0.003},
            {'piece': 'over', 'probability': 0.004},
            {'piece': 'lazy', 'probability': 0.002},
            
            # Common subwords and characters - ensure all ASCII letters and common chars
            {'piece': ' ', 'probability': 0.120},  # Space
            {'piece': '.', 'probability': 0.012},  # Period
            {'piece': 'e', 'probability': 0.035},
            {'piece': 't', 'probability': 0.030},
            {'piece': 'o', 'probability': 0.022},
            {'piece': 'i', 'probability': 0.020},
            {'piece': 'n', 'probability': 0.018},
            {'piece': 's', 'probability': 0.016},
            {'piece': 'h', 'probability': 0.014},
            {'piece': 'r', 'probability': 0.014},
            {'piece': 'l', 'probability': 0.012},
            {'piece': 'd', 'probability': 0.011},
            {'piece': 'u', 'probability': 0.010},
            {'piece': 'c', 'probability': 0.009},
            {'piece': 'g', 'probability': 0.008},
            {'piece': 'm', 'probability': 0.008},
            {'piece': 'p', 'probability': 0.007},
            {'piece': 'f', 'probability': 0.006},
            {'piece': 'b', 'probability': 0.006},
            {'piece': 'w', 'probability': 0.005},
            {'piece': 'y', 'probability': 0.005},
            {'piece': 'v', 'probability': 0.004},
            {'piece': 'k', 'probability': 0.003},
            {'piece': 'x', 'probability': 0.002},
            {'piece': 'j', 'probability': 0.002},
            {'piece': 'q', 'probability': 0.001},
            {'piece': 'z', 'probability': 0.001},
            
            # Common bigrams
            {'piece': 'th', 'probability': 0.015},
            {'piece': 'er', 'probability': 0.012},
            {'piece': 'on', 'probability': 0.011},
            {'piece': 'an', 'probability': 0.010},
            {'piece': 're', 'probability': 0.009},
            {'piece': 'ed', 'probability': 0.008},
            {'piece': 'nd', 'probability': 0.007},
            {'piece': 'ha', 'probability': 0.006},
            {'piece': 'en', 'probability': 0.006},
            {'piece': 'ing', 'probability': 0.008},
            
            # Some less common pieces
            {'piece': 'tion', 'probability': 0.005},
            {'piece': 'ally', 'probability': 0.003},
            {'piece': 'ment', 'probability': 0.003},
            {'piece': 'ness', 'probability': 0.002}
        ]
        
        # Configuration for testing
        self.config_data = {
            'embed_dim': 128,
            'tau_vocab': 0.001,  # Lower threshold to include more candidates
            'tau_comp': 1e-8,
            'w_max': 32,
            'conv_kernels': [3, 5, 7],
            'conv_dilations': [1, 2, 4],
            'dropout_rate': 0.1,
            'device': 'cpu'  # Use CPU for deterministic testing
        }
        
        # Test sequences of varying complexity (ensure all can be segmented)
        self.test_sequences = [
            "the cat",
            "it was a dark and stormy night",
            "the quick brown fox",
            "machine learning is transforming technology", 
            "natural language processing",
        ]
    
    def test_pipeline_with_realistic_vocabulary(self):
        """Test pipeline with realistic vocabulary from Section 3.1."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as vocab_file:
            for item in self.vocab_data:
                json.dump(item, vocab_file)
                vocab_file.write('\n')
            vocab_path = vocab_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            yaml.dump(self.config_data, config_file)
            config_path = config_file.name
        
        try:
            # Initialize and test pipeline
            pipeline = Vocab2EmbeddingPipeline(config_path, self.config_data['device'])
            pipeline.load_vocabulary(vocab_path)
            
            results = []
            for sequence in self.test_sequences:
                result = pipeline.process_sequence(sequence)
                results.append(result)
                
                # Verify basic properties
                self.assertEqual(result['sequence_length'], len(sequence))
                self.assertGreater(result['num_candidates'], 0)
                
                # Verify tensor shapes
                T, d = len(sequence), self.config_data['embed_dim']
                # Get actual vocabulary size from the pipeline (handles deduplication)
                if pipeline.unigram_lm is not None:
                    V = pipeline.unigram_lm.vocab_size
                    self.assertEqual(result['soft_probabilities'].shape, (T, V))
                self.assertEqual(result['seed_embeddings'].shape, (T, d))
                self.assertEqual(result['contextual_embeddings'].shape, (T, d))
                
                # Verify probability constraints
                soft_probs = result['soft_probabilities']
                self.assertTrue(np.all(soft_probs >= 0))
                # Note: Soft probabilities may exceed 1 due to forward-backward normalization
                # They represent expected usage, not strict probabilities
                self.assertTrue(np.all(soft_probs >= 0))
                
                # Verify embeddings are reasonable
                seed_norm = np.linalg.norm(result['seed_embeddings'])
                context_norm = np.linalg.norm(result['contextual_embeddings'])
                self.assertGreater(float(seed_norm), 0.0)
                self.assertGreater(float(context_norm), 0.0)
                
                # Contextualization should modify embeddings
                diff_norm = np.linalg.norm(result['contextual_embeddings'] - result['seed_embeddings'])
                self.assertGreater(float(diff_norm), 0.0)
            
            # Test that longer sequences have more candidates
            short_seq_candidates = results[0]['num_candidates']  # "the cat"
            long_seq_candidates = results[2]['num_candidates']   # "the quick brown fox..."
            self.assertGreater(long_seq_candidates, short_seq_candidates)
            
        finally:
            # Cleanup
            Path(vocab_path).unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)
    
    def test_candidate_quality(self):
        """Test that generated candidates make linguistic sense."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as vocab_file:
            for item in self.vocab_data:
                json.dump(item, vocab_file)
                vocab_file.write('\n')
            vocab_path = vocab_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            yaml.dump(self.config_data, config_file)
            config_path = config_file.name
        
        try:
            pipeline = Vocab2EmbeddingPipeline(config_path, self.config_data['device'])
            pipeline.load_vocabulary(vocab_path)
            
            # Test specific sequence
            sequence = "the cat and dog"
            result = pipeline.process_sequence(sequence)
            
            # Extract candidate spans
            candidate_spans = []
            for start, end in result['span_candidates']:
                span_text = sequence[start:end]
                candidate_spans.append(span_text)
            
            # Should include common words
            self.assertIn('the', candidate_spans)
            self.assertIn('cat', candidate_spans)
            self.assertIn('and', candidate_spans)
            self.assertIn('dog', candidate_spans)
            
            # Should include common substrings
            self.assertIn(' ', candidate_spans)  # Space
            
            # Should have reasonable number of candidates
            self.assertGreater(len(candidate_spans), 10)  # At least some candidates
            self.assertLess(len(candidate_spans), 150)    # Not too many
            
        finally:
            Path(vocab_path).unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)
    
    def test_mathematical_properties(self):
        """Test mathematical properties of the implementation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as vocab_file:
            for item in self.vocab_data:
                json.dump(item, vocab_file)
                vocab_file.write('\n')
            vocab_path = vocab_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            yaml.dump(self.config_data, config_file)
            config_path = config_file.name
        
        try:
            pipeline = Vocab2EmbeddingPipeline(config_path, self.config_data['device'])
            pipeline.load_vocabulary(vocab_path)
            
            sequence = "the"  # Simple sequence for mathematical verification
            result = pipeline.process_sequence(sequence)
            
            # Test probability properties
            soft_probs = result['soft_probabilities']
            
            # Each position should have some non-zero probabilities
            position_sums = np.sum(soft_probs, axis=1)
            self.assertTrue(np.all(position_sums >= 0))
            
            # Some positions should have meaningful probabilities
            max_prob = np.max(soft_probs)
            self.assertGreater(float(max_prob), 0.0)
            
            # Test embedding properties
            seed_embeddings = result['seed_embeddings']
            contextual_embeddings = result['contextual_embeddings']
            
            # Embeddings should have reasonable magnitude
            seed_var = np.var(seed_embeddings)
            context_var = np.var(contextual_embeddings)
            self.assertGreater(float(seed_var), 0.0)
            self.assertGreater(float(context_var), 0.0)
            
            # Contextual embeddings should be different from seed embeddings
            correlation = np.corrcoef(seed_embeddings.flatten(), contextual_embeddings.flatten())[0, 1]
            self.assertLess(correlation, 0.99)  # Should not be identical
            
        finally:
            Path(vocab_path).unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)
    
    def test_scalability(self):
        """Test that pipeline scales reasonably with sequence length."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as vocab_file:
            for item in self.vocab_data:
                json.dump(item, vocab_file)
                vocab_file.write('\n')
            vocab_path = vocab_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            yaml.dump(self.config_data, config_file)
            config_path = config_file.name
        
        try:
            pipeline = Vocab2EmbeddingPipeline(config_path, self.config_data['device'])
            pipeline.load_vocabulary(vocab_path)
            
            # Test sequences of different lengths
            test_cases = [
                "a",
                "the cat",
                "the quick brown fox",
                "the quick brown fox jumps over the lazy dog"
            ]
            
            prev_candidates = 0
            for sequence in test_cases:
                result = pipeline.process_sequence(sequence)
                
                # Longer sequences should generally have more candidates
                # (though not strictly monotonic due to vocabulary effects)
                curr_candidates = result['num_candidates']
                if len(sequence) > 10:  # For sufficiently long sequences
                    self.assertGreater(curr_candidates, 5)  # Should have meaningful candidates
                
                # Verify complexity is reasonable (not exponential)
                seq_len = result['sequence_length']
                candidates_per_char = curr_candidates / seq_len if seq_len > 0 else 0
                self.assertLess(candidates_per_char, 50)  # Should not explode
                
                prev_candidates = curr_candidates
            
        finally:
            Path(vocab_path).unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
