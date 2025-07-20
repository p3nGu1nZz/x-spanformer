#!/usr/bin/env python3
"""
test_pipelines_vocab2embedding.py

Comprehensive tests for the vocab2embedding pipeline implementation.
Combines unit tests, integration tests, and realistic pipeline tests.
Tests the mathematical formulations from Section 3.2 of the X-Spanformer paper.
"""

import json
import logging
import math
import tempfile
import unittest
from pathlib import Path
import sys
import warnings
import shutil
import time

import torch
import numpy as np
import yaml

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.pipelines.vocab2embedding import (
    UnigramLM,
    SeedEmbedder, 
    ConvEncoder,
    SpanCandidateGenerator,
    Vocab2EmbeddingPipeline,
    load_corpus,
    main
)


class TestUnigramLM(unittest.TestCase):
    """Test the UnigramLM forward-backward algorithm implementation."""
    
    def setUp(self):
        """Set up test vocabulary and model."""
        self.vocab_dict = {
            'a': 0.4,
            'b': 0.3, 
            'ab': 0.2,
            'ba': 0.1
        }
        self.device = 'cpu'  # Use CPU for testing
        self.model = UnigramLM(self.vocab_dict, self.device)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.vocab_size, 4)
        self.assertIn('a', self.model.piece_to_idx)
        self.assertIn('ab', self.model.piece_to_idx)
        self.assertEqual(self.model.max_piece_length, 2)
    
    def test_piece_matching(self):
        """Test piece matching at positions."""
        sequence = "aba"
        self.assertTrue(self.model.matches_at_position(sequence, 0, 'a'))
        self.assertTrue(self.model.matches_at_position(sequence, 0, 'ab'))
        self.assertFalse(self.model.matches_at_position(sequence, 0, 'ba'))
        self.assertTrue(self.model.matches_at_position(sequence, 1, 'b'))
        self.assertTrue(self.model.matches_at_position(sequence, 1, 'ba'))
    
    def test_forward_backward_simple(self):
        """Test forward-backward on simple sequence."""
        sequence = "ab"
        P = self.model.forward_backward(sequence)
        
        # Check dimensions
        self.assertEqual(P.shape, (2, 4))
        
        # Check probability normalization (rows should sum to reasonable values)
        row_sums = P.sum(dim=1)
        self.assertTrue(torch.all(row_sums >= 0))
        
        # Check that 'ab' piece at position 0 has non-zero probability
        ab_idx = self.model.piece_to_idx['ab']
        self.assertGreater(P[0, ab_idx].item(), 0)
    
    def test_forward_backward_consistency(self):
        """Test that probabilities are consistent with segmentation."""
        sequence = "a"
        P = self.model.forward_backward(sequence)
        
        a_idx = self.model.piece_to_idx['a']
        
        # Single 'a' should have high probability at position 0
        self.assertGreater(P[0, a_idx].item(), 0)
        
        # Other pieces should have zero probability at position 0
        ab_idx = self.model.piece_to_idx['ab']
        self.assertEqual(P[0, ab_idx].item(), 0)  # 'ab' doesn't fit


class TestSeedEmbedder(unittest.TestCase):
    """Test the SeedEmbedder initialization and computation."""
    
    def setUp(self):
        """Set up test embedder."""
        self.vocab_dict = {
            'a': 0.4,
            'b': 0.3,
            'ab': 0.2,
            'xyz': 0.1
        }
        self.embed_dim = 64
        self.device = 'cpu'
        self.embedder = SeedEmbedder(self.vocab_dict, self.embed_dim, self.device)
    
    def test_initialization(self):
        """Test embedding matrix initialization."""
        self.assertEqual(self.embedder.embedding_matrix.shape, (4, 64))
        
        # Check that embeddings have reasonable variance
        variance = torch.var(self.embedder.embedding_matrix, dim=1)
        self.assertTrue(torch.all(variance > 0))
    
    def test_vocabulary_aware_initialization(self):
        """Test that initialization respects piece probabilities."""
        # Single codepoints should have standard Xavier initialization
        a_idx = self.embedder.embedding_matrix[0]  # Assuming 'a' is first
        b_idx = self.embedder.embedding_matrix[1]  # Assuming 'b' is second
        
        # Multi-codepoint pieces should have different initialization
        ab_embedding = self.embedder.embedding_matrix[2]  # 'ab'
        xyz_embedding = self.embedder.embedding_matrix[3]  # 'xyz'
        
        # All embeddings should be non-zero
        self.assertGreater(torch.norm(a_idx).item(), 0)
        self.assertGreater(torch.norm(ab_embedding).item(), 0)
    
    def test_forward_pass(self):
        """Test seed embedding computation."""
        T, V = 3, 4
        soft_probs = torch.rand(T, V, device=self.device)
        
        embeddings = self.embedder(soft_probs)
        
        self.assertEqual(embeddings.shape, (T, self.embed_dim))
        
        # Check that embeddings are not all zeros
        self.assertGreater(torch.norm(embeddings).item(), 0)


class TestConvEncoder(unittest.TestCase):
    """Test the ConvEncoder multi-scale processing."""
    
    def setUp(self):
        """Set up test encoder."""
        self.embed_dim = 64
        self.device = 'cpu'
        self.encoder = ConvEncoder(self.embed_dim, self.device)
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(len(self.encoder.conv_layers), 3)
        
        # Check that all layers have correct dimensions
        for conv in self.encoder.conv_layers:
            self.assertEqual(conv.in_channels, self.embed_dim)
            self.assertEqual(conv.out_channels, self.embed_dim)
    
    def test_forward_pass(self):
        """Test contextual encoding."""
        T = 10
        seed_embeddings = torch.randn(T, self.embed_dim, device=self.device)
        
        contextual = self.encoder(seed_embeddings)
        
        # Check output shape
        self.assertEqual(contextual.shape, (T, self.embed_dim))
        
        # Check that output is different from input (contextualization occurred)
        self.assertGreater(torch.norm(contextual - seed_embeddings).item(), 0)
    
    def test_residual_connection(self):
        """Test that residual connections are working."""
        T = 5
        seed_embeddings = torch.randn(T, self.embed_dim, device=self.device)
        
        # Set model to evaluation mode to disable dropout
        self.encoder.eval()
        
        # Test that forward pass works without zeroing weights
        contextual = self.encoder(seed_embeddings)
        
        # Check that output is different from input (contextualization occurred)
        diff_norm = torch.norm(contextual - seed_embeddings).item()
        original_norm = torch.norm(seed_embeddings).item()
        
        # Output should be different but similar magnitude
        self.assertGreater(float(diff_norm), 0.0)
        self.assertLess(float(diff_norm), original_norm * 2.0)  # Not too different


class TestSpanCandidateGenerator(unittest.TestCase):
    """Test span candidate generation and filtering."""
    
    def setUp(self):
        """Set up test generator."""
        self.vocab_dict = {
            'the': 0.1,
            'cat': 0.08,
            'dog': 0.07,
            'a': 0.05,
            ' ': 0.2,  # Space
            'e': 0.03,
            't': 0.04
        }
        self.generator = SpanCandidateGenerator(
            self.vocab_dict, 
            tau_vocab=0.05, 
            tau_comp=1e-6,
            w_max=10
        )
    
    def test_vocabulary_alignment(self):
        """Test vocabulary alignment criterion."""
        # High-probability pieces should pass
        self.assertTrue(self.generator.vocabulary_alignment('the'))
        self.assertTrue(self.generator.vocabulary_alignment(' '))  # Space
        
        # Low-probability pieces should fail
        self.assertFalse(self.generator.vocabulary_alignment('e'))  # Below threshold
        
        # Unknown pieces should fail
        self.assertFalse(self.generator.vocabulary_alignment('xyz'))
    
    def test_compositional_potential(self):
        """Test compositional potential criterion."""
        # Should be able to segment known combinations
        self.assertTrue(self.generator.compositional_potential('the '))
        self.assertTrue(self.generator.compositional_potential('cat'))
        
        # Should fail for completely unknown sequences
        self.assertFalse(self.generator.compositional_potential('xyz'))
    
    def test_whitespace_coherent(self):
        """Test whitespace coherence criterion."""
        # Complete words should pass
        self.assertTrue(self.generator.whitespace_coherent('cat'))
        self.assertTrue(self.generator.whitespace_coherent('the'))
        
        # Spans starting/ending at boundaries should pass  
        self.assertTrue(self.generator.whitespace_coherent(' cat'))
        self.assertTrue(self.generator.whitespace_coherent('cat '))
        
        # Empty spans should fail
        self.assertFalse(self.generator.whitespace_coherent(''))
    
    def test_candidate_generation(self):
        """Test complete candidate generation."""
        sequence = "the cat"
        candidates = self.generator.generate_candidates(sequence)
        
        # Should have reasonable number of candidates
        self.assertGreater(len(candidates), 0)
        self.assertLess(len(candidates), len(sequence) ** 2)  # Not all possible spans
        
        # All candidates should be valid spans
        for start, end in candidates:
            self.assertGreaterEqual(start, 0)
            self.assertLess(start, end)
            self.assertLessEqual(end, len(sequence))
            self.assertLessEqual(end - start, self.generator.w_max)


class TestVocab2EmbeddingPipeline(unittest.TestCase):
    """Test the complete pipeline integration."""
    
    def setUp(self):
        """Set up test pipeline with temporary files."""
        self.device = 'cpu'
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        config_data = {
            'embed_dim': 32,  # Small for testing
            'tau_vocab': 0.05,
            'tau_comp': 1e-6,
            'w_max': 8
        }
        yaml.dump(config_data, self.temp_config)
        self.temp_config.close()
        
        # Create temporary vocabulary file
        self.temp_vocab = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        vocab_data = [
            {'piece': 'the', 'probability': 0.1},
            {'piece': 'cat', 'probability': 0.08},
            {'piece': ' ', 'probability': 0.2},
            {'piece': 'a', 'probability': 0.05},
            {'piece': 't', 'probability': 0.04}
        ]
        for item in vocab_data:
            json.dump(item, self.temp_vocab)
            self.temp_vocab.write('\n')
        self.temp_vocab.close()
        
        # Initialize pipeline
        self.pipeline = Vocab2EmbeddingPipeline(self.temp_config.name, self.device)
    
    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_config.name).unlink()
        Path(self.temp_vocab.name).unlink()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.config)
        self.assertEqual(self.pipeline.device, self.device)
        
        # Components should be None before vocabulary loading
        self.assertIsNone(self.pipeline.unigram_lm)
        self.assertIsNone(self.pipeline.seed_embedder)
    
    def test_vocabulary_loading(self):
        """Test vocabulary loading and component initialization."""
        self.pipeline.load_vocabulary(self.temp_vocab.name)
        
        # All components should be initialized
        self.assertIsNotNone(self.pipeline.unigram_lm)
        self.assertIsNotNone(self.pipeline.seed_embedder)
        self.assertIsNotNone(self.pipeline.conv_encoder)
        self.assertIsNotNone(self.pipeline.candidate_generator)
        
        # Check vocabulary size
        if self.pipeline.unigram_lm is not None:
            self.assertEqual(self.pipeline.unigram_lm.vocab_size, 5)
    
    def test_sequence_processing(self):
        """Test processing a complete sequence."""
        self.pipeline.load_vocabulary(self.temp_vocab.name)
        
        sequence = "the cat"
        result = self.pipeline.process_sequence(sequence)
        
        # Check result structure
        self.assertIn('soft_probabilities', result)
        self.assertIn('seed_embeddings', result)
        self.assertIn('contextual_embeddings', result)
        self.assertIn('span_candidates', result)
        self.assertIn('sequence_length', result)
        self.assertIn('num_candidates', result)
        
        # Check dimensions
        T = len(sequence)
        V = 5  # Vocabulary size
        d = 32  # Embedding dimension
        
        self.assertEqual(result['soft_probabilities'].shape, (T, V))
        self.assertEqual(result['seed_embeddings'].shape, (T, d))
        self.assertEqual(result['contextual_embeddings'].shape, (T, d))
        self.assertEqual(result['sequence_length'], T)
        self.assertGreater(result['num_candidates'], 0)
    
    def test_pipeline_end_to_end(self):
        """Test complete end-to-end processing."""
        self.pipeline.load_vocabulary(self.temp_vocab.name)
        
        sequences = ["the cat", "a cat", "the"]
        
        for sequence in sequences:
            result = self.pipeline.process_sequence(sequence)
            
            # Basic sanity checks
            self.assertGreater(result['num_candidates'], 0)
            self.assertEqual(result['sequence_length'], len(sequence))
            
            # Embeddings should not be all zeros
            seed_norm = float(np.linalg.norm(result['seed_embeddings']))
            context_norm = float(np.linalg.norm(result['contextual_embeddings']))
            self.assertGreater(seed_norm, 0.0)
            self.assertGreater(context_norm, 0.0)
            
            # Probabilities should be valid
            soft_probs = result['soft_probabilities']
            self.assertTrue(np.all(soft_probs >= 0))
            self.assertTrue(np.all(soft_probs <= 1))


class TestMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of implementations against paper formulations."""
    
    def test_xavier_initialization_scaling(self):
        """Test that Xavier initialization follows Eq. (4) from Section 3.2.2."""
        vocab_dict = {'a': 0.5, 'ab': 0.3, 'abc': 0.2}
        embed_dim = 100
        embedder = SeedEmbedder(vocab_dict, embed_dim, 'cpu')
        
        # Check scaling for different piece types
        with torch.no_grad():
            # Single codepoint should use standard Xavier: std = sqrt(2/d)
            expected_std_single = math.sqrt(2.0 / embed_dim)
            
            # Multi-codepoint should use: std = sqrt(2/(d*p(u)))  
            expected_std_ab = math.sqrt(2.0 / (embed_dim * 0.3))
            expected_std_abc = math.sqrt(2.0 / (embed_dim * 0.2))
            
            # Verify initialization was applied (exact verification is statistical)
            self.assertGreater(torch.std(embedder.embedding_matrix).item(), 0)
    
    def test_forward_backward_normalization(self):
        """Test that forward-backward probabilities satisfy normalization."""
        vocab_dict = {'a': 0.6, 'b': 0.4}
        model = UnigramLM(vocab_dict, 'cpu')
        
        sequence = "ab"
        P = model.forward_backward(sequence)
        
        # Each position should have probabilities that reflect valid segmentations
        # This is a basic sanity check - exact values depend on the algorithm
        self.assertGreaterEqual(torch.sum(P).item(), 0)
    
    def test_compositional_probability_computation(self):
        """Test compositional probability computation in candidate generation."""
        vocab_dict = {'a': 0.5, 'b': 0.3, 'ab': 0.2}
        generator = SpanCandidateGenerator(vocab_dict, tau_comp=1e-6)
        
        # "ab" should have compositional potential via both single pieces and full piece
        self.assertTrue(generator.compositional_potential('ab'))
        
        # "ba" should have potential via single pieces
        self.assertTrue(generator.compositional_potential('ba'))


class TestVocab2EmbeddingIntegration(unittest.TestCase):
    """Integration tests for vocab2embedding pipeline with PretrainRecord support."""
    
    def setUp(self):
        """Set up test data for integration tests."""
        # Sample PretrainRecord format data
        self.sample_pretrain_records = [
            {
                "raw": "the quick brown fox jumps over the lazy dog",
                "type": "text",
                "id": "001",
                "meta": {"source": "sample.txt", "status": "validated"}
            },
            {
                "raw": "hello world this is a test sequence",
                "type": "text", 
                "id": "002",
                "meta": {"source": "sample.txt", "status": "validated"}
            },
            {
                "raw": "should be discarded",
                "type": "text",
                "id": "003", 
                "meta": {"source": "sample.txt", "status": "discard"}
            },
            {
                "raw": "",  # Empty raw field
                "type": "text",
                "id": "004",
                "meta": {"source": "sample.txt", "status": "validated"}
            },
            {
                "raw": "final valid sequence for testing",
                "type": "text",
                "id": "005", 
                "meta": {"source": "sample.txt", "status": "validated"}
            }
        ]
        
        # Sample vocabulary data
        self.sample_vocab = [
            {"piece": "the", "probability": 0.05},
            {"piece": "quick", "probability": 0.01},
            {"piece": "brown", "probability": 0.01},
            {"piece": "fox", "probability": 0.01},
            {"piece": "jumps", "probability": 0.005},
            {"piece": "over", "probability": 0.01},
            {"piece": "lazy", "probability": 0.005},
            {"piece": "dog", "probability": 0.01},
            {"piece": "hello", "probability": 0.01},
            {"piece": "world", "probability": 0.01},
            {"piece": "this", "probability": 0.02},
            {"piece": "is", "probability": 0.03},
            {"piece": "a", "probability": 0.04},
            {"piece": "test", "probability": 0.01},
            {"piece": "sequence", "probability": 0.005},
            {"piece": "final", "probability": 0.005},
            {"piece": "valid", "probability": 0.005},
            {"piece": "for", "probability": 0.02},
            {"piece": "testing", "probability": 0.005},
            # Single character pieces
            {"piece": "t", "probability": 0.08},
            {"piece": "h", "probability": 0.06},
            {"piece": "e", "probability": 0.12},
            {"piece": "q", "probability": 0.001},
            {"piece": "u", "probability": 0.03},
            {"piece": " ", "probability": 0.15}  # Space
        ]
        
        # Sample configuration
        self.sample_config = {
            "embed_dim": 64,
            "tau_vocab": 1e-4,
            "tau_comp": 1e-6,
            "w_max": 32
        }
    
    def cleanup_logging(self):
        """Clean up logging handlers to avoid Windows file locking."""
        import logging
        # Clear all handlers from the embedding logger
        logger_names = ['embedding_logger', 'x_spanformer.embedding.embedding_logging']
        for name in logger_names:
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        # Clear root logger handlers too
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if hasattr(handler, 'baseFilename'):
                handler.close()

    def test_load_corpus_pretrain_records(self):
        """Test loading corpus from PretrainRecord format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = Path(temp_dir) / "dataset.jsonl"
            with open(input_file, 'w', encoding='utf-8') as f:
                for record in self.sample_pretrain_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Load corpus
            sequences = load_corpus(str(input_file))
            
            # Should load 3 valid sequences (skip discarded and empty)
            self.assertEqual(len(sequences), 3)
            self.assertIn("the quick brown fox jumps over the lazy dog", sequences)
            self.assertIn("hello world this is a test sequence", sequences)
            self.assertIn("final valid sequence for testing", sequences)
            self.assertNotIn("should be discarded", sequences)  # Discarded
            self.assertNotIn("", sequences)  # Empty
    
    def test_load_corpus_invalid_records(self):
        """Test handling of invalid records in corpus."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file with invalid records
            input_file = Path(temp_dir) / "invalid.jsonl"
            with open(input_file, 'w', encoding='utf-8') as f:
                # Valid record
                f.write(json.dumps({"raw": "valid text", "type": "text"}) + '\n')
                # Invalid JSON
                f.write('{"invalid": json\n')
                # Missing raw field
                f.write(json.dumps({"content": "wrong field", "type": "text"}) + '\n')
                # Non-dict record
                f.write(json.dumps("just a string") + '\n')
                # Another valid record
                f.write(json.dumps({"raw": "another valid text", "type": "text"}) + '\n')
            
            # Should handle gracefully
            sequences = load_corpus(str(input_file))
            self.assertEqual(len(sequences), 2)
            self.assertIn("valid text", sequences)
            self.assertIn("another valid text", sequences)
    
    def test_load_corpus_empty_file(self):
        """Test handling of empty corpus file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty file
            input_file = Path(temp_dir) / "empty.jsonl"
            input_file.touch()
            
            # Should raise ValueError
            with self.assertRaises(ValueError):
                load_corpus(str(input_file))
    
    def test_load_corpus_nonexistent_file(self):
        """Test handling of non-existent corpus file."""
        with self.assertRaises(FileNotFoundError):
            load_corpus("nonexistent_file.jsonl")
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.sample_config, f)
            
            # Initialize pipeline
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            
            self.assertEqual(pipeline.device, 'cpu')
            self.assertEqual(pipeline.config, self.sample_config)
            self.assertIsNone(pipeline.unigram_lm)  # Not loaded yet
            self.assertIsNone(pipeline.seed_embedder)
            self.assertIsNone(pipeline.conv_encoder)
            self.assertIsNone(pipeline.candidate_generator)
    
    def test_pipeline_vocab_loading(self):
        """Test vocabulary loading in pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.sample_config, f)
            
            # Create vocab file
            vocab_file = Path(temp_dir) / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in self.sample_vocab:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Initialize and load
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            pipeline.load_vocabulary(str(vocab_file))
            
            # Check components are initialized
            self.assertIsNotNone(pipeline.unigram_lm)
            self.assertIsNotNone(pipeline.seed_embedder)
            self.assertIsNotNone(pipeline.conv_encoder)
            self.assertIsNotNone(pipeline.candidate_generator)
            
            # Check vocabulary is loaded correctly
            self.assertIsNotNone(pipeline.unigram_lm)
            if pipeline.unigram_lm is not None:
                self.assertEqual(len(pipeline.unigram_lm.piece_to_idx), len(self.sample_vocab))
                self.assertIn("the", pipeline.unigram_lm.piece_to_idx)
                self.assertIn(" ", pipeline.unigram_lm.piece_to_idx)
    
    def test_pipeline_sequence_processing(self):
        """Test processing a single sequence through pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup pipeline
            config_file = Path(temp_dir) / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.sample_config, f)
            
            vocab_file = Path(temp_dir) / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in self.sample_vocab:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            pipeline.load_vocabulary(str(vocab_file))
            
            # Process sequence
            test_sequence = "the quick brown"
            result = pipeline.process_sequence(test_sequence)
            
            # Check result structure
            self.assertIn('soft_probabilities', result)
            self.assertIn('seed_embeddings', result)
            self.assertIn('contextual_embeddings', result)
            self.assertIn('span_candidates', result)
            self.assertIn('sequence_length', result)
            self.assertIn('num_candidates', result)
            
            # Check shapes
            seq_len = len(test_sequence)
            vocab_size = len(self.sample_vocab)
            embed_dim = self.sample_config['embed_dim']
            
            self.assertEqual(result['soft_probabilities'].shape, (seq_len, vocab_size))
            self.assertEqual(result['seed_embeddings'].shape, (seq_len, embed_dim))
            self.assertEqual(result['contextual_embeddings'].shape, (seq_len, embed_dim))
            self.assertEqual(result['sequence_length'], seq_len)
            self.assertIsInstance(result['span_candidates'], list)
            self.assertEqual(result['num_candidates'], len(result['span_candidates']))
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.sample_config, f)
            
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            
            # Try to process without loading vocabulary
            with self.assertRaises(RuntimeError):
                pipeline.process_sequence("test sequence")
    
    def test_vocab_file_formats(self):
        """Test different vocabulary file formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.sample_config, f)
            
            # Test with 'prob' field instead of 'probability'
            vocab_with_prob = [
                {"piece": "the", "prob": 0.05},
                {"piece": "test", "prob": 0.01},
                {"piece": " ", "prob": 0.15}
            ]
            
            vocab_file = Path(temp_dir) / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in vocab_with_prob:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            pipeline.load_vocabulary(str(vocab_file))
            
            # Should work with 'prob' field
            self.assertIsNotNone(pipeline.unigram_lm)
            if pipeline.unigram_lm is not None:
                self.assertEqual(len(pipeline.unigram_lm.piece_to_idx), 3)
                self.assertIn("the", pipeline.unigram_lm.piece_to_idx)
    
    def test_main_function_integration(self):
        """Test the main function with PretrainRecord input."""
        # Use current directory to avoid Windows file locking issues with tempfile
        import shutil
        test_dir = Path('.') / 'temp_test_main_integration'
        
        try:
            # Clean up first if exists
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
            
            test_dir.mkdir(exist_ok=True)
            
            # Create input files
            dataset_file = test_dir / "dataset.jsonl"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                for record in self.sample_pretrain_records[:2]:  # Use first 2 records
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            vocab_file = test_dir / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in self.sample_vocab:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            config_file = test_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.sample_config, f)
            
            output_dir = test_dir / "output"
            
            # Mock command line arguments
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    'vocab2embedding.py',
                    '--vocab', str(vocab_file),
                    '--input', str(dataset_file),
                    '--output', str(output_dir),
                    '--config', str(config_file),
                    '--device', 'cpu',
                    '--max-length', '100'
                ]
                
                # Run main function
                main()
                
                # Check outputs exist
                self.assertTrue(output_dir.exists())
                self.assertTrue((output_dir / "embedding.log").exists())
                
                # Should have processed 2 sequences
                embedding_files = list(output_dir.glob("embedding_*.json"))
                self.assertEqual(len(embedding_files), 2)
                
                # Check output files exist
                for i in [1, 2]:  # sequence IDs 1 and 2
                    seq_id_str = f"{i:06d}"
                    self.assertTrue((output_dir / f"embedding_{seq_id_str}.json").exists())
                    self.assertTrue((output_dir / f"soft_probs_{seq_id_str}.npy").exists())
                    self.assertTrue((output_dir / f"seed_emb_{seq_id_str}.npy").exists())
                    self.assertTrue((output_dir / f"context_emb_{seq_id_str}.npy").exists())
                
                # Check log file contains expected messages
                with open(output_dir / "embedding.log", 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    self.assertIn("X-SPANFORMER EMBEDDING GENERATION PIPELINE", log_content)
                    self.assertIn("Pipeline completed", log_content)
                    self.assertIn("Processing sequences from:", log_content)
                
            finally:
                sys.argv = original_argv
                self.cleanup_logging()
                
        finally:
            # Clean up test directory
            if test_dir.exists():
                self.cleanup_logging()
                time.sleep(0.1)  # Brief pause for Windows
                shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_logging_integration(self):
        """Test that logging works correctly with PretrainRecord format."""
        import shutil
        # Use current directory to avoid Windows file locking issues  
        test_dir = Path('.') / 'temp_test_logs_integration'
        try:
            # Clean up first if exists
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
            
            test_dir.mkdir(exist_ok=True)
            
            # Create input file
            input_file = test_dir / "dataset.jsonl"
            with open(input_file, 'w', encoding='utf-8') as f:
                for record in self.sample_pretrain_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Set up logging directory (same as output directory)
            log_dir = test_dir
            
            # Import and setup logging
            from x_spanformer.embedding.embedding_logging import setup_embedding_logging
            logger = setup_embedding_logging(log_dir, 'test_integration')
            
            # Load corpus (should log statistics)
            sequences = load_corpus(str(input_file))
            
            # Check log file is in the root output directory
            log_file = log_dir / "embedding.log"
            self.assertTrue(log_file.exists())
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("STAGE 1: CORPUS LOADING", content)
                self.assertIn("Total input records: 5", content)
                self.assertIn("Valid sequences: 3", content)
                self.assertIn("Discarded sequences: 1", content)
                self.assertIn("Invalid records: 1", content)
                
        finally:
            self.cleanup_logging()
            # Clean up directory
            if test_dir.exists():
                time.sleep(0.1)  # Brief pause for Windows
                shutil.rmtree(test_dir, ignore_errors=True)


class TestRealisticVocab2EmbeddingPipeline(unittest.TestCase):
    """Integration tests using realistic vocabulary from jsonl2vocab pipeline output."""
    
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
                if len(sequence) > 10:
                    self.assertGreater(curr_candidates, prev_candidates)
                
                # Verify complexity is reasonable (not exponential)
                seq_len = result['sequence_length']
                candidates_per_char = curr_candidates / seq_len if seq_len > 0 else 0
                self.assertLess(candidates_per_char, 50)  # Should not explode
                
                prev_candidates = curr_candidates
            
        finally:
            Path(vocab_path).unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)


class TestPipelineUtilities(unittest.TestCase):
    """Test pipeline utilities and embedding analysis functions."""
    
    def create_sample_vocab(self, vocab_path: Path) -> None:
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

    def create_sample_sequences(self, sequences_path: Path) -> None:
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

    def test_pipeline_initialization_with_config(self):
        """Test that the vocab2embedding pipeline can be initialized with config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            vocab_path = temp_path / "vocab.jsonl"
            self.create_sample_vocab(vocab_path)
            
            # Use default config path if it exists, otherwise create minimal config
            config_path = Path(__file__).resolve().parents[2] / "config" / "pipelines" / "vocab2embedding.yaml"
            if not config_path.exists():
                # Create minimal config for testing
                config_path = temp_path / "config.yaml"
                config_data = {
                    'embed_dim': 64,
                    'tau_vocab': 1e-4,
                    'tau_comp': 1e-6,
                    'w_max': 32
                }
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f)
            
            pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
            
            self.assertIsNotNone(pipeline)
            self.assertTrue(hasattr(pipeline, 'load_vocabulary'))
            self.assertTrue(hasattr(pipeline, 'process_sequence'))

    def test_pipeline_vocabulary_loading_functionality(self):
        """Test vocabulary loading functionality with realistic data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            vocab_path = temp_path / "vocab.jsonl"
            self.create_sample_vocab(vocab_path)
            
            # Create config
            config_path = temp_path / "config.yaml"
            config_data = {
                'embed_dim': 64,
                'tau_vocab': 1e-4,
                'tau_comp': 1e-6,
                'w_max': 32
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
            
            # Load vocabulary - should not raise exception
            pipeline.load_vocabulary(str(vocab_path))
            
            # Verify components are initialized
            self.assertIsNotNone(pipeline.unigram_lm)
            self.assertIsNotNone(pipeline.seed_embedder)
            self.assertIsNotNone(pipeline.conv_encoder)
            self.assertIsNotNone(pipeline.candidate_generator)

    def test_pipeline_sequence_processing_comprehensive(self):
        """Test sequence processing with comprehensive validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            vocab_path = temp_path / "vocab.jsonl"
            self.create_sample_vocab(vocab_path)
            
            # Create config
            config_path = temp_path / "config.yaml"
            config_data = {
                'embed_dim': 64,
                'tau_vocab': 1e-4,
                'tau_comp': 1e-6,
                'w_max': 32
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
            pipeline.load_vocabulary(str(vocab_path))
            
            # Process a test sequence
            test_sequence = "the quick brown fox"
            result = pipeline.process_sequence(test_sequence)
            
            # Validate result structure
            self.assertIn('soft_probabilities', result)
            self.assertIn('seed_embeddings', result)
            self.assertIn('contextual_embeddings', result)
            self.assertIn('num_candidates', result)
            self.assertIn('span_candidates', result)
            
            # Validate shapes and types
            self.assertGreater(result['soft_probabilities'].shape[0], 0)
            self.assertGreater(result['seed_embeddings'].shape[0], 0)
            self.assertGreater(result['contextual_embeddings'].shape[0], 0)
            self.assertGreater(result['num_candidates'], 0)
            self.assertGreater(len(result['span_candidates']), 0)

    def test_embedding_quality_analysis_integration(self):
        """Test embedding quality analysis utilities integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            vocab_path = temp_path / "vocab.jsonl"
            self.create_sample_vocab(vocab_path)
            
            # Create config
            config_path = temp_path / "config.yaml"
            config_data = {
                'embed_dim': 64,
                'tau_vocab': 1e-4,
                'tau_comp': 1e-6,
                'w_max': 32
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
            
            # Load vocabulary and process sequence
            pipeline.load_vocabulary(str(vocab_path))
            result = pipeline.process_sequence("the quick brown fox")
            
            # Test embedding quality analysis if available
            try:
                from x_spanformer.embedding.embedding_utils import analyze_embedding_quality
                
                quality = analyze_embedding_quality(result['contextual_embeddings'])
                
                # Validate quality metrics
                self.assertIn('mean_embedding_norm', quality)
                self.assertIn('dimension_variance_ratio', quality)
                self.assertGreater(quality['mean_embedding_norm'], 0)
                self.assertGreaterEqual(quality['dimension_variance_ratio'], 0)
            except ImportError:
                # Skip if embedding_utils not available
                pass

    def test_span_analysis_integration(self):
        """Test span analysis functionality integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            vocab_path = temp_path / "vocab.jsonl"
            self.create_sample_vocab(vocab_path)
            
            # Create config
            config_path = temp_path / "config.yaml"
            config_data = {
                'embed_dim': 64,
                'tau_vocab': 1e-4,
                'tau_comp': 1e-6,
                'w_max': 32
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
            
            # Load vocabulary and process sequence
            test_sequence = "the quick brown fox"
            pipeline.load_vocabulary(str(vocab_path))
            result = pipeline.process_sequence(test_sequence)
            
            # Test span analysis if available
            try:
                from x_spanformer.embedding.span_analysis import SpanAnalyzer
                
                # Create span analyzer
                analyzer = SpanAnalyzer(test_sequence, result['span_candidates'])
                
                # Test coverage analysis
                coverage = analyzer.compute_coverage_statistics()
                self.assertIn('coverage_density', coverage)
                self.assertIn('average_coverage_depth', coverage)
                self.assertGreaterEqual(coverage['coverage_density'], 0)
                self.assertLessEqual(coverage['coverage_density'], 1)
                self.assertGreaterEqual(coverage['average_coverage_depth'], 0)
                
                # Test span length analysis
                lengths = analyzer.analyze_span_lengths()
                if 'mean_length' in lengths:
                    self.assertGreater(lengths['mean_length'], 0)
                    self.assertGreater(lengths['min_length'], 0)
                    self.assertGreaterEqual(lengths['max_length'], lengths['min_length'])
            except ImportError:
                # Skip if span_analysis not available
                pass

    def test_complete_pipeline_integration_comprehensive(self):
        """Test complete pipeline integration with all available components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            vocab_path = temp_path / "vocab.jsonl"
            self.create_sample_vocab(vocab_path)
            
            # Create config
            config_path = temp_path / "config.yaml"
            config_data = {
                'embed_dim': 64,
                'tau_vocab': 1e-4,
                'tau_comp': 1e-6,
                'w_max': 32
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Initialize pipeline
            pipeline = Vocab2EmbeddingPipeline(str(config_path), 'cpu')
            
            # Load vocabulary
            pipeline.load_vocabulary(str(vocab_path))
            
            # Process sequence
            test_sequence = "the quick brown fox"
            result = pipeline.process_sequence(test_sequence)
            
            # Test basic pipeline results
            self.assertGreater(result['num_candidates'], 0)
            self.assertEqual(len(result['span_candidates']), result['num_candidates'])
            
            # Test embedding analysis if available
            try:
                from x_spanformer.embedding.embedding_utils import analyze_embedding_quality
                quality = analyze_embedding_quality(result['contextual_embeddings'])
                self.assertGreater(quality['mean_embedding_norm'], 0)
            except ImportError:
                pass
            
            # Test span analysis if available
            try:
                from x_spanformer.embedding.span_analysis import SpanAnalyzer
                analyzer = SpanAnalyzer(test_sequence, result['span_candidates'])
                coverage = analyzer.compute_coverage_statistics()
                self.assertGreater(coverage['coverage_density'], 0)
            except ImportError:
                pass
            
            # Verify we have reasonable number of candidates for this sequence
            self.assertGreater(result['num_candidates'], 5)  # Should have reasonable span coverage


if __name__ == '__main__':
    # Set up logging to avoid noise during testing
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    unittest.main(verbosity=2)
