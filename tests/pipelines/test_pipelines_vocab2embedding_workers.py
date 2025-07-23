#!/usr/bin/env python3
"""
test_pipelines_vocab2embedding_workers.py

Focused tests for parallel processing (workers) functionality in vocab2embedding pipeline.
Designed to run fast in CI/CD environments without GPU access.
Tests the multi-worker coordination, sequential ordering, and resume capabilities.
"""

import json
import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Dict

# Force CPU-only mode for testing to avoid multiprocessing tensor serialization issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import numpy as np
import yaml

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.pipelines.vocab2embedding import (
    WorkerTask,
    ProcessingResult,
    find_missing_sequences,
    process_sequences_parallel,
    process_sequences_sequential,
    load_existing_records,
    verify_processed_sequence,
    sequence_processor_worker,
    Vocab2EmbeddingPipeline
)


class TestWorkerComponents(unittest.TestCase):
    """Test individual worker components and data structures."""
    
    def test_worker_task_creation(self):
        """Test WorkerTask data structure."""
        task = WorkerTask(
            seq_id=123,
            sequence="test sequence",
            config_path="/path/to/config.yaml",
            vocab_path="/path/to/vocab.jsonl",
            dynamic_w_max=16
        )
        
        self.assertEqual(task.seq_id, 123)
        self.assertEqual(task.sequence, "test sequence")
        self.assertEqual(task.config_path, "/path/to/config.yaml")
        self.assertEqual(task.vocab_path, "/path/to/vocab.jsonl")
        self.assertEqual(task.dynamic_w_max, 16)
    
    def test_processing_result_success(self):
        """Test ProcessingResult for successful processing."""
        result_data = {
            'soft_probabilities': torch.randn(5, 10),
            'seed_embeddings': torch.randn(5, 64),
            'contextual_embeddings': torch.randn(5, 64),
            'span_candidates': [(0, 1), (1, 3), (2, 5)],
            'sequence_length': 5,
            'num_candidates': 3
        }
        
        result = ProcessingResult(seq_id=42, success=True, result=result_data)
        
        self.assertEqual(result.seq_id, 42)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.result)
        self.assertIsNone(result.error)
        if result.result is not None:
            self.assertEqual(result.result['num_candidates'], 3)
    
    def test_processing_result_error(self):
        """Test ProcessingResult for failed processing."""
        result = ProcessingResult(
            seq_id=42, 
            success=False, 
            error="Mock processing error"
        )
        
        self.assertEqual(result.seq_id, 42)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Mock processing error")


class TestMissingSequenceDetection(unittest.TestCase):
    """Test the enhanced resume functionality with discontinuous completion detection."""
    
    def test_continuous_completion(self):
        """Test detection when sequences are completed continuously."""
        existing_records = {1: {}, 2: {}, 3: {}}
        missing = find_missing_sequences(5, existing_records)
        
        # Should detect sequences 4 and 5 as missing
        self.assertEqual(missing, [4, 5])
    
    def test_discontinuous_completion(self):
        """Test detection when sequences are completed discontinuously."""
        # Simulates: 1-3 done, 5-7 done, but 4 and 8-10 missing
        existing_records = {1: {}, 2: {}, 3: {}, 5: {}, 6: {}, 7: {}}
        missing = find_missing_sequences(10, existing_records)
        
        # Should detect sequences 4, 8, 9, 10 as missing
        self.assertEqual(missing, [4, 8, 9, 10])
    
    def test_empty_completion(self):
        """Test detection when no sequences are completed."""
        existing_records = {}
        missing = find_missing_sequences(3, existing_records)
        
        # Should detect all sequences as missing
        self.assertEqual(missing, [1, 2, 3])
    
    def test_complete_completion(self):
        """Test detection when all sequences are completed."""
        existing_records = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        missing = find_missing_sequences(5, existing_records)
        
        # Should detect no missing sequences
        self.assertEqual(missing, [])
    
    def test_single_missing_sequence(self):
        """Test detection of single missing sequence in the middle."""
        existing_records = {1: {}, 2: {}, 4: {}, 5: {}}
        missing = find_missing_sequences(5, existing_records)
        
        # Should detect only sequence 3 as missing
        self.assertEqual(missing, [3])


class TestSequentialProcessing(unittest.TestCase):
    """Test sequential processing fallback and baseline functionality."""
    
    def setUp(self):
        """Set up minimal test configuration and data."""
        self.test_sequences = [
            "the cat",
            "hello world", 
            "test sequence"
        ]
        
        # Create minimal vocabulary for fast testing
        self.vocab_data = [
            {'piece': 'the', 'probability': 0.1},
            {'piece': 'cat', 'probability': 0.08},
            {'piece': 'hello', 'probability': 0.07},
            {'piece': 'world', 'probability': 0.06},
            {'piece': 'test', 'probability': 0.05},
            {'piece': 'sequence', 'probability': 0.04},
            {'piece': ' ', 'probability': 0.2},
            {'piece': 't', 'probability': 0.08},
            {'piece': 'e', 'probability': 0.12},
            {'piece': 'h', 'probability': 0.06},
            {'piece': 'o', 'probability': 0.05}
        ]
        
        # Minimal configuration for fast testing
        self.config_data = {
            'architecture': {
                'embed_dim': 32,  # Small for speed
                'conv_kernels': [3],  # Single kernel for speed
                'conv_dilations': [1],  # Single dilation for speed
                'dropout_rate': 0.0  # No dropout for deterministic testing
            },
            'span_generation': {
                'tau_vocab': 0.01,  # Lower threshold for more candidates
                'tau_comp': 1e-8,
                'w_max': 8  # Small for speed
            },
            'processing': {
                'device': 'cpu',  # Always CPU for CI/CD
                'batch_size': 32,
                'max_sequence_length': 128,
                'workers': 1
            },
            'numerical': {
                'epsilon': 1e-12,
                'max_piece_length': 8
            },
            'output': {
                'save_intermediate': False,  # Skip for speed
                'save_seed_embeddings': False,  # Skip for speed
                'save_json_metadata': True,  # Keep for verification
                'add_analysis': False,  # Skip for speed
                'save_soft_probabilities': False  # Skip for speed
            }
        }
    
    def create_test_pipeline(self, temp_dir: Path) -> Vocab2EmbeddingPipeline:
        """Create a test pipeline with temporary files."""
        # Create config file
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config_data, f)
        
        # Create vocab file
        vocab_file = temp_dir / "vocab.jsonl"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for item in self.vocab_data:
                json.dump(item, f)
                f.write('\n')
        
        # Initialize pipeline
        pipeline = Vocab2EmbeddingPipeline(str(config_file))
        pipeline.config['_config_path'] = str(config_file)
        pipeline.config['_vocab_path'] = str(vocab_file)
        pipeline.load_vocabulary(str(vocab_file))
        
        # Set dynamic w_max
        pipeline.w_max = pipeline.compute_dynamic_w_max(self.test_sequences)
        
        return pipeline
    
    def test_sequential_processing_basic(self):
        """Test basic sequential processing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path)
            
            # Create output directories
            output_dirs = {
                'json': temp_path / 'json',
                'seed': temp_path / 'seed',
                'context': temp_path / 'context',
                'soft_prob': temp_path / 'soft_prob'
            }
            for dir_path in output_dirs.values():
                dir_path.mkdir(exist_ok=True)
            
            # Process all sequences sequentially
            missing_seq_ids = [1, 2, 3]  # All sequences
            processed_count, error_count = process_sequences_sequential(
                self.test_sequences, missing_seq_ids, pipeline, output_dirs
            )
            
            # Verify results
            self.assertEqual(processed_count, 3)
            self.assertEqual(error_count, 0)
            
            # Verify output files exist
            context_files = list(output_dirs['context'].glob("context_emb_*.npy"))
            self.assertEqual(len(context_files), 3)
            
            # Verify correct sequence IDs
            seq_ids = [int(f.stem.split('_')[2]) for f in context_files]
            self.assertEqual(sorted(seq_ids), [1, 2, 3])
    
    def test_sequential_processing_partial(self):
        """Test sequential processing with only some sequences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path)
            
            # Create output directories
            output_dirs = {
                'json': temp_path / 'json',
                'seed': temp_path / 'seed', 
                'context': temp_path / 'context',
                'soft_prob': temp_path / 'soft_prob'
            }
            for dir_path in output_dirs.values():
                dir_path.mkdir(exist_ok=True)
            
            # Process only sequences 1 and 3 (simulating discontinuous resume)
            missing_seq_ids = [1, 3]
            processed_count, error_count = process_sequences_sequential(
                self.test_sequences, missing_seq_ids, pipeline, output_dirs
            )
            
            # Verify results
            self.assertEqual(processed_count, 2)
            self.assertEqual(error_count, 0)
            
            # Verify only sequences 1 and 3 were processed
            context_files = list(output_dirs['context'].glob("context_emb_*.npy"))
            self.assertEqual(len(context_files), 2)
            
            seq_ids = [int(f.stem.split('_')[2]) for f in context_files]
            self.assertEqual(sorted(seq_ids), [1, 3])


class TestParallelProcessing(unittest.TestCase):
    """Test parallel processing with multiple workers."""
    
    def setUp(self):
        """Set up test data for parallel processing."""
        # Use longer sequences to make parallel processing more meaningful
        self.test_sequences = [
            "the quick brown fox jumps over lazy dog",
            "hello world this is a longer test sequence",
            "parallel processing should handle multiple sequences",
            "each worker processes different sequences concurrently",
            "final sequence to ensure proper ordering"
        ]
        
        # Vocabulary suitable for test sequences
        self.vocab_data = [
            {'piece': 'the', 'probability': 0.08},
            {'piece': 'quick', 'probability': 0.02},
            {'piece': 'brown', 'probability': 0.02},
            {'piece': 'fox', 'probability': 0.015},
            {'piece': 'jumps', 'probability': 0.01},
            {'piece': 'over', 'probability': 0.02},
            {'piece': 'lazy', 'probability': 0.015},
            {'piece': 'dog', 'probability': 0.02},
            {'piece': 'hello', 'probability': 0.03},
            {'piece': 'world', 'probability': 0.025},
            {'piece': 'this', 'probability': 0.04},
            {'piece': 'is', 'probability': 0.05},
            {'piece': 'a', 'probability': 0.06},
            {'piece': 'longer', 'probability': 0.01},
            {'piece': 'test', 'probability': 0.02},
            {'piece': 'sequence', 'probability': 0.015},
            {'piece': 'parallel', 'probability': 0.008},
            {'piece': 'processing', 'probability': 0.008},
            {'piece': 'should', 'probability': 0.025},
            {'piece': 'handle', 'probability': 0.015},
            {'piece': 'multiple', 'probability': 0.01},
            {'piece': 'sequences', 'probability': 0.01},
            {'piece': 'each', 'probability': 0.02},
            {'piece': 'worker', 'probability': 0.008},
            {'piece': 'different', 'probability': 0.015},
            {'piece': 'concurrently', 'probability': 0.005},
            {'piece': 'final', 'probability': 0.015},
            {'piece': 'ensure', 'probability': 0.01},
            {'piece': 'proper', 'probability': 0.01},
            {'piece': 'ordering', 'probability': 0.008},
            {'piece': ' ', 'probability': 0.15},
            {'piece': 't', 'probability': 0.08},
            {'piece': 'e', 'probability': 0.10},
            {'piece': 'h', 'probability': 0.06},
            {'piece': 'o', 'probability': 0.05},
            {'piece': 'r', 'probability': 0.06},
            {'piece': 'n', 'probability': 0.06},
            {'piece': 's', 'probability': 0.06},
            {'piece': 'l', 'probability': 0.05}
        ]
        
        # Configuration optimized for parallel testing
        self.config_data = {
            'architecture': {
                'embed_dim': 64,  # Reasonable size for testing
                'conv_kernels': [3, 5],  # Two kernels for some complexity
                'conv_dilations': [1, 2],  # Two dilations
                'dropout_rate': 0.0  # Deterministic
            },
            'span_generation': {
                'tau_vocab': 0.005,  # Lower threshold for more candidates
                'tau_comp': 1e-8,
                'w_max': 16
            },
            'processing': {
                'device': 'cpu',  # Always CPU for CI/CD
                'batch_size': 32,
                'max_sequence_length': 256,
                'workers': 2  # Will be overridden in tests
            },
            'numerical': {
                'epsilon': 1e-12,
                'max_piece_length': 16
            },
            'output': {
                'save_intermediate': False,
                'save_seed_embeddings': False,  # Skip for speed
                'save_json_metadata': True,  # Keep for verification
                'add_analysis': False,
                'save_soft_probabilities': False
            }
        }
    
    def create_test_pipeline(self, temp_dir: Path, num_workers: int = 2) -> Vocab2EmbeddingPipeline:
        """Create a test pipeline with specified number of workers."""
        # Update config with specified workers
        config_data = self.config_data.copy()
        config_data['processing']['workers'] = num_workers
        
        # Create config file
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create vocab file
        vocab_file = temp_dir / "vocab.jsonl"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for item in self.vocab_data:
                json.dump(item, f)
                f.write('\n')
        
        # Initialize pipeline
        pipeline = Vocab2EmbeddingPipeline(str(config_file))
        pipeline.config['_config_path'] = str(config_file)
        pipeline.config['_vocab_path'] = str(vocab_file)
        pipeline.workers = num_workers
        pipeline.load_vocabulary(str(vocab_file))
        
        # Set dynamic w_max
        pipeline.w_max = pipeline.compute_dynamic_w_max(self.test_sequences)
        
        return pipeline
    
    def test_parallel_vs_sequential_equivalence(self):
        """Test that parallel processing produces same results as sequential."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create separate directories for sequential and parallel results
            seq_dir = temp_path / "sequential"
            par_dir = temp_path / "parallel"
            seq_dir.mkdir()
            par_dir.mkdir()
            
            # Sequential processing
            pipeline_seq = self.create_test_pipeline(seq_dir, num_workers=1)
            output_dirs_seq = {
                'json': seq_dir / 'json',
                'seed': seq_dir / 'seed',
                'context': seq_dir / 'context',
                'soft_prob': seq_dir / 'soft_prob'
            }
            for dir_path in output_dirs_seq.values():
                dir_path.mkdir(exist_ok=True)
            
            missing_seq_ids = [1, 2, 3, 4, 5]
            seq_processed, seq_errors = process_sequences_sequential(
                self.test_sequences, missing_seq_ids, pipeline_seq, output_dirs_seq
            )
            
            # Parallel processing
            pipeline_par = self.create_test_pipeline(par_dir, num_workers=2)
            output_dirs_par = {
                'json': par_dir / 'json',
                'seed': par_dir / 'seed',
                'context': par_dir / 'context',
                'soft_prob': par_dir / 'soft_prob'
            }
            for dir_path in output_dirs_par.values():
                dir_path.mkdir(exist_ok=True)
            
            par_processed, par_errors = process_sequences_parallel(
                self.test_sequences, missing_seq_ids, pipeline_par, 2, output_dirs_par
            )
            
            # Compare results
            self.assertEqual(seq_processed, par_processed)
            self.assertEqual(seq_errors, par_errors)
            self.assertEqual(seq_processed, 5)
            self.assertEqual(seq_errors, 0)
            
            # Verify same files were created
            seq_files = sorted(list(output_dirs_seq['context'].glob("context_emb_*.npy")))
            par_files = sorted(list(output_dirs_par['context'].glob("context_emb_*.npy")))
            
            self.assertEqual(len(seq_files), len(par_files))
            self.assertEqual(len(seq_files), 5)
            
            # Verify file names match (sequence IDs)
            seq_ids_seq = [int(f.stem.split('_')[2]) for f in seq_files]
            seq_ids_par = [int(f.stem.split('_')[2]) for f in par_files]
            
            self.assertEqual(sorted(seq_ids_seq), sorted(seq_ids_par))
            self.assertEqual(sorted(seq_ids_seq), [1, 2, 3, 4, 5])
    
    def test_parallel_processing_discontinuous(self):
        """Test parallel processing with discontinuous missing sequences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path, num_workers=2)
            
            # Create output directories
            output_dirs = {
                'json': temp_path / 'json',
                'seed': temp_path / 'seed',
                'context': temp_path / 'context',
                'soft_prob': temp_path / 'soft_prob'
            }
            for dir_path in output_dirs.values():
                dir_path.mkdir(exist_ok=True)
            
            # Simulate discontinuous missing sequences (1, 3, 5 missing, 2, 4 already done)
            missing_seq_ids = [1, 3, 5]
            processed_count, error_count = process_sequences_parallel(
                self.test_sequences, missing_seq_ids, pipeline, 2, output_dirs
            )
            
            # Verify results
            self.assertEqual(processed_count, 3)
            self.assertEqual(error_count, 0)
            
            # Verify correct files were created
            context_files = list(output_dirs['context'].glob("context_emb_*.npy"))
            self.assertEqual(len(context_files), 3)
            
            # Verify correct sequence IDs
            seq_ids = [int(f.stem.split('_')[2]) for f in context_files]
            self.assertEqual(sorted(seq_ids), [1, 3, 5])
    
    def test_parallel_processing_single_worker_fallback(self):
        """Test that single worker falls back to sequential processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path, num_workers=1)
            
            # Create output directories
            output_dirs = {
                'json': temp_path / 'json',
                'seed': temp_path / 'seed',
                'context': temp_path / 'context',
                'soft_prob': temp_path / 'soft_prob'
            }
            for dir_path in output_dirs.values():
                dir_path.mkdir(exist_ok=True)
            
            # Should fallback to sequential processing
            missing_seq_ids = [1, 2, 3]
            processed_count, error_count = process_sequences_parallel(
                self.test_sequences, missing_seq_ids, pipeline, 1, output_dirs
            )
            
            # Verify results (should work same as sequential)
            self.assertEqual(processed_count, 3)
            self.assertEqual(error_count, 0)
    
    def test_parallel_processing_multiple_workers(self):
        """Test parallel processing with different numbers of workers."""
        worker_counts = [2, 3]  # Test different worker counts
        
        for num_workers in worker_counts:
            with self.subTest(workers=num_workers):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    pipeline = self.create_test_pipeline(temp_path, num_workers=num_workers)
                    
                    # Create output directories
                    output_dirs = {
                        'json': temp_path / 'json',
                        'seed': temp_path / 'seed',
                        'context': temp_path / 'context',
                        'soft_prob': temp_path / 'soft_prob'
                    }
                    for dir_path in output_dirs.values():
                        dir_path.mkdir(exist_ok=True)
                    
                    # Process all sequences
                    missing_seq_ids = [1, 2, 3, 4, 5]
                    start_time = time.time()
                    processed_count, error_count = process_sequences_parallel(
                        self.test_sequences, missing_seq_ids, pipeline, num_workers, output_dirs
                    )
                    end_time = time.time()
                    
                    # Verify results
                    self.assertEqual(processed_count, 5)
                    self.assertEqual(error_count, 0)
                    
                    # Verify all sequences were processed
                    context_files = list(output_dirs['context'].glob("context_emb_*.npy"))
                    self.assertEqual(len(context_files), 5)
                    
                    # Verify sequence ordering is maintained
                    seq_ids = [int(f.stem.split('_')[2]) for f in context_files]
                    self.assertEqual(sorted(seq_ids), [1, 2, 3, 4, 5])
                    
                    print(f"Workers {num_workers}: {processed_count} sequences in {end_time - start_time:.2f}s")


class TestResumeCapabilities(unittest.TestCase):
    """Test enhanced resume capabilities with file verification."""
    
    def setUp(self):
        """Set up test environment for resume testing."""
        self.test_sequences = ["seq1", "seq2", "seq3", "seq4", "seq5"]
        
        # Minimal config for fast testing
        self.config_data = {
            'output': {
                'save_json_metadata': True,
                'save_seed_embeddings': False,
                'save_soft_probabilities': False
            }
        }
    
    def create_mock_output_files(self, output_dir: Path, completed_seq_ids: List[int]):
        """Create mock output files for specified sequence IDs."""
        # Create directories
        json_dir = output_dir / "json"
        context_dir = output_dir / "context"
        json_dir.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)
        
        for seq_id in completed_seq_ids:
            # Create context embedding file (always required)
            context_file = context_dir / f"context_emb_{seq_id:06d}.npy"
            mock_embedding = np.random.randn(10, 64).astype(np.float32)
            np.save(context_file, mock_embedding)
            
            # Create JSON metadata if enabled
            if self.config_data['output']['save_json_metadata']:
                json_file = json_dir / f"embedding_{seq_id:06d}.json"
                mock_metadata = {
                    'sequence_id': seq_id,
                    'sequence': f'sequence_{seq_id}',
                    'sequence_length': 10,
                    'num_candidates': 5,
                    'span_candidates': [(0, 2), (1, 3), (2, 5)]
                }
                with open(json_file, 'w') as f:
                    json.dump(mock_metadata, f)
    
    def test_load_existing_records_continuous(self):
        """Test loading existing records when sequences are completed continuously."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create mock files for sequences 1-3
            completed_seq_ids = [1, 2, 3]
            self.create_mock_output_files(output_dir, completed_seq_ids)
            
            # Load existing records
            existing_records, last_processed = load_existing_records(
                output_dir, self.config_data
            )
            
            # Verify results
            self.assertEqual(len(existing_records), 3)
            self.assertEqual(last_processed, 3)
            self.assertEqual(set(existing_records.keys()), {1, 2, 3})
    
    def test_load_existing_records_discontinuous(self):
        """Test loading existing records when sequences are completed discontinuously."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create mock files for sequences 1, 3, 5 (discontinuous)
            completed_seq_ids = [1, 3, 5]
            self.create_mock_output_files(output_dir, completed_seq_ids)
            
            # Load existing records
            existing_records, last_processed = load_existing_records(
                output_dir, self.config_data
            )
            
            # Verify results
            self.assertEqual(len(existing_records), 3)
            self.assertEqual(last_processed, 5)  # Should be max processed ID
            self.assertEqual(set(existing_records.keys()), {1, 3, 5})
    
    def test_verify_processed_sequence_complete(self):
        """Test sequence verification when all required files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create complete files for sequence 1
            self.create_mock_output_files(output_dir, [1])
            
            json_dir = output_dir / "json"
            seed_dir = output_dir / "seed"
            context_dir = output_dir / "context"
            soft_prob_dir = output_dir / "soft_prob"
            
            # Verify sequence is complete
            is_complete = verify_processed_sequence(
                json_dir, seed_dir, context_dir, soft_prob_dir, 1, self.config_data
            )
            
            self.assertTrue(is_complete)
    
    def test_verify_processed_sequence_incomplete(self):
        """Test sequence verification when required files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create only context file (missing JSON)
            context_dir = output_dir / "context"
            context_dir.mkdir(parents=True, exist_ok=True)
            
            context_file = context_dir / f"context_emb_{1:06d}.npy"
            mock_embedding = np.random.randn(10, 64).astype(np.float32)
            np.save(context_file, mock_embedding)
            
            json_dir = output_dir / "json"
            seed_dir = output_dir / "seed"
            soft_prob_dir = output_dir / "soft_prob"
            
            # Verify sequence is incomplete (missing JSON)
            is_complete = verify_processed_sequence(
                json_dir, seed_dir, context_dir, soft_prob_dir, 1, self.config_data
            )
            
            self.assertFalse(is_complete)
    
    def test_integrated_resume_workflow(self):
        """Test complete resume workflow with missing sequence detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create mock completed sequences: 1, 2, 4, 5 (missing 3)
            completed_seq_ids = [1, 2, 4, 5]
            self.create_mock_output_files(output_dir, completed_seq_ids)
            
            # Load existing records
            existing_records, last_processed = load_existing_records(
                output_dir, self.config_data
            )
            
            # Find missing sequences
            missing_seq_ids = find_missing_sequences(5, existing_records)
            
            # Verify results
            self.assertEqual(len(existing_records), 4)
            self.assertEqual(set(existing_records.keys()), {1, 2, 4, 5})
            self.assertEqual(missing_seq_ids, [3])  # Only sequence 3 missing
            self.assertEqual(last_processed, 5)


if __name__ == '__main__':
    # Suppress logging during tests to reduce noise
    logging.getLogger().setLevel(logging.WARNING)
    
    # Run tests with higher verbosity for CI/CD
    unittest.main(verbosity=2)
