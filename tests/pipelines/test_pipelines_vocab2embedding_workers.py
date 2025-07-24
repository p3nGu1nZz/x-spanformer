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
import pytest
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
from x_spanformer.embedding.embedding_chunk import ChunkManager


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
                'embed_dim': 16,  # Reduced from 32 for speed
                'conv_kernels': [3],  # Single kernel for speed
                'conv_dilations': [1],  # Single dilation for speed
                'dropout_rate': 0.0  # No dropout for deterministic testing
            },
            'span_generation': {
                'tau_vocab': 0.1,  # Higher threshold for fewer candidates (faster)
                'tau_comp': 1e-6,  # Relaxed for speed
                'w_max': 4  # Reduced from 8 for speed
            },
            'processing': {
                'device': 'cpu',  # Always CPU for CI/CD
                'batch_size': 16,  # Reduced from 32
                'max_sequence_length': 64,  # Reduced from 128
                'workers': 1
            },
            'numerical': {
                'epsilon': 1e-8,  # Relaxed from 1e-12
                'max_piece_length': 4  # Reduced from 8
            },
            'output': {
                'save_intermediate': False,  # Skip for speed
                'save_seed_embeddings': False,  # Skip for speed
                'save_json_metadata': False,  # Skip for speed (changed from True)
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
            
            # Create output directory for chunks
            output_path = temp_path / 'embeddings'
            output_path.mkdir(exist_ok=True)
            
            # Create ChunkManager instance
            chunk_manager = ChunkManager(output_path, chunk_size=10)
            
            # Process all sequences sequentially
            missing_seq_ids = [1, 2, 3]  # All sequences
            processed_count, error_count = process_sequences_sequential(
                self.test_sequences, missing_seq_ids, pipeline, chunk_manager
            )
            
            # Verify results
            self.assertEqual(processed_count, 3)
            self.assertEqual(error_count, 0)
            
            # Verify chunk files exist
            chunk_files = list(chunk_manager.chunks_dir.glob("embeddings_*.npz"))
            self.assertGreater(len(chunk_files), 0)
            
            # Verify sequences were processed
            existing_sequences = chunk_manager.get_existing_sequences()
            self.assertEqual(sorted(existing_sequences), [1, 2, 3])
    
    def test_sequential_processing_partial(self):
        """Test sequential processing with only some sequences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path)
            
            # Create output directory for chunks
            output_path = temp_path / 'embeddings'
            output_path.mkdir(exist_ok=True)
            
            # Create ChunkManager instance
            chunk_manager = ChunkManager(output_path, chunk_size=10)
            
            # Process only sequences 1 and 3 (simulating discontinuous resume)
            missing_seq_ids = [1, 3]
            processed_count, error_count = process_sequences_sequential(
                self.test_sequences, missing_seq_ids, pipeline, chunk_manager
            )
            
            # Verify results
            self.assertEqual(processed_count, 2)
            self.assertEqual(error_count, 0)
            
            # Verify only sequences 1 and 3 were processed by checking chunk data
            # With chunked storage, we check which sequences actually have data
            chunk_data = chunk_manager.load_chunk(1)  # Sequences 1&3 go to chunk 1
            self.assertIsNotNone(chunk_data)
            if chunk_data is not None:  # Type guard for mypy
                actual_sequences = set(chunk_data.keys())
                self.assertEqual(actual_sequences, {1, 3})


class TestParallelProcessing(unittest.TestCase):
    """Test parallel processing with multiple workers."""
    
    def setUp(self):
        """Set up test data for parallel processing."""
        # Use shorter sequences for faster CI processing
        self.test_sequences = [
            "the quick brown fox",  # Shortened
            "hello world test",     # Shortened  
            "parallel processing"   # Shortened
        ]
        
        # Simplified vocabulary for faster testing
        self.vocab_data = [
            {'piece': 'the', 'probability': 0.2},
            {'piece': 'quick', 'probability': 0.1},
            {'piece': 'brown', 'probability': 0.1},
            {'piece': 'fox', 'probability': 0.1},
            {'piece': 'hello', 'probability': 0.1},
            {'piece': 'world', 'probability': 0.1},
            {'piece': 'test', 'probability': 0.1},
            {'piece': 'parallel', 'probability': 0.05},
            {'piece': 'processing', 'probability': 0.05},
            {'piece': ' ', 'probability': 0.1}
        ]
        
        # Configuration optimized for fast parallel testing
        self.config_data = {
            'architecture': {
                'embed_dim': 32,  # Reduced from 64 for speed
                'conv_kernels': [3],  # Reduced from [3, 5] for speed
                'conv_dilations': [1],  # Reduced from [1, 2] for speed
                'dropout_rate': 0.0  # Deterministic
            },
            'span_generation': {
                'tau_vocab': 0.05,  # Increased from 0.005 for fewer candidates (faster)
                'tau_comp': 1e-6,  # Relaxed from 1e-8
                'w_max': 8  # Reduced from 16
            },
            'processing': {
                'device': 'cpu',  # Always CPU for CI/CD
                'batch_size': 16,  # Reduced from 32
                'max_sequence_length': 128,  # Reduced from 256
                'workers': 2  # Will be overridden in tests
            },
            'numerical': {
                'epsilon': 1e-8,  # Relaxed from 1e-12
                'max_piece_length': 8  # Reduced from 16
            },
            'output': {
                'save_intermediate': False,
                'save_seed_embeddings': False,  # Skip for speed
                'save_json_metadata': False,  # Skip for speed (changed from True)
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
            seq_dir = temp_path / "sequential" / "embeddings"
            par_dir = temp_path / "parallel" / "embeddings"
            seq_dir.mkdir(parents=True)
            par_dir.mkdir(parents=True)
            
            # Sequential processing
            pipeline_seq = self.create_test_pipeline(temp_path / "sequential", num_workers=1)
            chunk_manager_seq = ChunkManager(seq_dir, chunk_size=10)
            
            missing_seq_ids = [1, 2, 3]  # Reduced from 5 to 3 for speed
            seq_processed, seq_errors = process_sequences_sequential(
                self.test_sequences[:3], missing_seq_ids, pipeline_seq, chunk_manager_seq
            )
            
            # Parallel processing
            pipeline_par = self.create_test_pipeline(temp_path / "parallel", num_workers=2)
            chunk_manager_par = ChunkManager(par_dir, chunk_size=10)
            
            par_processed, par_errors = process_sequences_parallel(
                self.test_sequences[:3], missing_seq_ids, pipeline_par, 2, chunk_manager_par
            )
            
            # Compare results
            self.assertEqual(seq_processed, par_processed)
            self.assertEqual(seq_errors, par_errors)
            self.assertEqual(seq_processed, 3)
            self.assertEqual(seq_errors, 0)
            
            # Verify same sequences were processed
            seq_sequences = chunk_manager_seq.get_existing_sequences()
            par_sequences = chunk_manager_par.get_existing_sequences()
            
            self.assertEqual(seq_sequences, par_sequences)
            self.assertEqual(sorted(seq_sequences), [1, 2, 3])
    
    def test_parallel_processing_discontinuous(self):
        """Test parallel processing with discontinuous missing sequences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path, num_workers=2)
            
            # Create output directory for chunks
            output_path = temp_path / 'embeddings'
            output_path.mkdir(exist_ok=True)
            
            # Create ChunkManager instance
            chunk_manager = ChunkManager(output_path, chunk_size=10)
            
            # Simulate discontinuous missing sequences (1, 3 missing, 2 already done)
            # Use only 3 sequences for speed
            missing_seq_ids = [1, 3]
            processed_count, error_count = process_sequences_parallel(
                self.test_sequences[:3], missing_seq_ids, pipeline, 2, chunk_manager
            )
            
            # Verify results
            self.assertEqual(processed_count, 2)
            self.assertEqual(error_count, 0)
            
            # Verify correct sequences were processed by checking chunk data
            chunk_data = chunk_manager.load_chunk(1)  # Sequences 1&3 go to chunk 1
            self.assertIsNotNone(chunk_data)
            if chunk_data is not None:
                actual_sequences = set(chunk_data.keys())
                self.assertEqual(actual_sequences, {1, 3})
    
    def test_parallel_processing_single_worker_fallback(self):
        """Test that single worker falls back to sequential processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path, num_workers=1)
            
            # Create output directory for chunks
            output_path = temp_path / 'embeddings'
            output_path.mkdir(exist_ok=True)
            
            # Create ChunkManager instance
            chunk_manager = ChunkManager(output_path, chunk_size=10)
            
            # Should fallback to sequential processing
            missing_seq_ids = [1, 2, 3]
            processed_count, error_count = process_sequences_parallel(
                self.test_sequences[:3], missing_seq_ids, pipeline, 1, chunk_manager
            )
            
            # Verify results (should work same as sequential)
            self.assertEqual(processed_count, 3)
            self.assertEqual(error_count, 0)
            
            # Verify chunk files exist
            chunk_files = list(chunk_manager.chunks_dir.glob("embeddings_*.npz"))
            self.assertGreater(len(chunk_files), 0)
            
            # Verify sequences were processed
            existing_sequences = chunk_manager.get_existing_sequences()
            self.assertEqual(sorted(existing_sequences), [1, 2, 3])
    
    def test_parallel_processing_multiple_workers(self):
        """Test parallel processing with different numbers of workers (simplified for CI)."""
        # Simplified test with just 2 workers and shorter sequences for CI performance
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pipeline = self.create_test_pipeline(temp_path, num_workers=2)
            
            # Create output directory for chunks
            output_path = temp_path / 'embeddings'
            output_path.mkdir(exist_ok=True)
            
            # Create ChunkManager instance
            chunk_manager = ChunkManager(output_path, chunk_size=10)
            
            # Process only 3 sequences for speed
            missing_seq_ids = [1, 2, 3]
            start_time = time.time()
            processed_count, error_count = process_sequences_parallel(
                self.test_sequences[:3], missing_seq_ids, pipeline, 2, chunk_manager
            )
            end_time = time.time()
            
            # Verify results
            self.assertEqual(processed_count, 3)
            self.assertEqual(error_count, 0)
            
            # Verify all sequences were processed
            existing_sequences = chunk_manager.get_existing_sequences()
            self.assertEqual(sorted(existing_sequences), [1, 2, 3])
            
            # Verify chunk files exist
            chunk_files = list(chunk_manager.chunks_dir.glob("embeddings_*.npz"))
            self.assertGreater(len(chunk_files), 0)
            
            print(f"Workers 2: {processed_count} sequences in {end_time - start_time:.2f}s")


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
