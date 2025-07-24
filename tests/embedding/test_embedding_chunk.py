#!/usr/bin/env python3
"""
test_embedding_chunk.py

Comprehensive test suite for the embedding chunk storage system.

Tests cover:
- ChunkMetadata creation and serialization
- ChunkManager initialization and metadata persistence
- Chunk saving and loading operations
- Sequence range calculations and integrity verification
- Error handling and edge cases
- Performance characteristics

Run with: python -m pytest tests/embedding/test_embedding_chunk.py -v
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from x_spanformer.embedding.embedding_chunk import (
    ChunkMetadata, ChunkManager, save_sequence_results_chunked
)


class TestChunkMetadata:
    """Test suite for ChunkMetadata class."""
    
    def test_chunk_metadata_creation(self):
        """Test basic ChunkMetadata creation."""
        meta = ChunkMetadata(
            chunk_id=1,
            start_seq_id=1,
            end_seq_id=100,
            sequence_count=100,
            file_path="chunks/embeddings_000001.npz",
            created_at="2025-07-23T12:00:00",
            components=["context", "seed"],
            file_size_mb=15.5
        )
        
        assert meta.chunk_id == 1
        assert meta.start_seq_id == 1
        assert meta.end_seq_id == 100
        assert meta.sequence_count == 100
        assert meta.file_size_mb == 15.5
        assert "context" in meta.components
        assert "seed" in meta.components
    
    def test_chunk_metadata_serialization(self):
        """Test to_dict and from_dict methods."""
        original = ChunkMetadata(
            chunk_id=42,
            start_seq_id=301,
            end_seq_id=400,
            sequence_count=100,
            file_path="chunks/embeddings_000042.npz",
            created_at="2025-07-23T12:00:00",
            components=["context"],
            file_size_mb=22.3
        )
        
        # Test serialization
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data['chunk_id'] == 42
        assert data['sequence_count'] == 100
        
        # Test deserialization
        restored = ChunkMetadata.from_dict(data)
        assert restored.chunk_id == original.chunk_id
        assert restored.start_seq_id == original.start_seq_id
        assert restored.end_seq_id == original.end_seq_id
        assert restored.components == original.components
    
    def test_contains_sequence(self):
        """Test sequence containment checking."""
        meta = ChunkMetadata(
            chunk_id=1,
            start_seq_id=10,
            end_seq_id=20,
            sequence_count=11,
            file_path="test.npz",
            created_at="2025-07-23T12:00:00",
            components=["context"],
            file_size_mb=1.0
        )
        
        assert meta.contains_sequence(10)  # Start boundary
        assert meta.contains_sequence(15)  # Middle
        assert meta.contains_sequence(20)  # End boundary
        assert not meta.contains_sequence(9)   # Before range
        assert not meta.contains_sequence(21)  # After range
    
    def test_get_sequence_range(self):
        """Test sequence range generation."""
        meta = ChunkMetadata(
            chunk_id=1,
            start_seq_id=5,
            end_seq_id=8,
            sequence_count=4,
            file_path="test.npz",
            created_at="2025-07-23T12:00:00",
            components=["context"],
            file_size_mb=1.0
        )
        
        seq_range = meta.get_sequence_range()
        assert list(seq_range) == [5, 6, 7, 8]


class TestChunkManager:
    """Test suite for ChunkManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def chunk_manager(self, temp_dir):
        """Provide ChunkManager instance for testing."""
        return ChunkManager(temp_dir, chunk_size=10)
    
    @pytest.fixture
    def sample_pipeline_config(self):
        """Provide sample pipeline configuration."""
        return {
            'output': {
                'save_seed_embeddings': False,  # Don't require by default
                'save_soft_probabilities': False,
                'save_json_metadata': True
            }
        }
    
    def test_chunk_manager_initialization(self, temp_dir):
        """Test ChunkManager initialization."""
        manager = ChunkManager(temp_dir, chunk_size=50)
        
        assert manager.output_dir == temp_dir
        assert manager.chunk_size == 50
        assert manager.chunks_dir == temp_dir / "chunks"
        assert manager.chunks_dir.exists()
        assert manager.metadata_file == temp_dir / "metadata.json"
        assert isinstance(manager.chunks_metadata, dict)
    
    def test_chunk_id_calculation(self, chunk_manager):
        """Test chunk ID calculation for sequence IDs."""
        # Chunk size = 10
        assert chunk_manager.get_chunk_id(1) == 1    # First sequence
        assert chunk_manager.get_chunk_id(10) == 1   # Last in first chunk
        assert chunk_manager.get_chunk_id(11) == 2   # First in second chunk
        assert chunk_manager.get_chunk_id(25) == 3   # Mid third chunk
        assert chunk_manager.get_chunk_id(100) == 10 # Later chunk
    
    def test_chunk_range_calculation(self, chunk_manager):
        """Test chunk range calculation."""
        start, end = chunk_manager.get_chunk_range(1)
        assert start == 1 and end == 10
        
        start, end = chunk_manager.get_chunk_range(2)
        assert start == 11 and end == 20
        
        start, end = chunk_manager.get_chunk_range(5)
        assert start == 41 and end == 50
    
    def test_chunk_file_path(self, chunk_manager):
        """Test chunk file path generation."""
        path = chunk_manager.get_chunk_file_path(1)
        assert path.name == "embeddings_000001.npz"
        assert path.parent == chunk_manager.chunks_dir
        
        path = chunk_manager.get_chunk_file_path(42)
        assert path.name == "embeddings_000042.npz"
    
    def test_save_and_load_empty_chunk(self, chunk_manager, sample_pipeline_config):
        """Test handling of empty chunk data."""
        result = chunk_manager.save_chunk({}, sample_pipeline_config)
        assert result is None
    
    def test_save_and_load_chunk(self, chunk_manager, sample_pipeline_config):
        """Test saving and loading a chunk with real data."""
        # Create sample embedding data
        chunk_data = {}
        for seq_id in range(1, 6):  # 5 sequences
            chunk_data[seq_id] = {
                'sequence': f"This is test sequence {seq_id}",
                'contextual_embeddings': np.random.randn(20, 512).astype(np.float32),
                'seed_embeddings': np.random.randn(20, 512).astype(np.float32),
                'span_candidates': [(0, 5), (2, 8), (10, 15)],
                'soft_probabilities': np.random.randn(20, 100).astype(np.float32)
            }
        
        # Save chunk
        chunk_meta = chunk_manager.save_chunk(chunk_data, sample_pipeline_config)
        
        # Verify metadata
        assert chunk_meta is not None
        assert chunk_meta.chunk_id == 1
        assert chunk_meta.start_seq_id == 1
        assert chunk_meta.end_seq_id == 5
        assert chunk_meta.sequence_count == 5
        assert "context" in chunk_meta.components
        # Seed embeddings should not be saved with current config
        assert "seed" not in chunk_meta.components
        assert chunk_meta.file_size_mb > 0
        
        # Verify file was created
        chunk_file = chunk_manager.get_chunk_file_path(1)
        assert chunk_file.exists()
        
        # Load chunk back
        loaded_data = chunk_manager.load_chunk(1)
        assert loaded_data is not None
        assert len(loaded_data) == 5
        
        # Verify data integrity
        for seq_id in range(1, 6):
            assert seq_id in loaded_data
            loaded_seq = loaded_data[seq_id]
            assert loaded_seq['sequence'] == f"This is test sequence {seq_id}"
            assert isinstance(loaded_seq['contextual_embeddings'], np.ndarray)
            # Seed embeddings should not be in loaded data since config disabled them
            assert 'seed_embeddings' not in loaded_seq
            assert loaded_seq['span_candidates'] == [(0, 5), (2, 8), (10, 15)]
    
    def test_metadata_persistence(self, temp_dir, sample_pipeline_config):
        """Test that metadata persists across manager instances."""
        # Create first manager and save data
        manager1 = ChunkManager(temp_dir, chunk_size=5)
        
        chunk_data = {
            1: {
                'sequence': "Test sequence",
                'contextual_embeddings': np.random.randn(10, 512).astype(np.float32),
                'span_candidates': [(0, 4)]
            }
        }
        
        chunk_meta = manager1.save_chunk(chunk_data, sample_pipeline_config)
        assert chunk_meta is not None
        
        # Create second manager instance
        manager2 = ChunkManager(temp_dir, chunk_size=5)
        
        # Verify metadata was loaded
        assert 1 in manager2.chunks_metadata
        assert manager2.chunks_metadata[1].sequence_count == 1
        
        # Verify we can still load the data
        loaded_data = manager2.load_chunk(1)
        assert loaded_data is not None
        assert 1 in loaded_data
    
    def test_get_existing_sequences(self, chunk_manager, sample_pipeline_config):
        """Test retrieval of existing sequence IDs."""
        # Initially empty
        existing = chunk_manager.get_existing_sequences()
        assert len(existing) == 0
        
        # Save first chunk (sequences 1-5)
        chunk_data_1 = {}
        for seq_id in range(1, 6):
            chunk_data_1[seq_id] = {
                'sequence': f"Sequence {seq_id}",
                'contextual_embeddings': np.random.randn(10, 512).astype(np.float32),
                'span_candidates': [(0, 4)]
            }
        
        chunk_manager.save_chunk(chunk_data_1, sample_pipeline_config)
        
        # Save second chunk (sequences 11-15, gap intentional)
        chunk_data_2 = {}
        for seq_id in range(11, 16):
            chunk_data_2[seq_id] = {
                'sequence': f"Sequence {seq_id}",
                'contextual_embeddings': np.random.randn(10, 512).astype(np.float32),
                'span_candidates': [(0, 4)]
            }
        
        chunk_manager.save_chunk(chunk_data_2, sample_pipeline_config)
        
        # Check existing sequences
        existing = chunk_manager.get_existing_sequences()
        
        # Should include only the actual saved sequences, not the full chunk ranges
        expected = set(range(1, 6)) | set(range(11, 16))  # Only actual saved sequences
        assert existing == expected
    
    def test_verify_chunk_integrity(self, chunk_manager, sample_pipeline_config):
        """Test chunk integrity verification."""
        # Non-existent chunk
        assert not chunk_manager.verify_chunk_integrity(999)
        
        # Save a valid chunk
        chunk_data = {
            1: {
                'sequence': "Test sequence",
                'contextual_embeddings': np.random.randn(10, 512).astype(np.float32),
                'span_candidates': [(0, 4)]
            }
        }
        
        chunk_meta = chunk_manager.save_chunk(chunk_data, sample_pipeline_config)
        assert chunk_meta is not None
        
        # Should verify as valid
        assert chunk_manager.verify_chunk_integrity(1)
        
        # Corrupt the file
        chunk_file = chunk_manager.get_chunk_file_path(1)
        with open(chunk_file, 'wb') as f:
            f.write(b"corrupted data")
        
        # Should now fail verification
        assert not chunk_manager.verify_chunk_integrity(1)
    
    def test_chunk_statistics(self, chunk_manager, sample_pipeline_config):
        """Test chunk statistics calculation."""
        # Empty stats
        stats = chunk_manager.get_chunk_statistics()
        assert stats['total_chunks'] == 0
        assert stats['total_sequences'] == 0
        assert stats['total_size_mb'] == 0.0
        
        # Save some chunks
        for chunk_start in [1, 11]:  # Two chunks
            chunk_data = {}
            for seq_id in range(chunk_start, chunk_start + 5):
                chunk_data[seq_id] = {
                    'sequence': f"Sequence {seq_id}",
                    'contextual_embeddings': np.random.randn(10, 512).astype(np.float32),
                    'span_candidates': [(0, 4)]
                }
            chunk_manager.save_chunk(chunk_data, sample_pipeline_config)
        
        # Check updated stats
        stats = chunk_manager.get_chunk_statistics()
        assert stats['total_chunks'] == 2
        assert stats['total_sequences'] == 10  # 5 sequences per chunk
        assert stats['total_size_mb'] > 0
        assert stats['chunk_range'] == (1, 2)
        assert stats['chunk_size_setting'] == 10
    
    def test_cleanup_orphaned_files(self, chunk_manager):
        """Test cleanup of orphaned chunk files."""
        # Create an orphaned file
        orphaned_file = chunk_manager.chunks_dir / "embeddings_999999.npz"
        orphaned_file.write_bytes(b"fake chunk data")
        
        assert orphaned_file.exists()
        
        # Cleanup should remove it
        removed_files = chunk_manager.cleanup_orphaned_files()
        
        assert len(removed_files) == 1
        assert removed_files[0] == orphaned_file
        assert not orphaned_file.exists()
    
    def test_chunk_size_mismatch_warning(self, temp_dir, sample_pipeline_config):
        """Test warning when chunk size changes between runs."""
        # Create metadata with different chunk size
        metadata = {
            'chunk_size': 50,
            'total_chunks': 0,
            'chunks': {}
        }
        
        metadata_file = temp_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create manager with different chunk size
        with patch('x_spanformer.embedding.embedding_chunk.get_embedding_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            manager = ChunkManager(temp_dir, chunk_size=100)
            
            # Should have logged a warning
            mock_logger.return_value.warning.assert_called_once()
            warning_call = mock_logger.return_value.warning.call_args[0][0]
            assert "Chunk size mismatch" in warning_call


class TestSaveSequenceResultsChunked:
    """Test suite for save_sequence_results_chunked function."""
    
    @pytest.fixture
    def mock_logger(self):
        """Provide mock logger for testing."""
        return MagicMock()
    
    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def chunk_manager(self, temp_dir):
        """Provide ChunkManager instance for testing."""
        return ChunkManager(temp_dir, chunk_size=10)
    
    @pytest.fixture
    def sample_config(self):
        """Provide sample pipeline configuration."""
        return {
            'output': {
                'save_seed_embeddings': True,
                'save_soft_probabilities': True
            }
        }
    
    def test_empty_buffer(self, chunk_manager, sample_config, mock_logger):
        """Test handling of empty result buffer."""
        result = save_sequence_results_chunked(chunk_manager, {}, sample_config, mock_logger)
        assert result is None
        mock_logger.debug.assert_called_once_with("Empty result buffer, skipping chunk save")
    
    def test_gpu_tensor_conversion(self, chunk_manager, sample_config, mock_logger):
        """Test conversion of GPU tensors to CPU arrays."""
        # Mock GPU tensors
        mock_tensor = MagicMock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(10, 512)
        mock_tensor.numel.return_value = 5120  # 10 * 512
        
        result_buffer = {
            1: {
                'sequence': "Test sequence",
                'span_candidates': [(0, 4)],
                'contextual_embeddings': mock_tensor,
                'seed_embeddings': mock_tensor,
                'soft_probabilities': mock_tensor
            }
        }
        
        chunk_meta = save_sequence_results_chunked(chunk_manager, result_buffer, sample_config, mock_logger)
        
        assert chunk_meta is not None
        assert chunk_meta.chunk_id == 1
        assert chunk_meta.sequence_count == 1
        
        # Verify tensor conversion was called
        assert mock_tensor.detach.called
        assert mock_tensor.detach.return_value.cpu.called
        assert mock_tensor.detach.return_value.cpu.return_value.numpy.called


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    def test_load_nonexistent_chunk(self, temp_dir):
        """Test loading a chunk that doesn't exist."""
        manager = ChunkManager(temp_dir)
        result = manager.load_chunk(999)
        assert result is None
    
    def test_corrupted_metadata_file(self, temp_dir):
        """Test handling of corrupted metadata file."""
        # Create corrupted metadata file
        metadata_file = temp_dir / "metadata.json"
        metadata_file.write_text("invalid json {")
        
        # Should handle gracefully
        with patch('x_spanformer.embedding.embedding_chunk.get_embedding_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            manager = ChunkManager(temp_dir)
            assert len(manager.chunks_metadata) == 0
            mock_logger.return_value.error.assert_called()
    
    def test_sequence_ids_spanning_chunks(self, temp_dir):
        """Test automatic handling when sequence IDs span multiple chunks."""
        manager = ChunkManager(temp_dir, chunk_size=5)
        
        # Create data spanning chunk boundary
        chunk_data = {
            4: {  # End of chunk 1
                'sequence': "Sequence 4",
                'contextual_embeddings': np.random.randn(10, 512).astype(np.float32),
                'span_candidates': [(0, 4)]
            },
            6: {  # Start of chunk 2
                'sequence': "Sequence 6", 
                'contextual_embeddings': np.random.randn(10, 512).astype(np.float32),
                'span_candidates': [(0, 4)]
            }
        }
        
        config = {'output': {}}
        
        # Should automatically split and save to appropriate chunks
        chunk_meta = manager.save_chunk(chunk_data, config)
        assert chunk_meta is not None
        
        # Verify sequences were saved to correct chunks
        chunk1_data = manager.load_chunk(1)
        chunk2_data = manager.load_chunk(2)
        
        assert chunk1_data is not None
        assert chunk2_data is not None
        assert 4 in chunk1_data
        assert 6 in chunk2_data
        
        # Verify all sequences are accessible
        existing_sequences = manager.get_existing_sequences()
        assert {4, 6}.issubset(existing_sequences)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
