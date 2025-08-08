#!/usr/bin/env python3
"""
Tests for SpanAnnotatorPipeline.

Tests the SpanAnnotatorPipeline class with focus on proper initialization,
sequence processing, and result handling.
"""
import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from x_spanformer.pipelines.span_annotator import SpanAnnotatorPipeline
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.identifier import RecordID


class TestSpanAnnotatorPipelineInitialization:
    """Test cases for SpanAnnotatorPipeline initialization."""
    
    def test_pipeline_initialization_basic(self):
        """Test basic pipeline initialization with default parameters."""
        pipeline = SpanAnnotatorPipeline()
        
        assert pipeline.model_name == "llama3.2:3b"
        assert pipeline.temperature == 0.2
        assert pipeline.conversation_timeout == 180.0
        assert pipeline.max_retries == 3
        assert pipeline.session is not None
    
    def test_pipeline_initialization_custom(self):
        """Test pipeline initialization with custom parameters."""
        pipeline = SpanAnnotatorPipeline(
            model_name="qwen2.5-coder:14b",
            temperature=0.1,
            conversation_timeout=300.0,
            max_retries=5
        )
        
        assert pipeline.model_name == "qwen2.5-coder:14b"
        assert pipeline.temperature == 0.1
        assert pipeline.conversation_timeout == 300.0
        assert pipeline.max_retries == 5
        assert pipeline.session is not None


class TestSequenceLoading:
    """Test cases for sequence loading functionality."""
    
    def test_parse_range_specification_single(self):
        """Test parsing single sequence range."""
        pipeline = SpanAnnotatorPipeline()
        result = pipeline.parse_range_specification("5")
        assert result == [5]
    
    def test_parse_range_specification_range(self):
        """Test parsing sequence range."""
        pipeline = SpanAnnotatorPipeline()
        result = pipeline.parse_range_specification("1-5")
        assert result == [1, 2, 3, 4, 5]
    
    def test_parse_range_specification_mixed(self):
        """Test parsing mixed range specification."""
        pipeline = SpanAnnotatorPipeline()
        result = pipeline.parse_range_specification("1-3,5,7-9")
        assert result == [1, 2, 3, 5, 7, 8, 9]
    
    def test_load_sequences_empty_file(self):
        """Test loading from empty corpus file."""
        pipeline = SpanAnnotatorPipeline()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            corpus_file = Path(f.name)
        
        try:
            sequences = pipeline.load_sequences(corpus_file)
            assert sequences == []
        finally:
            corpus_file.unlink()
    
    def test_load_sequences_with_valid_data(self):
        """Test loading sequences from valid corpus file."""
        pipeline = SpanAnnotatorPipeline()
        
        # Create test sequence data
        test_sequences = []
        for i in range(3):
            sequence_data = {
                "id": {"id": f"seq{i}"},
                "raw": f"This is test sequence {i}.",
                "meta": {
                    "sequence_number": i + 1,
                    "timestamp": "2025-01-01T00:00:00",
                    "source": "test"
                }
            }
            test_sequences.append(sequence_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for seq_data in test_sequences:
                f.write(json.dumps(seq_data) + '\n')
            corpus_file = Path(f.name)
        
        try:
            sequences = pipeline.load_sequences(corpus_file)
            assert len(sequences) == 3
            assert all(isinstance(seq, PretrainRecord) for seq in sequences)
            assert sequences[0].raw == "This is test sequence 0."
        finally:
            corpus_file.unlink()
    
    def test_load_sequences_with_range_filter(self):
        """Test loading sequences with range filtering."""
        pipeline = SpanAnnotatorPipeline()
        
        # Create test sequence data
        test_sequences = []
        for i in range(5):
            sequence_data = {
                "id": {"id": f"seq{i}"},
                "raw": f"This is test sequence {i}.",
                "meta": {
                    "sequence_number": i + 1,
                    "timestamp": "2025-01-01T00:00:00",
                    "source": "test"
                }
            }
            test_sequences.append(sequence_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for seq_data in test_sequences:
                f.write(json.dumps(seq_data) + '\n')
            corpus_file = Path(f.name)
        
        try:
            # Filter for sequences 2-4
            sequences = pipeline.load_sequences(corpus_file, range_spec="2-4")
            assert len(sequences) == 3
            assert all(seq.meta.sequence_number in [2, 3, 4] for seq in sequences)
        finally:
            corpus_file.unlink()


class TestOutputStructure:
    """Test cases for output directory management."""
    
    def test_ensure_output_structure(self):
        """Test creation of output directory structure."""
        pipeline = SpanAnnotatorPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "annotations"
            pipeline.ensure_output_structure(output_dir)
            
            assert output_dir.exists()
            assert (output_dir / "working").exists()
            # Note: "consolidated" directory is no longer created - annotations.jsonl saves directly in output_dir
    
    def test_load_existing_results_empty(self):
        """Test loading existing results from empty directory."""
        pipeline = SpanAnnotatorPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            results = pipeline.load_existing_results(output_dir)
            assert results == {}
    
    def test_save_working_file(self):
        """Test saving working file for sequence annotation."""
        pipeline = SpanAnnotatorPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline.ensure_output_structure(output_dir)
            
            # Create test sequence
            test_sequence = PretrainRecord(
                id=RecordID(id="seq1"),
                raw="This is a test sequence.",
                sequence_number=1,
                meta=RecordMeta(
                    sequence_number=1,
                    timestamp="2025-01-01T00:00:00",
                    source="test",
                    doc_language=None,
                    extracted_by=None,
                    confidence=None,
                    source_file=None,
                    notes=None
                )
            )
            
            # Save working file
            pipeline.save_working_file(output_dir, test_sequence, error_message="Test error")
            
            # Check file was created
            working_file = output_dir / "working" / "sequence-00000001.json"
            assert working_file.exists()
            
            # Check file contents
            with open(working_file, 'r') as f:
                data = json.load(f)
            
            assert data["sequence_number"] == 1
            assert data["raw_text"] == "This is a test sequence."
            assert data["status"] == "failed"
            assert data["error_message"] == "Test error"


@pytest.mark.asyncio
class TestProcessingIntegration:
    """Integration tests for sequence processing."""
    
    async def test_process_sequences_empty_corpus(self):
        """Test processing with empty corpus."""
        pipeline = SpanAnnotatorPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_file = Path(temp_dir) / "empty.jsonl"
            output_dir = Path(temp_dir) / "output"
            
            # Create empty corpus file
            corpus_file.touch()
            
            # Mock the session to avoid actual LLM calls
            with patch.object(pipeline, 'session'):
                stats = await pipeline.process_sequences(corpus_file, output_dir)
                
                assert stats["total_sequences"] == 0
                assert stats["processed_sequences"] == 0
                assert stats["successful_annotations"] == 0
                assert stats["failed_annotations"] == 0
    
    async def test_process_sequences_mock_success(self):
        """Test successful sequence processing with mocked session."""
        pipeline = SpanAnnotatorPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_file = Path(temp_dir) / "test.jsonl" 
            output_dir = Path(temp_dir) / "output"
            
            # Create test corpus
            test_data = {
                "id": {"id": "seq1"},
                "raw": "This is a test.",
                "meta": {
                    "sequence_number": 1,
                    "timestamp": "2025-01-01T00:00:00",
                    "source": "test"
                }
            }
            
            with open(corpus_file, 'w') as f:
                f.write(json.dumps(test_data) + '\n')
            
            # Mock successful annotation result
            mock_result = Mock()
            mock_result.success = True
            mock_result.annotation_record = Mock()
            mock_result.annotation_record.span_annotations = []
            mock_result.annotation_record.agent_metadata = {"test": "data"}
            mock_result.error_message = None
            
            with patch.object(pipeline.session, 'annotate_single_sequence', return_value=mock_result):
                stats = await pipeline.process_sequences(corpus_file, output_dir)
                
                assert stats["total_sequences"] == 1
                assert stats["processed_sequences"] == 1
                assert stats["successful_annotations"] == 1
                assert stats["failed_annotations"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
