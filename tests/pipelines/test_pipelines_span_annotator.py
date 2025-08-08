#!/usr/bin/env python3
"""
Enhanced tests for SpanAnnotatorPipeline with shared modules integration.

Tests the SpanAnnotatorPipeline class with focus on the integration
of shared telemetry, logging, and annotation processing modules.
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from x_spanformer.pipelines.span_annotator import SpanAnnotatorPipeline
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord


class TestSpanAnnotatorPipelineInitialization:
    """Test cases for SpanAnnotatorPipeline initialization."""
    
    def test_initialization_success(self):
        """Test successful pipeline initialization with default parameters."""
        with patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            pipeline = SpanAnnotatorPipeline()
        
        # Verify initialization
        assert pipeline.model_name == "llama3.2:3b"
        assert pipeline.temperature == 0.2
        assert pipeline.conversation_timeout == 180.0
        assert pipeline.max_retries == 3
        assert hasattr(pipeline, 'session')
        assert hasattr(pipeline, 'pipeline_stats')
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            pipeline = SpanAnnotatorPipeline(
                model_name="custom-model",
                temperature=0.5,
                conversation_timeout=120.0,
                max_retries=5
            )
        
        # Verify custom parameters
        assert pipeline.model_name == "custom-model"
        assert pipeline.temperature == 0.5
        assert pipeline.conversation_timeout == 120.0
        assert pipeline.max_retries == 5


class TestSpanAnnotatorPipelineSequenceHandling:
    """Test cases for sequence loading and handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            self.pipeline = SpanAnnotatorPipeline(
                model_name="test_model",
                temperature=0.7,
                conversation_timeout=180.0,
                max_retries=3
            )
    
    def test_parse_range_specification_single(self):
        """Test parsing single sequence ID."""
        result = self.pipeline.parse_range_specification("42")
        assert result == [42]
    
    def test_parse_range_specification_list(self):
        """Test parsing list of sequence IDs."""
        result = self.pipeline.parse_range_specification("1,5,10")
        assert result == [1, 5, 10]
    
    def test_parse_range_specification_range(self):
        """Test parsing range of sequence IDs."""
        result = self.pipeline.parse_range_specification("1-5")
        assert result == [1, 2, 3, 4, 5]
    
    def test_parse_range_specification_mixed(self):
        """Test parsing mixed specification."""
        result = self.pipeline.parse_range_specification("1-3,7,10-12")
        assert result == [1, 2, 3, 7, 10, 11, 12]
    
    def test_parse_range_specification_duplicates(self):
        """Test parsing with duplicates (should be removed)."""
        result = self.pipeline.parse_range_specification("1,2,1,3,2")
        assert result == [1, 2, 3]
    
    def test_load_sequences_with_range(self):
        """Test loading specific sequences with range specification."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            # Write test corpus data with proper PretrainRecord schema  
            test_data = [
                {"raw": "Test 1", "sequence_number": 1, "meta": {"sequence_number": 1}},
                {"raw": "Test 2", "sequence_number": 2, "meta": {"sequence_number": 2}}, 
                {"raw": "Test 3", "sequence_number": 3, "meta": {"sequence_number": 3}},
                {"raw": "Test 4", "sequence_number": 4, "meta": {"sequence_number": 4}},
            ]
            
            for data in test_data:
                f.write(json.dumps(data) + '\n')
            f.flush()
            
            # Load specific sequences using range
            result = self.pipeline.load_sequences(Path(f.name), "1,3")
            
        # Clean up
        Path(f.name).unlink()
        
        assert len(result) == 2
        assert all(isinstance(record, PretrainRecord) for record in result)
        sequence_numbers = [record.sequence_number for record in result]
        assert set(sequence_numbers) == {1, 3}


class TestSpanAnnotatorPipelineOutputHandling:
    """Test cases for output directory and file handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            self.pipeline = SpanAnnotatorPipeline()
    
    def test_ensure_output_structure(self):
        """Test output directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_output"
            
            self.pipeline.ensure_output_structure(output_dir)
            
            # Verify directory structure
            assert output_dir.exists()
            assert (output_dir / "working").exists()
    
    def test_load_existing_results_empty_directory(self):
        """Test loading existing results from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            results = self.pipeline.load_existing_results(output_dir)
            
            assert results == {}


if __name__ == "__main__":
    pytest.main([__file__])
