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
    
    @patch('x_spanformer.pipelines.span_annotator.load_config')
    @patch('x_spanformer.pipelines.span_annotator.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_initialization_success(self, mock_yaml_load, mock_open, mock_exists, mock_load_config):
        """Test successful pipeline initialization."""
        # Mock configuration loading
        mock_config = Mock()
        mock_config.processing.max_retries = 3
        mock_config.processing.conversation_timeout = 300
        mock_load_config.return_value = mock_config
        
        # Mock agent config file exists
        mock_exists.return_value = True
        
        # Mock agent configuration
        mock_agent_config = {
            "model": {
                "name": "test-model",
                "temperature": 0.7
            },
            "dialogue": {
                "max_turns": 5
            },
            "agent": {
                "early_termination": {"enabled": True}
            }
        }
        mock_yaml_load.return_value = mock_agent_config
        
        # Create pipeline
        with patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            pipeline = SpanAnnotatorPipeline()
        
        # Verify initialization
        assert pipeline.config == mock_config
        assert hasattr(pipeline, 'annotation_processor')
        assert hasattr(pipeline, 'telemetry')
        assert hasattr(pipeline, 'agent')
        assert hasattr(pipeline, 'stats')
    
    @patch('x_spanformer.pipelines.span_annotator.load_config')
    @patch('x_spanformer.pipelines.span_annotator.Path.exists')
    def test_initialization_missing_agent_config(self, mock_exists, mock_load_config):
        """Test initialization failure when agent config is missing."""
        mock_load_config.return_value = Mock()
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Agent configuration file not found"):
            SpanAnnotatorPipeline()
    
    @patch('x_spanformer.pipelines.span_annotator.load_config')
    @patch('x_spanformer.pipelines.span_annotator.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_initialization_invalid_agent_config(self, mock_yaml_load, mock_open, mock_exists, mock_load_config):
        """Test initialization failure with invalid agent config."""
        mock_load_config.return_value = Mock()
        mock_exists.return_value = True
        
        # Mock empty/invalid agent config
        mock_yaml_load.return_value = {}
        
        with pytest.raises(RuntimeError, match="Failed to load agent configuration.*Agent configuration file is empty or invalid"):
            SpanAnnotatorPipeline()


class TestSpanAnnotatorPipelineSequenceHandling:
    """Test cases for sequence loading and handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('x_spanformer.pipelines.span_annotator.load_config'), \
             patch('x_spanformer.pipelines.span_annotator.Path.exists', return_value=True), \
             patch('builtins.open'), \
             patch('yaml.safe_load') as mock_yaml, \
             patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            
            # Mock valid agent config
            mock_yaml.return_value = {
                "model": {"name": "test", "temperature": 0.7},
                "dialogue": {"max_turns": 5},
                "agent": {"early_termination": {}}
            }
            
            return SpanAnnotatorPipeline()
    
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
    
    def test_get_all_corpus_sequences(self):
        """Test getting all sequence numbers from corpus file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            # Write test corpus data
            test_data = [
                {"text": "Test 1", "meta": {"sequence_number": 1}},
                {"text": "Test 2", "meta": {"sequence_number": 5}},
                {"text": "Test 3"},  # No sequence_number, should use line number
            ]
            
            for data in test_data:
                f.write(json.dumps(data) + '\n')
            f.flush()
            
            result = self.pipeline.get_all_corpus_sequences(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result == [1, 3, 5]  # 1 from meta, 3 from line number, 5 from meta
    
    def test_load_target_sequences(self):
        """Test loading specific target sequences."""
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
            
            # Load specific sequences
            result = self.pipeline.load_target_sequences(Path(f.name), [1, 3])
            
        # Clean up
        Path(f.name).unlink()
        
        assert len(result) == 2
        assert all(isinstance(record, PretrainRecord) for record in result)
        sequence_numbers = [record.sequence_number for record in result]
        assert set(sequence_numbers) == {1, 3}


class TestSpanAnnotatorPipelineTelemetryIntegration:
    """Test cases for telemetry integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('x_spanformer.pipelines.span_annotator.load_config'), \
             patch('x_spanformer.pipelines.span_annotator.Path.exists', return_value=True), \
             patch('builtins.open'), \
             patch('yaml.safe_load') as mock_yaml, \
             patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            
            mock_yaml.return_value = {
                "model": {"name": "test", "temperature": 0.7},
                "dialogue": {"max_turns": 5},
                "agent": {"early_termination": {}}
            }
            
            return SpanAnnotatorPipeline()
    
    def test_telemetry_initialization(self):
        """Test that telemetry is properly initialized."""
        from x_spanformer.pipelines.shared.pipeline_telemetry import SpanAnnotationTelemetry
        
        assert isinstance(self.pipeline.telemetry, SpanAnnotationTelemetry)
        assert self.pipeline.telemetry.pipeline_name == "Span Annotation"
    
    def test_telemetry_direct_method_calls(self):
        """Test that telemetry methods are called directly (no wrapper methods)."""
        # Verify that the pipeline doesn't have wrapper methods
        assert not hasattr(self.pipeline, 'display_telemetry_panel')
        assert not hasattr(self.pipeline, 'update_telemetry_on_completion')
        assert not hasattr(self.pipeline, 'update_telemetry_on_failure')
        
        # Verify that telemetry object has the expected methods
        assert hasattr(self.pipeline.telemetry, 'display_progress_panel')
        assert hasattr(self.pipeline.telemetry, 'update_on_completion')
        assert hasattr(self.pipeline.telemetry, 'update_on_failure')
    
    def test_telemetry_method_integration(self):
        """Test that telemetry methods are called correctly."""  
        # Test that the telemetry object exists and has the right methods
        assert hasattr(self.pipeline.telemetry, 'update_on_completion')
        assert hasattr(self.pipeline.telemetry, 'update_on_failure')
        assert hasattr(self.pipeline.telemetry, 'display_progress_panel')
        
        # Test that we can call the methods without error (they're mocked in setup)
        mock_annotation_result = Mock()
        mock_annotation_result.span_annotations = []
        start_time = datetime.now()
        
        # These calls should work without raising exceptions
        self.pipeline.telemetry.update_on_completion(mock_annotation_result, start_time)
        self.pipeline.telemetry.update_on_failure(start_time)
        self.pipeline.telemetry.display_progress_panel()
        self.pipeline.telemetry.update_on_completion(mock_annotation_result, start_time)
        
        # Test failure update  
        self.pipeline.telemetry.update_on_failure(start_time)
        
        # Test display
        self.pipeline.telemetry.display_progress_panel()
        
        # If we get here without errors, the interface is correct


class TestSpanAnnotatorPipelineOutputHandling:
    """Test cases for output directory and file handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('x_spanformer.pipelines.span_annotator.load_config'), \
             patch('x_spanformer.pipelines.span_annotator.Path.exists', return_value=True), \
             patch('builtins.open'), \
             patch('yaml.safe_load') as mock_yaml, \
             patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            
            mock_yaml.return_value = {
                "model": {"name": "test", "temperature": 0.7},
                "dialogue": {"max_turns": 5},
                "agent": {"early_termination": {}}
            }
            
            return SpanAnnotatorPipeline()
    
    def test_ensure_output_structure(self):
        """Test output directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_output"
            
            self.pipeline.ensure_output_structure(output_dir)
            
            # Verify directory structure
            assert output_dir.exists()
            assert (output_dir / "working").exists()
            assert (output_dir / "annotations.jsonl").exists()
            assert (output_dir / "metadata.json").exists()
            
            # Verify metadata file content
            with open(output_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
                
            assert "pipeline_version" in metadata
            assert "started_at" in metadata
            assert "total_sequences" in metadata
            assert metadata["processed_sequences"] == 0
    
    def test_load_existing_results(self):
        """Test loading existing working files for resume functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            working_dir = output_dir / "working"
            working_dir.mkdir(parents=True)
            
            # Create mock working files with correct structure
            test_files = [
                ("corpus-seq-001.json", {
                    "sequence_number": 1, 
                    "annotation_session": {
                        "annotation_status": "completed", 
                        "spans_extracted": 5
                    }
                }),
                ("corpus-seq-002.json", {
                    "sequence_number": 2, 
                    "annotation_session": {
                        "annotation_status": "failed", 
                        "spans_extracted": 0
                    }
                }),
                ("corpus-seq-005.json", {
                    "sequence_number": 5, 
                    "annotation_session": {
                        "annotation_status": "completed", 
                        "spans_extracted": 3
                    }
                })
            ]
            
            for filename, content in test_files:
                with open(working_dir / filename, 'w') as f:
                    json.dump(content, f)
            
            results = self.pipeline.load_existing_results(output_dir)
            
            assert len(results) == 3
            assert results[1] == "completed"
            assert results[2] == "failed"  
            assert results[5] == "completed"
    
    def test_load_existing_results_empty_directory(self):
        """Test loading existing results from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            results = self.pipeline.load_existing_results(output_dir)
            
            assert results == {}


class TestSpanAnnotatorPipelineAnnotationProcessorIntegration:
    """Test cases for annotation processor integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('x_spanformer.pipelines.span_annotator.load_config'), \
             patch('x_spanformer.pipelines.span_annotator.Path.exists', return_value=True), \
             patch('builtins.open'), \
             patch('yaml.safe_load') as mock_yaml, \
             patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            
            mock_yaml.return_value = {
                "model": {"name": "test", "temperature": 0.7},
                "dialogue": {"max_turns": 5},
                "agent": {"early_termination": {}}
            }
            
            return SpanAnnotatorPipeline()
    
    def test_annotation_processor_initialization(self):
        """Test that annotation processor is properly initialized."""
        from x_spanformer.pipelines.shared.annotation_processor import AnnotationProcessor
        
        assert isinstance(self.pipeline.annotation_processor, AnnotationProcessor)
    
    def test_annotation_processor_method_access(self):
        """Test that annotation processor methods are accessible."""
        # Test that enhanced methods are available
        expected_methods = [
            'calculate_working_file_statistics',
            'update_metadata_file',
            'fix_metadata_from_working_files',
            'analyze_processing_gaps'
        ]
        
        for method_name in expected_methods:
            assert hasattr(self.pipeline.annotation_processor, method_name)
            method = getattr(self.pipeline.annotation_processor, method_name)
            assert callable(method)


class TestSpanAnnotatorPipelineStatistics:
    """Test cases for pipeline statistics tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('x_spanformer.pipelines.span_annotator.load_config'), \
             patch('x_spanformer.pipelines.span_annotator.Path.exists', return_value=True), \
             patch('builtins.open'), \
             patch('yaml.safe_load') as mock_yaml, \
             patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            
            mock_yaml.return_value = {
                "model": {"name": "test", "temperature": 0.7},
                "dialogue": {"max_turns": 5},
                "agent": {"early_termination": {}}
            }
            
            return SpanAnnotatorPipeline()
    
    def test_initial_stats(self):
        """Test initial statistics state."""
        expected_keys = [
            "total_sequences", "processed_sequences", "successful_annotations",
            "failed_annotations", "total_spans", "processing_time",
            "started_at", "completed_at", "consecutive_failures", "max_consecutive_failures"
        ]
        
        for key in expected_keys:
            assert key in self.pipeline.stats
        
        # Check initial values
        assert self.pipeline.stats["total_sequences"] == 0
        assert self.pipeline.stats["processed_sequences"] == 0
        assert self.pipeline.stats["consecutive_failures"] == 0
        assert self.pipeline.stats["max_consecutive_failures"] == 3


class TestSpanAnnotatorPipelineMissingSequences:
    """Test cases for missing sequence detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('x_spanformer.pipelines.span_annotator.load_config'), \
             patch('x_spanformer.pipelines.span_annotator.Path.exists', return_value=True), \
             patch('builtins.open'), \
             patch('yaml.safe_load') as mock_yaml, \
             patch('x_spanformer.pipelines.span_annotator.SpanAnnotatorSession'):
            
            mock_yaml.return_value = {
                "model": {"name": "test", "temperature": 0.7},
                "dialogue": {"max_turns": 5},
                "agent": {"early_termination": {}}
            }
            
            return SpanAnnotatorPipeline()
    
    def test_find_missing_sequences(self):
        """Test finding gaps in processed sequences."""
        # Create mock target sequences
        target_sequences = []
        for i in range(1, 11):  # Sequences 1-10
            mock_record = Mock()
            mock_record.sequence_number = i
            target_sequences.append(mock_record)
        
        # Mock existing results with gaps
        existing_results = {
            1: "completed",
            2: "completed", 
            # 3 is missing (gap)
            4: "completed",
            5: "failed",
            # 6-10 are new (not processed yet)
        }
        
        missing = self.pipeline.find_missing_sequences(target_sequences, existing_results)
        
        # Should find sequence 3 as a gap (within processed range 1-5)
        # Sequences 6-10 are not gaps, they're just new
        assert 3 in missing
        assert len([seq for seq in missing if seq <= 5]) == 1  # Only one gap in processed range


if __name__ == "__main__":
    pytest.main([__file__])
