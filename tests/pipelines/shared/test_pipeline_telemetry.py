#!/usr/bin/env python3
"""
Tests for pipeline_telemetry.py shared module.

Tests the PipelineTelemetry and SpanAnnotationTelemetry classes for
centralized telemetry tracking across X-Spanformer pipelines.
"""
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from x_spanformer.pipelines.shared.pipeline_telemetry import (
    PipelineTelemetry, 
    SpanAnnotationTelemetry
)


class TestPipelineTelemetry:
    """Test cases for the base PipelineTelemetry class."""
    
    def test_initialization(self):
        """Test telemetry initialization with default values."""
        telemetry = PipelineTelemetry("Test Pipeline")
        
        assert telemetry.pipeline_name == "Test Pipeline"
        assert telemetry.telemetry["start_time"] is None
        assert telemetry.telemetry["completed_sequences"] == 0
        assert telemetry.telemetry["failed_sequences"] == 0
        assert telemetry.telemetry["total_sequences"] == 0
        assert isinstance(telemetry.telemetry["spans_by_type"], dict)
        assert isinstance(telemetry.telemetry["spans_by_modality"], dict)
        assert isinstance(telemetry.telemetry["sequence_times"], list)
    
    def test_initialize_with_values(self):
        """Test telemetry initialization with specific values."""
        telemetry = PipelineTelemetry("Test Pipeline")
        
        # Mock datetime.now to have predictable start time
        with patch('x_spanformer.pipelines.shared.pipeline_telemetry.datetime') as mock_datetime:
            mock_now = datetime(2025, 8, 3, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            telemetry.initialize(
                total_sequences=100,
                existing_completed=25,
                existing_failed=5
            )
        
        assert telemetry.telemetry["start_time"] == mock_now
        assert telemetry.telemetry["total_sequences"] == 100
        assert telemetry.telemetry["completed_sequences"] == 25
        assert telemetry.telemetry["failed_sequences"] == 5
        assert telemetry.telemetry["spans_by_type"] == {}
        assert telemetry.telemetry["spans_by_modality"] == {}
        assert telemetry.telemetry["sequence_times"] == []
    
    def test_update_on_completion_basic(self):
        """Test basic completion update without annotation result."""
        telemetry = PipelineTelemetry("Test Pipeline")
        telemetry.initialize(100)
        
        initial_completed = telemetry.telemetry["completed_sequences"]
        
        # Mock datetime for sequence timing
        with patch('x_spanformer.pipelines.shared.pipeline_telemetry.datetime') as mock_datetime:
            start_time = datetime(2025, 8, 3, 12, 0, 0)
            end_time = datetime(2025, 8, 3, 12, 0, 10)  # 10 seconds later
            mock_datetime.now.return_value = end_time
            
            telemetry.update_on_completion(None, start_time)
        
        assert telemetry.telemetry["completed_sequences"] == initial_completed + 1
        assert len(telemetry.telemetry["sequence_times"]) == 1
        assert telemetry.telemetry["sequence_times"][0] == 10.0  # 10 seconds
        assert telemetry.telemetry["last_sequence_time"] == 10.0
    
    def test_update_on_completion_with_annotation_result(self):
        """Test completion update with annotation result containing spans."""
        telemetry = PipelineTelemetry("Test Pipeline")
        telemetry.initialize(100)
        
        # Create mock annotation result with spans
        mock_annotation_result = Mock()
        mock_annotation_result.span_annotations = []
        
        # Create mock spans
        span1 = Mock()
        span1.xbar_class = "NP"
        span2 = Mock()
        span2.xbar_class = "VP"
        span3 = Mock()
        span3.xbar_class = "NP"  # Duplicate to test counting
        
        mock_annotation_result.span_annotations = [span1, span2, span3]
        
        telemetry.update_on_completion(mock_annotation_result)
        
        # Check span counting by type
        assert telemetry.telemetry["spans_by_type"]["NP"] == 2
        assert telemetry.telemetry["spans_by_type"]["VP"] == 1
        
        # Check span counting by modality (using _infer_span_modality)
        assert "syntactic" in telemetry.telemetry["spans_by_modality"]
    
    def test_update_on_failure(self):
        """Test failure update with timing."""
        telemetry = PipelineTelemetry("Test Pipeline")
        telemetry.initialize(100)
        
        initial_failed = telemetry.telemetry["failed_sequences"]
        
        # Mock datetime for sequence timing
        with patch('x_spanformer.pipelines.shared.pipeline_telemetry.datetime') as mock_datetime:
            start_time = datetime(2025, 8, 3, 12, 0, 0)
            end_time = datetime(2025, 8, 3, 12, 0, 5)  # 5 seconds later
            mock_datetime.now.return_value = end_time
            
            telemetry.update_on_failure(start_time)
        
        assert telemetry.telemetry["failed_sequences"] == initial_failed + 1
        assert len(telemetry.telemetry["sequence_times"]) == 1
        assert telemetry.telemetry["sequence_times"][0] == 5.0  # 5 seconds
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        telemetry = PipelineTelemetry("Test Pipeline")
        
        # Initialize with mock start time
        with patch('x_spanformer.pipelines.shared.pipeline_telemetry.datetime') as mock_datetime:
            start_time = datetime(2025, 8, 3, 12, 0, 0)
            mock_datetime.now.return_value = start_time
            telemetry.initialize(100, existing_completed=10, existing_failed=2)
            
            # Simulate some processing time
            current_time = datetime(2025, 8, 3, 12, 5, 0)  # 5 minutes later
            mock_datetime.now.return_value = current_time
            
            # Add some sequence times - these represent current session sequences
            telemetry.telemetry["sequence_times"] = [5.0, 8.0, 6.0]  # 3 sequences processed in current session
            
            stats = telemetry.get_statistics()
        
        assert stats["total_sequences"] == 100
        assert stats["completed_sequences"] == 10
        assert stats["failed_sequences"] == 2
        assert stats["processed_sequences"] == 12
        assert stats["success_rate_percent"] == pytest.approx(83.33, rel=1e-2)
        assert stats["elapsed_time_minutes"] == pytest.approx(5.0)
        # Processing rate is now based on improved logic:
        # - Total processed = 10 (existing) + 3 (current session) = 13
        # - Overall rate = 13 sequences / 5 minutes = 2.6 seq/min
        # - Current session rate = 3 sequences / 5 minutes = 0.6 seq/min
        # - Since total_processed >= 5, uses max(overall_rate, current_session_rate * 0.8)
        # - max(2.6, 0.6 * 0.8) = max(2.6, 0.48) = 2.6 seq/min
        assert stats["processing_rate_per_min"] == pytest.approx(2.6)  # Uses improved calculation
        assert stats["average_sequence_time_seconds"] == pytest.approx(6.33, rel=1e-2)
        assert stats["current_session_processed"] == 3  # New field for current session count
    
    def test_infer_span_modality(self):
        """Test span modality inference."""
        telemetry = PipelineTelemetry("Test Pipeline")
        
        # Create mock spans with different xbar_class values
        span_punct = Mock()
        span_punct.xbar_class = "PUNCT"
        
        span_lexical = Mock()
        span_lexical.xbar_class = "NOUN"
        
        span_syntactic = Mock()
        span_syntactic.xbar_class = "NP"
        
        span_structural = Mock()
        span_structural.xbar_class = "SENTENCE"
        
        span_other = Mock()
        span_other.xbar_class = "UNKNOWN"
        
        span_none = Mock()
        del span_none.xbar_class  # No xbar_class attribute
        
        # Test modality inference
        assert telemetry._infer_span_modality(span_punct) == "punctuation"
        assert telemetry._infer_span_modality(span_lexical) == "lexical"
        assert telemetry._infer_span_modality(span_syntactic) == "syntactic"
        assert telemetry._infer_span_modality(span_structural) == "structural"
        assert telemetry._infer_span_modality(span_other) == "other"
        assert telemetry._infer_span_modality(span_none) == "other"
    
    def test_format_eta(self):
        """Test ETA formatting."""
        telemetry = PipelineTelemetry("Test Pipeline")
        
        # Test minutes
        assert telemetry._format_eta(45.5) == "45.5 minutes"
        
        # Test hours and minutes
        assert telemetry._format_eta(125.0) == "2h 5m"
        assert telemetry._format_eta(90.0) == "1h 30m"
    
    def test_format_span_summary(self):
        """Test span summary formatting."""
        telemetry = PipelineTelemetry("Test Pipeline")
        
        span_dict = {
            "NP": 15,
            "VP": 10,
            "PP": 5,
            "ADJP": 3,
            "ADVP": 1
        }
        
        summary = telemetry._format_span_summary(span_dict, max_items=3)
        
        # Should be sorted by count descending
        assert "NP: 15" in summary
        assert "VP: 10" in summary
        assert "PP: 5" in summary
        assert "... and 2 more" in summary
    
    @patch('x_spanformer.pipelines.shared.pipeline_telemetry.logger')
    def test_display_progress_panel(self, mock_logger):
        """Test progress panel display."""
        telemetry = PipelineTelemetry("Test Pipeline")
        
        # Initialize with mock data
        with patch('x_spanformer.pipelines.shared.pipeline_telemetry.datetime') as mock_datetime:
            start_time = datetime(2025, 8, 3, 12, 0, 0)
            current_time = datetime(2025, 8, 3, 12, 10, 0)  # 10 minutes later
            
            mock_datetime.now.side_effect = [start_time, current_time]
            
            telemetry.initialize(100, existing_completed=25, existing_failed=5)
            telemetry.telemetry["spans_by_type"] = {"NP": 50, "VP": 30}
            telemetry.telemetry["spans_by_modality"] = {"syntactic": 60, "lexical": 20}
            # Add some sequence times to simulate current session activity
            telemetry.telemetry["sequence_times"] = [5.0, 8.0, 6.0]  # 3 sequences in current session
            
            # Call display
            telemetry.display_progress_panel()
        
        # Verify logger was called with expected content
        assert mock_logger.info.called
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        
        # Check that key information is logged with new format
        assert any("[TELEMETRY] Test Pipeline Progress Panel" in arg for arg in call_args)
        assert any("Overall Progress: 30/100" in arg for arg in call_args)  # 25 completed + 5 failed
        assert any("Success Rate: 25/30" in arg for arg in call_args)
        assert any("Current Session: 3 sequences processed" in arg for arg in call_args)  # New session info


class TestSpanAnnotationTelemetry:
    """Test cases for the SpanAnnotationTelemetry specialized class."""
    
    def test_initialization(self):
        """Test SpanAnnotationTelemetry initialization."""
        telemetry = SpanAnnotationTelemetry()
        
        assert telemetry.pipeline_name == "Span Annotation"
        assert isinstance(telemetry.telemetry, dict)
    
    @patch('x_spanformer.pipelines.shared.pipeline_telemetry.logger')
    def test_update_on_completion_with_logging(self, mock_logger):
        """Test span annotation specific completion logging."""
        telemetry = SpanAnnotationTelemetry()
        telemetry.initialize(100)
        
        # Create mock annotation result
        mock_annotation_result = Mock()
        mock_annotation_result.span_annotations = [Mock(), Mock(), Mock()]  # 3 spans
        mock_annotation_result.sequence_id = 42
        
        telemetry.update_on_completion(mock_annotation_result)
        
        # Verify debug logging was called
        mock_logger.debug.assert_called_with("Sequence 42: extracted 3 spans")
    
    def test_update_on_completion_without_sequence_id(self):
        """Test completion update when annotation result has no sequence_id."""
        telemetry = SpanAnnotationTelemetry()
        telemetry.initialize(100)
        
        # Create mock annotation result without sequence_id
        mock_annotation_result = Mock()
        mock_annotation_result.span_annotations = [Mock()]
        del mock_annotation_result.sequence_id  # Remove sequence_id attribute
        
        # Should not raise an error, should use 'unknown'
        with patch('x_spanformer.pipelines.shared.pipeline_telemetry.logger') as mock_logger:
            telemetry.update_on_completion(mock_annotation_result)
            mock_logger.debug.assert_called_with("Sequence unknown: extracted 1 spans")


class TestTelemetryIntegration:
    """Integration tests for telemetry functionality."""
    
    def test_full_pipeline_simulation(self):
        """Test a complete pipeline simulation."""
        telemetry = SpanAnnotationTelemetry()
        
        # Initialize
        with patch('x_spanformer.pipelines.shared.pipeline_telemetry.datetime') as mock_datetime:
            start_time = datetime(2025, 8, 3, 12, 0, 0)
            mock_datetime.now.return_value = start_time
            telemetry.initialize(total_sequences=10)
            
            # Simulate processing sequences
            sequence_times = [
                datetime(2025, 8, 3, 12, 0, 5),   # 5 seconds
                datetime(2025, 8, 3, 12, 0, 12),  # 7 seconds
                datetime(2025, 8, 3, 12, 0, 20),  # 8 seconds
                datetime(2025, 8, 3, 12, 0, 25),  # 5 seconds (failure)
            ]
            
            # Process 3 successful sequences
            for i, end_time in enumerate(sequence_times[:3]):
                mock_datetime.now.return_value = end_time
                
                # Create mock annotation result
                mock_result = Mock()
                mock_result.span_annotations = [Mock() for _ in range(i + 2)]  # 2, 3, 4 spans
                mock_result.sequence_id = i + 1
                
                seq_start = start_time if i == 0 else sequence_times[i-1]
                telemetry.update_on_completion(mock_result, seq_start)
            
            # Process 1 failed sequence
            mock_datetime.now.return_value = sequence_times[3]
            telemetry.update_on_failure(sequence_times[2])
            
            # Get final statistics
            mock_datetime.now.return_value = sequence_times[3]
            stats = telemetry.get_statistics()
        
        # Verify results
        assert stats["total_sequences"] == 10
        assert stats["completed_sequences"] == 3
        assert stats["failed_sequences"] == 1
        assert stats["processed_sequences"] == 4
        assert stats["success_rate_percent"] == 75.0
        assert stats["total_spans"] == 9  # 2 + 3 + 4 spans
        
        # Verify timing calculations
        assert len(telemetry.telemetry["sequence_times"]) == 4
        expected_times = [5.0, 7.0, 8.0, 5.0]  # seconds
        assert telemetry.telemetry["sequence_times"] == expected_times


if __name__ == "__main__":
    pytest.main([__file__])
