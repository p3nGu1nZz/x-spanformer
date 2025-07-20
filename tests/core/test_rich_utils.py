"""
Test suite for x_spanformer.agents.rich_utils module.

Tests Rich console utilities including:
- Logging functions with panels
- Telemetry display
- Summary panels
- Console formatting utilities
- Error handling in display functions
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import Counter
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.agents import rich_utils


class TestRichUtils:
    """Test rich_utils functionality."""
    
    def setup_method(self):
        # Capture console output for testing
        self.captured_output = StringIO()
        
    def test_rich_log_basic(self):
        """Test basic rich logging functionality."""
        test_data = {
            "status": "success",
            "count": 42,
            "message": "Processing complete"
        }
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.rich_log(test_data)
            mock_print.assert_called_once()
            
            # Check that Panel was created with correct content
            call_args = mock_print.call_args[0]
            panel = call_args[0]
            assert hasattr(panel, 'renderable')  # Panel should have renderable content
    
    def test_rich_log_custom_title_color(self):
        """Test rich logging with custom title and color."""
        test_data = {"key": "value"}
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.rich_log(test_data, title="Custom Title", color="red")
            mock_print.assert_called_once()
    
    def test_rich_log_empty_data(self):
        """Test rich logging with empty data dictionary."""
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.rich_log({})
            mock_print.assert_called_once()
    
    def test_rich_log_complex_values(self):
        """Test rich logging with various data types."""
        test_data = {
            "string": "test",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
            "list": [1, 2, 3],
            "none_value": None
        }
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.rich_log(test_data)
            mock_print.assert_called_once()
    
    def test_display_telemetry_panel_basic(self):
        """Test basic telemetry panel display."""
        start_time = time.time() - 60  # 1 minute ago
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_telemetry_panel(
                processed_count=50,
                total_count=100,
                start_time=start_time,
                save_count=5,
                estimated_total_saves=10,
                records_saved_this_session=25
            )
            assert mock_print.call_count >= 1  # Should print at least once (panel content)
    
    def test_display_telemetry_panel_with_keep_count(self):
        """Test telemetry panel with keep count."""
        start_time = time.time() - 30  # 30 seconds ago
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_telemetry_panel(
                processed_count=25,
                total_count=100,
                start_time=start_time,
                save_count=2,
                estimated_total_saves=8,
                records_saved_this_session=15,
                keep_count=20,
                session_start_count=10
            )
            assert mock_print.call_count >= 1  # Should print at least once
    
    def test_display_telemetry_panel_zero_time(self):
        """Test telemetry panel with zero elapsed time."""
        start_time = time.time()  # Current time
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_telemetry_panel(
                processed_count=1,
                total_count=100,
                start_time=start_time,
                save_count=0,
                estimated_total_saves=10,
                records_saved_this_session=0
            )
            assert mock_print.call_count >= 1  # Should print at least once
    
    def test_display_telemetry_panel_complete(self):
        """Test telemetry panel when processing is complete."""
        start_time = time.time() - 120  # 2 minutes ago
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_telemetry_panel(
                processed_count=100,
                total_count=100,
                start_time=start_time,
                save_count=10,
                estimated_total_saves=10,
                records_saved_this_session=100
            )
            assert mock_print.call_count >= 1  # Should print at least once
    
    def test_display_summary_panel_basic(self):
        """Test basic summary panel display."""
        stats = Counter({"keep": 75, "discard": 25})
        reasons = ["good quality"] * 75 + ["poor quality"] * 25
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_summary_panel("test.csv", stats, reasons)
            mock_print.assert_called()  # May be called multiple times for table and panel
    
    def test_display_summary_panel_empty_stats(self):
        """Test summary panel with empty statistics."""
        stats = Counter()
        reasons = []
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_summary_panel("empty.csv", stats, reasons)
            mock_print.assert_called()
    
    def test_display_summary_panel_single_status(self):
        """Test summary panel with only one status type."""
        stats = Counter({"keep": 100})
        reasons = ["excellent quality"] * 100
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_summary_panel("single_status.csv", stats, reasons)
            mock_print.assert_called()
    
    def test_display_summary_panel_multiple_statuses(self):
        """Test summary panel with multiple status types."""
        stats = Counter({
            "keep": 50,
            "discard": 30,
            "review": 15,
            "pending": 5
        })
        reasons = ["various reasons"] * 100
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_summary_panel("multi_status.csv", stats, reasons)
            mock_print.assert_called()
    
    def test_display_summary_panel_long_filename(self):
        """Test summary panel with long filename."""
        stats = Counter({"keep": 10, "discard": 5})
        reasons = ["reason"] * 15
        long_filename = "very_long_filename_that_might_affect_display_formatting.csv"
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.display_summary_panel(long_filename, stats, reasons)
            mock_print.assert_called()
    
    def test_console_instance(self):
        """Test that console instance is properly initialized."""
        assert hasattr(rich_utils, 'console')
        assert rich_utils.console is not None
        # Should be a Rich Console instance
        from rich.console import Console
        assert isinstance(rich_utils.console, Console)


class TestRichUtilsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_display_telemetry_negative_values(self):
        """Test telemetry display with negative or invalid values."""
        start_time = time.time() + 60  # Future time (invalid)
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            # Should handle negative elapsed time gracefully
            rich_utils.display_telemetry_panel(
                processed_count=-1,  # Invalid negative count
                total_count=0,       # Zero total
                start_time=start_time,
                save_count=-1,       # Invalid negative save count
                estimated_total_saves=0,
                records_saved_this_session=-5  # Invalid negative
            )
            assert mock_print.call_count >= 1  # Should print at least once
    
    def test_display_summary_panel_none_values(self):
        """Test summary panel with None values."""
        with patch.object(rich_utils.console, 'print') as mock_print:
            # Should handle None values gracefully without crashing
            # Type ignore since we're testing edge case handling
            rich_utils.display_summary_panel(None, None, None)  # type: ignore
            mock_print.assert_called()
    
    def test_rich_log_non_string_keys(self):
        """Test rich log with non-string dictionary keys."""
        test_data = {
            123: "numeric key",
            (1, 2): "tuple key",
            None: "none key"
        }
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            rich_utils.rich_log(test_data)
            mock_print.assert_called_once()
    
    def test_display_functions_with_console_error(self):
        """Test that display functions handle console errors gracefully."""
        with patch.object(rich_utils.console, 'print', side_effect=Exception("Console error")):
            # Functions should either handle exceptions or let them propagate
            with pytest.raises(Exception):  # Expected to raise since no error handling in implementation
                rich_utils.rich_log({"test": "data"})
                
        # Test the other functions separately    
        with patch.object(rich_utils.console, 'print', side_effect=Exception("Console error")):
            with pytest.raises(Exception):
                rich_utils.display_telemetry_panel(1, 10, time.time(), 1, 2, 1)
                
        with patch.object(rich_utils.console, 'print', side_effect=Exception("Console error")):        
            with pytest.raises(Exception):
                rich_utils.display_summary_panel("test", Counter(), [])


class TestRichUtilsIntegration:
    """Integration tests for rich_utils components."""
    
    def test_telemetry_and_summary_together(self):
        """Test using telemetry and summary panels together."""
        start_time = time.time() - 45
        stats = Counter({"keep": 80, "discard": 20})
        reasons = ["good"] * 80 + ["poor"] * 20
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            # Display telemetry
            rich_utils.display_telemetry_panel(
                processed_count=100,
                total_count=100,
                start_time=start_time,
                save_count=10,
                estimated_total_saves=10,
                records_saved_this_session=100,
                keep_count=80
            )
            
            # Display summary
            rich_utils.display_summary_panel("completed.csv", stats, reasons)
            
            # Both should have been called
            assert mock_print.call_count >= 2
    
    def test_multiple_rich_logs(self):
        """Test multiple rich log calls."""
        data_sets = [
            {"stage": "preprocessing", "status": "started"},
            {"stage": "processing", "progress": "50%"},
            {"stage": "postprocessing", "status": "complete"}
        ]
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            for data in data_sets:
                rich_utils.rich_log(data)
            
            assert mock_print.call_count == 3
    
    def test_realistic_processing_scenario(self):
        """Test a realistic processing scenario with all components."""
        start_time = time.time() - 300  # 5 minutes ago
        
        with patch.object(rich_utils.console, 'print') as mock_print:
            # Initial log
            rich_utils.rich_log({"status": "starting", "files": 5})
            
            # Mid-processing telemetry
            rich_utils.display_telemetry_panel(
                processed_count=250,
                total_count=500,
                start_time=start_time,
                save_count=25,
                estimated_total_saves=50,
                records_saved_this_session=200,
                keep_count=180,
                session_start_count=50
            )
            
            # Final summary
            final_stats = Counter({"keep": 350, "discard": 150})
            final_reasons = ["high quality"] * 350 + ["low quality"] * 150
            rich_utils.display_summary_panel("final_results.csv", final_stats, final_reasons)
            
            # Completion log
            rich_utils.rich_log({
                "status": "completed",
                "total_processed": 500,
                "keep_rate": "70%",
                "duration": "5 minutes"
            })
            
            # Should have made multiple calls
            assert mock_print.call_count >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
