"""
Test suite for shared text_processor module.

Tests text processing utilities used across different pipeline implementations.
"""
import pytest
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.pipelines.shared.text_processor import (
    split_long_text,
    concatenate_small_segments,
    split_long_text_for_code,
    normalize_text_segments
)


class TestTextProcessor:
    """Test shared text processing functions."""
    
    def test_split_long_text_short_text(self):
        """Test splitting text that is already short enough."""
        text = "This is a short text."
        result = split_long_text(text, max_length=100)
        assert result == [text]
    
    def test_split_long_text_sentence_boundaries(self):
        """Test splitting on sentence boundaries."""
        text = "First sentence. Second sentence! Third sentence?"
        result = split_long_text(text, max_length=20)
        assert len(result) >= 3
        assert all(len(chunk) <= 20 for chunk in result)
    
    def test_split_long_text_word_boundaries(self):
        """Test splitting on word boundaries when sentences are too long."""
        text = "This is a very long sentence that should be split on word boundaries"
        result = split_long_text(text, max_length=20)
        assert len(result) > 1
        assert all(len(chunk) <= 20 for chunk in result)
    
    def test_split_long_text_character_level(self):
        """Test character-level splitting for very long words."""
        text = "verylongwordthatcannotbesplitonnormalboundaries" * 20
        result = split_long_text(text, max_length=50)
        assert len(result) > 1
        assert all(len(chunk) <= 50 for chunk in result)
    
    def test_concatenate_small_segments_basic(self):
        """Test basic concatenation of small segments."""
        spans = ["Short", "text here"]
        source_mapping = ["doc1.pdf", "doc1.pdf"]
        
        result_spans, result_sources = concatenate_small_segments(
            spans, source_mapping, min_length=15, max_length=100
        )
        
        assert len(result_spans) == 1
        assert result_spans[0] == "Short text here"
        assert result_sources[0] == "doc1.pdf"
    
    def test_concatenate_small_segments_different_sources(self):
        """Test that concatenation respects source boundaries."""
        spans = ["Short", "text"]
        source_mapping = ["doc1.pdf", "doc2.pdf"]
        
        result_spans, result_sources = concatenate_small_segments(
            spans, source_mapping, min_length=15, max_length=100
        )
        
        # Should not concatenate across different sources
        assert len(result_spans) == 2
        assert result_spans == ["Short", "text"]
        assert result_sources == ["doc1.pdf", "doc2.pdf"]
    
    def test_concatenate_small_segments_respects_max_length(self):
        """Test that concatenation respects max length."""
        spans = ["Short", "A" * 90]  # Second segment is 90 chars
        source_mapping = ["doc1.pdf", "doc1.pdf"]
        
        result_spans, result_sources = concatenate_small_segments(
            spans, source_mapping, min_length=15, max_length=100
        )
        
        # Should not concatenate because it would exceed max_length
        assert len(result_spans) == 2
        assert result_spans[0] == "Short"
        assert len(result_spans[1]) == 90
    
    def test_split_long_text_for_code(self):
        """Test code-specific text splitting."""
        code = "def function():\n    print('hello')\n    return True\n\ndef another():\n    pass"
        result = split_long_text_for_code(code, max_length=30)
        
        assert len(result) > 1
        assert all(len(chunk) <= 30 for chunk in result)
        # Should prefer splitting on newlines
        assert any('\n' not in chunk or chunk.count('\n') < code.count('\n') for chunk in result)
    
    def test_normalize_text_segments_natural_content(self):
        """Test normalize_text_segments with natural language content."""
        spans = ["Short", "Another short", "This is a much longer segment that should be kept as is"]
        source_mapping = ["doc1.pdf"] * 3
        
        result_spans, result_sources = normalize_text_segments(
            spans, source_mapping, max_length=100, min_length=20, content_type="natural"
        )
        
        # Should concatenate the first two short segments
        assert len(result_spans) == 2
        assert result_spans[0] == "Short Another short"  # Concatenated
        assert "This is a much longer segment" in result_spans[1]  # Long segment kept
    
    def test_normalize_text_segments_code_content(self):
        """Test normalize_text_segments with code content."""
        spans = ["short_var = 1", "another_var = 2", "A" * 600]  # Third is very long
        source_mapping = ["file.py"] * 3
        
        result_spans, result_sources = normalize_text_segments(
            spans, source_mapping, max_length=100, min_length=20, content_type="code"
        )
        
        # Should NOT concatenate small segments for code (preserves structure)
        # Should split the long segment
        assert len(result_spans) > 3  # Long segment was split
        assert "short_var = 1" in result_spans  # Short segments preserved
        assert "another_var = 2" in result_spans
    
    def test_normalize_text_segments_empty_input(self):
        """Test normalize_text_segments with empty input."""
        result_spans, result_sources = normalize_text_segments([], [])
        assert result_spans == []
        assert result_sources == []
    
    def test_normalize_text_segments_filters_empty_chunks(self):
        """Test that normalize_text_segments filters out empty chunks."""
        spans = ["Normal text", "  ", "\n\n", "Another normal text"]
        source_mapping = ["doc.pdf"] * 4
        
        result_spans, result_sources = normalize_text_segments(
            spans, source_mapping, min_length=5, max_length=100, content_type="natural"
        )
        
        # Should filter out whitespace-only chunks but keep both text chunks separate
        assert len(result_spans) == 2
        assert "Normal text" in result_spans
        assert "Another normal text" in result_spans


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
