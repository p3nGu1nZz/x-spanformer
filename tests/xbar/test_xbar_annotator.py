#!/usr/bin/env python3
"""
Tests for XBarAnnotator - comprehensive X-bar theory span annotator.

This module tests the main annotation logic, including:
- Model configuration
- Domain detection
- System prompt building
- Span parsing and validation
- JSON response handling
- Annotation pipeline
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from x_spanformer.xbar.xbar_annotator import XBarAnnotator, ModelConfig
from x_spanformer.xbar.xbar_map import XBarLabelMap, DomainType
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord, SpanAnnotation
from x_spanformer.schema.span import SpanLabel


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_defaults(self):
        """Test that ModelConfig has correct default values."""
        config = ModelConfig()
        assert config.name == "llama3.2:3b"
        assert config.temperature == 0.2
        assert config.timeout == 180.0
    
    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            name="llama3.1:8b",
            temperature=0.5,
            timeout=300.0
        )
        assert config.name == "llama3.1:8b"
        assert config.temperature == 0.5
        assert config.timeout == 300.0


class TestXBarAnnotator:
    """Test XBarAnnotator functionality."""
    
    @pytest.fixture
    def annotator(self):
        """Create a test annotator instance."""
        config = ModelConfig(name="test-model", temperature=0.1, timeout=60.0)
        return XBarAnnotator(config)
    
    @pytest.fixture
    def sample_pretrain_record(self):
        """Create a sample PretrainRecord for testing."""
        return PretrainRecord(
            raw="The quick brown fox jumps over the lazy dog.",
            sequence_number=1,
            type="natural"
        )
    
    def test_annotator_initialization(self, annotator):
        """Test that XBarAnnotator initializes correctly."""
        assert isinstance(annotator, XBarAnnotator)
        assert annotator.model_config.name == "test-model"
        assert annotator.model_config.temperature == 0.1
        assert annotator.model_config.timeout == 60.0
    
    def test_detect_domain_from_record_natural(self, annotator):
        """Test domain detection for natural language."""
        record = PretrainRecord(raw="Hello world", type="natural")
        domain = annotator._detect_domain_from_record(record)
        assert domain == DomainType.NATURAL
    
    def test_detect_domain_from_record_code(self, annotator):
        """Test domain detection for code."""
        record = PretrainRecord(raw="def hello():", type="code")
        domain = annotator._detect_domain_from_record(record)
        assert domain == DomainType.CODE
    
    def test_detect_domain_from_record_mixed(self, annotator):
        """Test domain detection for mixed content."""
        record = PretrainRecord(raw="# This is a function\ndef hello():", type="mixed")
        domain = annotator._detect_domain_from_record(record)
        assert domain == DomainType.MIXED
    
    def test_detect_domain_from_record_invalid(self, annotator):
        """Test domain detection with invalid type defaults to natural."""
        record = PretrainRecord(raw="Some text", type="invalid")
        domain = annotator._detect_domain_from_record(record)
        assert domain == DomainType.NATURAL
    
    def test_detect_domain_from_record_missing(self, annotator):
        """Test domain detection with missing type defaults to natural."""
        record = PretrainRecord(raw="Some text")
        domain = annotator._detect_domain_from_record(record)
        assert domain == DomainType.NATURAL
    
    def test_build_system_prompt_natural(self, annotator):
        """Test system prompt building for natural domain."""
        prompt = annotator._build_system_prompt(DomainType.NATURAL)
        
        assert "natural text analysis" in prompt
        assert "noun:" in prompt
        assert "verb:" in prompt
        assert "adjective:" in prompt
        assert "Focus on accuracy and consistency" in prompt
    
    def test_build_system_prompt_code(self, annotator):
        """Test system prompt building for code domain."""
        prompt = annotator._build_system_prompt(DomainType.CODE)
        
        assert "code text analysis" in prompt
        assert "keyword:" in prompt
        assert "identifier:" in prompt
        assert "operator:" in prompt
        assert "Focus on accuracy and consistency" in prompt
    
    def test_build_system_prompt_mixed(self, annotator):
        """Test system prompt building for mixed domain."""
        prompt = annotator._build_system_prompt(DomainType.MIXED)
        
        assert "mixed text analysis" in prompt
        assert "noun:" in prompt  # Natural labels
        assert "keyword:" in prompt  # Code labels
        assert "inline_code:" in prompt  # Mixed labels
        assert "Focus on accuracy and consistency" in prompt


class TestSpanParsing:
    """Test span parsing functionality."""
    
    @pytest.fixture
    def annotator(self):
        """Create a test annotator instance."""
        return XBarAnnotator(ModelConfig())
    
    def test_parse_json_response_valid_array(self, annotator):
        """Test parsing valid JSON array response."""
        response = '''```json
[
    {"text": "The", "xbar_label": "determiner"},
    {"text": "fox", "xbar_label": "noun"}
]
```'''
        
        annotations = annotator._parse_json_response(response)
        assert len(annotations) == 2
        assert annotations[0]["text"] == "The"
        assert annotations[0]["xbar_label"] == "determiner"
        assert annotations[1]["text"] == "fox"
        assert annotations[1]["xbar_label"] == "noun"
    
    def test_parse_json_response_single_object(self, annotator):
        """Test parsing single JSON object response."""
        response = '''```json
{"text": "running", "xbar_label": "verb"}
```'''
        
        annotations = annotator._parse_json_response(response)
        assert len(annotations) == 1
        assert annotations[0]["text"] == "running"
        assert annotations[0]["xbar_label"] == "verb"
    
    def test_parse_json_response_malformed(self, annotator):
        """Test parsing malformed JSON with recovery."""
        response = '''```json
[
    {"text": "The", "xbar_label": "determiner",},
    {"text": "fox", "xbar_label": "noun"}
]
```'''
        
        annotations = annotator._parse_json_response(response)
        assert len(annotations) >= 1  # Should recover at least some data
    
    def test_parse_spans_from_response_with_json(self, annotator):
        """Test parsing spans from JSON response."""
        response = '''```json
[
    {"text": "The", "xbar_label": "determiner"},
    {"text": "fox", "xbar_label": "noun"}
]
```'''
        text = "The quick brown fox"
        
        spans = annotator._parse_spans_from_response(response, text)
        assert len(spans) == 2
        
        # Check first span
        assert spans[0].text == "The"
        assert spans[0].xbar_label == "determiner"
        assert spans[0].span == (0, 2)  # "The" is at positions 0-2
        
        # Check second span
        assert spans[1].text == "fox"
        assert spans[1].xbar_label == "noun"
        assert spans[1].span == (16, 18)  # "fox" is at positions 16-18
    
    def test_parse_spans_from_response_text_not_found(self, annotator):
        """Test parsing when span text is not found in source."""
        response = '''```json
[
    {"text": "elephant", "xbar_label": "noun"}
]
```'''
        text = "The quick brown fox"
        
        spans = annotator._parse_spans_from_response(response, text)
        assert len(spans) == 0  # No spans should be found
    
    def test_parse_spans_from_response_case_insensitive(self, annotator):
        """Test parsing with case-insensitive matching."""
        response = '''```json
[
    {"text": "THE", "xbar_label": "determiner"}
]
```'''
        text = "The quick brown fox"
        
        spans = annotator._parse_spans_from_response(response, text)
        assert len(spans) == 1
        assert spans[0].text == "THE"
        assert spans[0].span == (0, 2)  # Should find "The" despite case difference
    
    def test_parse_spans_fallback_text_format(self, annotator):
        """Test parsing with fallback text format."""
        response = '"The" (0-2) -> determiner\n"fox" (16-18) -> noun'
        text = "The quick brown fox"
        
        spans = annotator._parse_spans_from_response(response, text)
        assert len(spans) == 2
        assert spans[0].text == "The"
        assert spans[0].xbar_label == "determiner"
        assert spans[1].text == "fox"
        assert spans[1].xbar_label == "noun"


class TestSpanValidation:
    """Test span validation functionality."""
    
    @pytest.fixture
    def annotator(self):
        """Create a test annotator instance."""
        return XBarAnnotator(ModelConfig())
    
    def test_validate_and_filter_span_labels_valid(self, annotator):
        """Test validation of valid spans."""
        spans = [
            SpanLabel(span=(0, 2), xbar_label="determiner", text="The"),
            SpanLabel(span=(4, 8), xbar_label="adjective", text="quick"),
            SpanLabel(span=(16, 18), xbar_label="noun", text="fox")
        ]
        text = "The quick brown fox jumps"
        
        valid_spans = annotator._validate_and_filter_span_labels(spans, text)
        assert len(valid_spans) == 3
    
    def test_validate_and_filter_span_labels_invalid_positions(self, annotator):
        """Test filtering of spans with invalid positions."""
        spans = [
            SpanLabel(span=(-1, 2), xbar_label="determiner", text="The"),  # Negative start
            SpanLabel(span=(0, 100), xbar_label="noun", text="fox"),  # End beyond text
            SpanLabel(span=(5, 2), xbar_label="adjective", text="quick")  # Start > end
        ]
        text = "The quick brown fox"
        
        valid_spans = annotator._validate_and_filter_span_labels(spans, text)
        assert len(valid_spans) == 0
    
    def test_validate_and_filter_span_labels_missing_data(self, annotator):
        """Test filtering of spans with missing text or label."""
        spans = [
            SpanLabel(span=(0, 2), xbar_label="", text="The"),  # Missing label
            SpanLabel(span=(4, 8), xbar_label="adjective", text=""),  # Missing text
            SpanLabel(span=(16, 18), xbar_label="noun", text="fox")  # Valid
        ]
        text = "The quick brown fox"
        
        valid_spans = annotator._validate_and_filter_span_labels(spans, text)
        assert len(valid_spans) == 1
        assert valid_spans[0].text == "fox"
    
    def test_validate_and_filter_span_labels_duplicates(self, annotator):
        """Test deduplication of spans."""
        spans = [
            SpanLabel(span=(0, 2), xbar_label="determiner", text="The"),
            SpanLabel(span=(0, 2), xbar_label="determiner", text="The"),  # Duplicate
            SpanLabel(span=(16, 18), xbar_label="noun", text="fox")
        ]
        text = "The quick brown fox"
        
        valid_spans = annotator._validate_and_filter_span_labels(spans, text)
        assert len(valid_spans) == 2  # Duplicate should be removed


class TestSpanConversion:
    """Test span conversion functionality."""
    
    @pytest.fixture
    def annotator(self):
        """Create a test annotator instance."""
        return XBarAnnotator(ModelConfig())
    
    def test_convert_span_labels_to_annotations(self, annotator):
        """Test conversion of SpanLabel to SpanAnnotation."""
        spans = [
            SpanLabel(span=(0, 2), xbar_label="determiner", text="The"),
            SpanLabel(span=(16, 18), xbar_label="noun", text="fox")
        ]
        text = "The quick brown fox"
        
        annotations = annotator._convert_span_labels_to_annotations(spans, text)
        assert len(annotations) == 2
        
        # Check first annotation
        assert annotations[0].start_pos == 0
        assert annotations[0].end_pos == 3  # Exclusive end (2 + 1)
        assert annotations[0].xbar_label == "determiner"
        assert annotations[0].linguistic_features["extracted_text"] == "The"
        
        # Check second annotation
        assert annotations[1].start_pos == 16
        assert annotations[1].end_pos == 19  # Exclusive end (18 + 1)
        assert annotations[1].xbar_label == "noun"
        assert annotations[1].linguistic_features["extracted_text"] == "fox"


class TestUtilityMethods:
    """Test utility methods."""
    
    @pytest.fixture
    def annotator(self):
        """Create a test annotator instance."""
        return XBarAnnotator(ModelConfig())
    
    def test_normalize_xbar_class_standard_labels(self, annotator):
        """Test normalization of standard X-bar labels."""
        assert annotator._normalize_xbar_class("noun") == "noun"
        assert annotator._normalize_xbar_class("verb") == "verb"
        assert annotator._normalize_xbar_class("Adjective") == "adjective"
        assert annotator._normalize_xbar_class("DETERMINER") == "determiner"
    
    def test_normalize_xbar_class_phrases(self, annotator):
        """Test normalization of phrase labels."""
        assert annotator._normalize_xbar_class("noun phrase") == "noun_phrase"
        assert annotator._normalize_xbar_class("verb phrase") == "verb_phrase"
        assert annotator._normalize_xbar_class("prepositional phrase") == "prepositional_phrase"
    
    def test_normalize_xbar_class_abbreviations(self, annotator):
        """Test normalization of abbreviations."""
        assert annotator._normalize_xbar_class("np") == "noun_phrase"
        assert annotator._normalize_xbar_class("vp") == "verb_phrase"
        assert annotator._normalize_xbar_class("pp") == "prepositional_phrase"
        assert annotator._normalize_xbar_class("n") == "noun"
        assert annotator._normalize_xbar_class("v") == "verb"
    
    def test_normalize_xbar_class_empty_or_invalid(self, annotator):
        """Test normalization of empty or invalid labels."""
        assert annotator._normalize_xbar_class("") == "unknown"
        assert annotator._normalize_xbar_class(None) == "unknown"
        assert annotator._normalize_xbar_class("invalid_label") == "invalid_label"
    
    def test_extract_text_boundaries(self, annotator):
        """Test text boundary extraction."""
        text = "The quick brown fox jumps over the lazy dog"
        
        # Test exact match
        boundaries = annotator._extract_text_boundaries(text, "fox")
        assert len(boundaries) == 1
        assert boundaries[0] == (16, 19)  # "fox" positions
        
        # Test multiple occurrences (only finds case-insensitive match)
        boundaries = annotator._extract_text_boundaries(text, "the")
        assert len(boundaries) == 1  # Only finds lowercase "the", not "The"
        
        # Test not found
        boundaries = annotator._extract_text_boundaries(text, "elephant")
        assert len(boundaries) == 0
    
    def test_validate_span_boundaries(self, annotator):
        """Test span boundary validation."""
        text = "Hello world"
        
        # Valid boundaries
        is_valid, start, end = annotator._validate_span_boundaries(text, 0, 4, "Hello")
        assert is_valid
        assert start == 0
        assert end == 4
        
        # Negative start
        is_valid, start, end = annotator._validate_span_boundaries(text, -1, 4, "Hello")
        assert is_valid
        assert start == 0
        
        # End beyond text - when span_text is provided, it finds the correct boundaries
        is_valid, start, end = annotator._validate_span_boundaries(text, 0, 20, "Hello")
        assert is_valid
        assert start == 0  # Found "Hello" at correct position
        assert end == 4   # Found "Hello" end at correct position
        
        # Invalid range (start > end) - should find "Hello" via text boundaries
        is_valid, start, end = annotator._validate_span_boundaries(text, 5, 2, "Hello")
        assert is_valid  # Should find "Hello" at correct position
        assert start == 0  # "Hello" starts at 0
        assert end == 4   # "Hello" ends at 4


@pytest.mark.asyncio
class TestAnnotationPipeline:
    """Test the full annotation pipeline (requires mocking external dependencies)."""
    
    @pytest.fixture
    def annotator(self):
        """Create a test annotator instance."""
        return XBarAnnotator(ModelConfig(name="test-model", timeout=10.0))
    
    @pytest.fixture
    def sample_record(self):
        """Create a sample record for testing."""
        return PretrainRecord(
            raw="The fox runs quickly.",
            sequence_number=1,
            type="natural"
        )
    
    async def test_extract_spans_via_dialogue_error_handling(self, annotator):
        """Test span extraction error handling."""
        with patch('x_spanformer.agents.ollama_client.chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = Exception("Connection error")
            
            spans = await annotator._extract_spans_via_dialogue(
                "The fox runs", 
                DomainType.NATURAL, 
                "word_level"
            )
            
            # Should return empty list on error but not raise
            assert spans == []
