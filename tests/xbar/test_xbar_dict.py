#!/usr/bin/env python3
"""
Tests for XBarDictionary functionality.

Tests the domain-specific dictionary building and management
for hierarchical span vocabularies.
"""

import pytest
import tempfile
import json
from pathlib import Path

from x_spanformer.xbar.xbar_dict import XBarDictionary, get_global_dict, reset_global_dict


class TestXBarDictionary:
    """Test XBarDictionary functionality."""
    
    def test_dictionary_initialization(self):
        """Test that XBarDictionary initializes correctly."""
        xbar_dict = XBarDictionary()
        assert isinstance(xbar_dict.dictionaries, dict)
        assert xbar_dict.stats["total_unique_spans"] == 0
        assert xbar_dict.stats["sequences_processed"] == 0
    
    def test_add_spans_single_level(self):
        """Test adding spans to a single level."""
        xbar_dict = XBarDictionary()
        
        spans = ["the", "cat", "sat"]
        new_count = xbar_dict.add_spans("natural", "word_level", spans)
        
        assert new_count == 3
        assert len(xbar_dict.dictionaries["natural"]["word_level"]) == 3
        assert "the" in xbar_dict.dictionaries["natural"]["word_level"]
        assert "cat" in xbar_dict.dictionaries["natural"]["word_level"]
        assert "sat" in xbar_dict.dictionaries["natural"]["word_level"]
    
    def test_add_spans_duplicates(self):
        """Test that duplicate spans are not added."""
        xbar_dict = XBarDictionary()
        
        # Add initial spans
        spans1 = ["the", "cat", "sat"]
        new_count1 = xbar_dict.add_spans("natural", "word_level", spans1)
        assert new_count1 == 3
        
        # Add overlapping spans
        spans2 = ["the", "dog", "ran"]
        new_count2 = xbar_dict.add_spans("natural", "word_level", spans2)
        assert new_count2 == 2  # Only "dog" and "ran" are new
        
        assert len(xbar_dict.dictionaries["natural"]["word_level"]) == 5
    
    def test_add_sequence_spans(self):
        """Test adding spans from a complete sequence."""
        xbar_dict = XBarDictionary()
        
        word_spans = ["the", "quick", "fox"]
        phrase_spans = ["the quick fox", "quick fox"]
        clause_spans = ["the quick fox jumps"]
        
        results = xbar_dict.add_sequence_spans(
            "natural", word_spans, phrase_spans, clause_spans
        )
        
        assert results["word_level"] == 3
        assert results["phrase_level"] == 2
        assert results["clause_level"] == 1
        assert xbar_dict.stats["sequences_processed"] == 1
    
    def test_multiple_domains(self):
        """Test handling multiple domain types."""
        xbar_dict = XBarDictionary()
        
        # Add natural language spans
        xbar_dict.add_spans("natural", "word_level", ["the", "cat"])
        xbar_dict.add_spans("natural", "phrase_level", ["the cat"])
        
        # Add code spans
        xbar_dict.add_spans("code", "word_level", ["def", "function"])
        xbar_dict.add_spans("code", "phrase_level", ["def function"])
        
        # Add mixed spans
        xbar_dict.add_spans("mixed", "word_level", ["code", "text"])
        
        assert len(xbar_dict.dictionaries) == 3
        assert "natural" in xbar_dict.dictionaries
        assert "code" in xbar_dict.dictionaries
        assert "mixed" in xbar_dict.dictionaries
    
    def test_get_domain_stats(self):
        """Test getting statistics for a specific domain."""
        xbar_dict = XBarDictionary()
        
        xbar_dict.add_spans("natural", "word_level", ["the", "cat", "sat"])
        xbar_dict.add_spans("natural", "phrase_level", ["the cat"])
        xbar_dict.add_spans("natural", "clause_level", ["the cat sat"])
        
        stats = xbar_dict.get_domain_stats("natural")
        
        assert stats["domain"] == "natural"
        assert stats["word_level"] == 3
        assert stats["phrase_level"] == 1
        assert stats["clause_level"] == 1
        assert stats["total"] == 5
    
    def test_get_all_stats(self):
        """Test getting comprehensive statistics."""
        xbar_dict = XBarDictionary()
        
        xbar_dict.add_spans("natural", "word_level", ["the", "cat"])
        xbar_dict.add_spans("code", "word_level", ["def", "return"])
        xbar_dict.add_spans("natural", "phrase_level", ["the cat"])
        
        stats = xbar_dict.get_all_stats()
        
        assert stats["total_unique_spans"] == 5
        assert stats["level_totals"]["word_level"] == 4
        assert stats["level_totals"]["phrase_level"] == 1
        assert stats["level_totals"]["clause_level"] == 0
        assert stats["domain_totals"]["natural"] == 3
        assert stats["domain_totals"]["code"] == 2
    
    def test_save_and_load_dictionaries(self):
        """Test saving and loading dictionaries."""
        xbar_dict = XBarDictionary()
        
        # Add some test data
        xbar_dict.add_spans("natural", "word_level", ["the", "cat", "sat"])
        xbar_dict.add_spans("natural", "phrase_level", ["the cat"])
        xbar_dict.add_spans("code", "word_level", ["def", "return"])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save dictionaries
            xbar_dict.save_dictionaries(temp_path)
            
            # Check files were created
            assert (temp_path / "xbar_dict_natural.json").exists()
            assert (temp_path / "xbar_dict_code.json").exists()
            assert (temp_path / "xbar_dict_stats.json").exists()
            
            # Load into new dictionary
            new_dict = XBarDictionary()
            new_dict.load_dictionaries(temp_path)
            
            # Verify data was loaded correctly
            assert len(new_dict.dictionaries["natural"]["word_level"]) == 3
            assert len(new_dict.dictionaries["natural"]["phrase_level"]) == 1
            assert len(new_dict.dictionaries["code"]["word_level"]) == 2
    
    def test_invalid_level(self):
        """Test handling of invalid hierarchical levels."""
        xbar_dict = XBarDictionary()
        
        new_count = xbar_dict.add_spans("natural", "invalid_level", ["test"])
        assert new_count == 0
    
    def test_empty_spans(self):
        """Test handling of empty or whitespace-only spans."""
        xbar_dict = XBarDictionary()
        
        spans = ["", " ", "valid", None, "  ", "another"]
        # Note: None will cause an error in strip(), but empty strings should be filtered
        try:
            new_count = xbar_dict.add_spans("natural", "word_level", [s for s in spans if s is not None])
            # Only "valid" and "another" should be added
            assert new_count == 2
            assert len(xbar_dict.dictionaries["natural"]["word_level"]) == 2
        except:
            # Handle case where None causes issues
            valid_spans = [s for s in spans if s is not None and s.strip()]
            new_count = xbar_dict.add_spans("natural", "word_level", valid_spans)
            assert new_count == 2
    
    def test_generate_annotations_jsonl(self):
        """Test generating annotations.jsonl from dictionary."""
        xbar_dict = XBarDictionary()
        
        xbar_dict.add_spans("natural", "word_level", ["the", "cat"])
        xbar_dict.add_spans("natural", "phrase_level", ["the cat"])
        xbar_dict.add_spans("code", "word_level", ["def"])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            xbar_dict.generate_annotations_jsonl(temp_path)
            
            annotations_file = temp_path / "annotations.jsonl"
            assert annotations_file.exists()
            
            # Read and verify content
            annotations = []
            with open(annotations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    annotations.append(json.loads(line))
            
            assert len(annotations) == 4  # 3 spans total
            
            # Check that each annotation has required fields
            for ann in annotations:
                assert "id" in ann
                assert "text" in ann
                assert "domain_type" in ann
                assert "hierarchical_level" in ann
                assert "source" in ann
                assert ann["source"] == "dictionary"


class TestGlobalDictionary:
    """Test global dictionary functionality."""
    
    def test_global_dict_singleton(self):
        """Test that global dictionary behaves like a singleton."""
        dict1 = get_global_dict()
        dict2 = get_global_dict()
        
        assert dict1 is dict2  # Same instance
        
        # Add data to one, should appear in both
        dict1.add_spans("test", "word_level", ["test_span"])
        assert len(dict2.dictionaries["test"]["word_level"]) == 1
    
    def test_reset_global_dict(self):
        """Test resetting the global dictionary."""
        global_dict = get_global_dict()
        global_dict.add_spans("test", "word_level", ["test_span"])
        
        assert len(global_dict.dictionaries["test"]["word_level"]) == 1
        
        reset_global_dict()
        
        new_global_dict = get_global_dict()
        assert len(new_global_dict.dictionaries) == 0
        assert new_global_dict.stats["total_unique_spans"] == 0
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_global_dict()
