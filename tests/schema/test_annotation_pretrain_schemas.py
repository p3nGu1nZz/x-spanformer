"""
Test suite for annotation_record and pretrain_record schemas.

Tests the enhanced schemas for position-wise embedding alignment
and span annotation pipeline integration.
"""

import unittest
import pytest
from typing import List, Dict, Any
from pydantic import ValidationError

from x_spanformer.schema.annotation_record import (
    SpanAnnotation, 
    AnnotationRecord
)
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.identifier import RecordID
from x_spanformer.schema.metadata import RecordMeta


class TestSpanAnnotation(unittest.TestCase):
    """Test SpanAnnotation schema for position-wise boundary annotations."""
    
    def test_span_annotation_basic(self):
        """Test basic span annotation creation."""
        annotation = SpanAnnotation(
            start_pos=4,
            end_pos=19,
            xbar_label="NP"
        )
        
        self.assertEqual(annotation.start_pos, 4)
        self.assertEqual(annotation.end_pos, 19)
        self.assertEqual(annotation.xbar_label, "NP")
        self.assertIsNone(annotation.linguistic_features)
    
    def test_span_annotation_with_features(self):
        """Test span annotation with linguistic features."""
        features = {
            "head": "fox",
            "specifier": "the quick brown",
            "syntactic_role": "subject"
        }
        
        annotation = SpanAnnotation(
            start_pos=0,
            end_pos=18,
            xbar_label="NP",
            linguistic_features=features
        )
        
        self.assertEqual(annotation.linguistic_features, features)
    
    def test_span_annotation_validation(self):
        """Test span annotation validation."""
        # Valid annotation
        SpanAnnotation(start_pos=0, end_pos=5, xbar_label="VP")
        
        # Test with various labels
        annotation = SpanAnnotation(
            start_pos=0, 
            end_pos=5, 
            xbar_label="NP"
        )
        self.assertEqual(annotation.xbar_label, "NP")
    
    def test_span_annotation_serialization(self):
        """Test JSON serialization/deserialization."""
        annotation = SpanAnnotation(
            start_pos=10,
            end_pos=25,
            xbar_label="VP",
            linguistic_features={"tense": "past", "voice": "active"}
        )
        
        # Serialize to dict
        data = annotation.model_dump()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["start_pos"], 10)
        
        # Deserialize from dict
        restored = SpanAnnotation(**data)
        self.assertEqual(restored.start_pos, annotation.start_pos)
        self.assertEqual(restored.linguistic_features, annotation.linguistic_features)


class TestAnnotationRecord(unittest.TestCase):
    """Test AnnotationRecord schema for training data."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_annotations = [
            SpanAnnotation(start_pos=0, end_pos=3, xbar_label="Det"),
            SpanAnnotation(start_pos=4, end_pos=9, xbar_label="Adj"),
            SpanAnnotation(start_pos=10, end_pos=15, xbar_label="N"),
            SpanAnnotation(start_pos=0, end_pos=15, xbar_label="NP")
        ]
        
        self.sample_conversation = [
            {"role": "user", "content": "Analyze this text for noun phrases."},
            {"role": "assistant", "content": "I found several spans..."}
        ]
    
    def test_annotation_record_basic(self):
        """Test basic annotation record creation."""
        record = AnnotationRecord(
            raw="The quick brown fox",
            sequence_number=42,
            span_annotations=self.sample_annotations,
            total_positions=19
        )
        
        self.assertEqual(record.raw, "The quick brown fox")
        self.assertEqual(record.sequence_number, 42)
        self.assertEqual(len(record.span_annotations), 4)
        self.assertEqual(record.total_positions, 19)
    
    def test_annotation_record_with_conversation(self):
        """Test annotation record with conversation context."""
        record = AnnotationRecord(
            raw="Complex sentence requiring multi-turn analysis.",
            sequence_number=100,
            span_annotations=self.sample_annotations,
            total_positions=47,
            conversation_turns=self.sample_conversation,
            agent_metadata={"model": "phi4-mini", "temperature": 0.7}
        )
        
        self.assertIsNotNone(record.conversation_turns)
        if record.conversation_turns is not None:
            self.assertEqual(len(record.conversation_turns), 2)
        self.assertIsNotNone(record.agent_metadata)
        if record.agent_metadata is not None:
            self.assertEqual(record.agent_metadata["model"], "phi4-mini")
    
    def test_annotation_record_empty_annotations(self):
        """Test annotation record with no spans."""
        record = AnnotationRecord(
            raw="Short.",
            sequence_number=1,
            total_positions=6
        )
        
        self.assertEqual(len(record.span_annotations), 0)
        self.assertEqual(record.total_positions, 6)
    
    def test_annotation_record_validation(self):
        """Test annotation record validation."""
        # Should create successfully
        record = AnnotationRecord(
            raw="Valid text",
            sequence_number=1,
            total_positions=10
        )
        
        # Test that overlapping spans are allowed (hierarchical structure)
        overlapping_spans = [
            SpanAnnotation(start_pos=0, end_pos=5, xbar_label="NP"),
            SpanAnnotation(start_pos=2, end_pos=8, xbar_label="VP"),  # Overlaps
            SpanAnnotation(start_pos=0, end_pos=8, xbar_label="S")    # Contains both
        ]
        
        record_with_overlaps = AnnotationRecord(
            raw="Test text",
            sequence_number=2,
            span_annotations=overlapping_spans,
            total_positions=9
        )
        
        self.assertEqual(len(record_with_overlaps.span_annotations), 3)


class TestPretrainRecord(unittest.TestCase):
    """Test enhanced PretrainRecord schema."""
    
    def test_pretrain_record_basic(self):
        """Test basic pretrain record creation."""
        record = PretrainRecord(raw="Test sequence for pretraining.")
        
        self.assertEqual(record.raw, "Test sequence for pretraining.")
        self.assertIsNone(record.type)
        self.assertIsInstance(record.id, RecordID)
        self.assertIsInstance(record.meta, RecordMeta)
        self.assertIsNone(record.sequence_number)
        self.assertIsNone(record.embedding_positions)
    
    def test_pretrain_record_enhanced(self):
        """Test pretrain record with embedding alignment fields."""
        record = PretrainRecord(
            raw="The quick brown fox jumps over the lazy dog.",
            type="natural",
            sequence_number=42,
            embedding_positions=44
        )
        
        self.assertEqual(record.type, "natural")
        self.assertEqual(record.sequence_number, 42)
        self.assertEqual(record.embedding_positions, 44)
        self.assertEqual(len(record.raw), 44)  # Character count matches
    
    def test_pretrain_record_types(self):
        """Test different content types."""
        # Natural language
        natural = PretrainRecord(raw="Hello world!", type="natural")
        self.assertEqual(natural.type, "natural")
        
        # Code
        code = PretrainRecord(raw="def hello(): pass", type="code")
        self.assertEqual(code.type, "code")
        
        # Mixed content
        mixed = PretrainRecord(raw="# Comment\nprint('hello')", type="mixed")
        self.assertEqual(mixed.type, "mixed")
    
    def test_pretrain_record_metadata_integration(self):
        """Test integration with metadata system."""
        meta = RecordMeta(
            tags=["test", "natural"],
            doc_language="en",
            extracted_by="test_pipeline",
            source_file="test_corpus.jsonl",
            notes="Test metadata",
            sequence_number=1,
            timestamp="2025-01-01T00:00:00",
            source="test",
            confidence=None
        )
        
        record = PretrainRecord(
            raw="Test with custom metadata.",
            type="natural",
            meta=meta,
            sequence_number=1,
            embedding_positions=27
        )
        
        self.assertEqual(record.meta.tags, ["test", "natural"])
    
    def test_pretrain_record_serialization(self):
        """Test pretrain record serialization."""
        record = PretrainRecord(
            raw="Serialization test sequence.",
            type="natural",
            sequence_number=999,
            embedding_positions=28
        )
        
        # Test JSON serialization
        data = record.model_dump()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["raw"], "Serialization test sequence.")
        self.assertEqual(data["sequence_number"], 999)
        
        # Test deserialization
        restored = PretrainRecord(**data)
        self.assertEqual(restored.raw, record.raw)
        self.assertEqual(restored.sequence_number, record.sequence_number)


class TestSchemaIntegration(unittest.TestCase):
    """Test integration between annotation and pretrain schemas."""
    
    def test_sequence_number_alignment(self):
        """Test that sequence numbers align between pretrain and annotation records."""
        # Pretrain record
        pretrain = PretrainRecord(
            raw="Integration test sequence.",
            sequence_number=123,
            embedding_positions=26
        )
        
        # Corresponding annotation record
        annotation = AnnotationRecord(
            raw=pretrain.raw,
            sequence_number=pretrain.sequence_number or 123,
            span_annotations=[
                SpanAnnotation(start_pos=0, end_pos=11, xbar_label="NP"),
                SpanAnnotation(start_pos=12, end_pos=16, xbar_label="N"),
                SpanAnnotation(start_pos=17, end_pos=25, xbar_label="N")
            ],
            total_positions=26
        )
        
        # Verify alignment
        self.assertEqual(pretrain.raw, annotation.raw)
        self.assertEqual(pretrain.sequence_number, annotation.sequence_number)
        self.assertEqual(pretrain.embedding_positions, annotation.total_positions)
    
    def test_position_boundary_validation(self):
        """Test that position boundaries make sense."""
        sequence_length = 20
        
        # Valid spans
        valid_annotations = [
            SpanAnnotation(start_pos=0, end_pos=5, xbar_label="Det"),
            SpanAnnotation(start_pos=5, end_pos=15, xbar_label="N"),
            SpanAnnotation(start_pos=15, end_pos=20, xbar_label="V")
        ]
        
        record = AnnotationRecord(
            raw="A" * sequence_length,  # 20 characters
            sequence_number=1,
            span_annotations=valid_annotations,
            total_positions=sequence_length
        )
        
        # All spans should be within bounds
        for annotation in record.span_annotations:
            self.assertGreaterEqual(annotation.start_pos, 0)
            self.assertLessEqual(annotation.end_pos, sequence_length)
            self.assertLess(annotation.start_pos, annotation.end_pos)


if __name__ == "__main__":
    unittest.main()
