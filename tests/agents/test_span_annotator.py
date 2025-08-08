"""
Simplified test suite for span_annotator_session.py - Basic functionality tests.

Tests the core functionality of the unified span annotation session.
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from typing import List, Dict, Any
import json

from x_spanformer.agents.session.span_annotator_session import (
    SpanAnnotatorSession,
    AnnotationTask,
    AnnotationResult
)
from x_spanformer.schema.annotation_record import (
    SpanAnnotation,
    AnnotationRecord,
    AnnotationBatch
)
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.metadata import RecordMeta


class TestAnnotationTask(unittest.TestCase):
    """Test the AnnotationTask dataclass."""
    
    def test_annotation_task_creation(self):
        """Test annotation task creation."""
        pretrain_record = PretrainRecord(
            raw="Test text for annotation",
            type="natural",
            meta=RecordMeta(
                sequence_number=1,
                doc_language="en",
                extracted_by="test",
                confidence=0.9,
                source_file="test.txt",
                notes="test record",
                timestamp="2025-01-01",
                source="test"
            )
        )
        
        task = AnnotationTask(
            sequence_id=1,
            text="Test text for annotation",
            embedding_chunk_id=0,
            pretrain_record=pretrain_record,
            priority=1,
            retry_count=0
        )
        
        self.assertEqual(task.sequence_id, 1)
        self.assertEqual(task.text, "Test text for annotation")
        self.assertEqual(task.embedding_chunk_id, 0)
        self.assertEqual(task.priority, 1)
        self.assertEqual(task.retry_count, 0)
        self.assertEqual(task.pretrain_record, pretrain_record)


class TestAnnotationResult(unittest.TestCase):
    """Test the AnnotationResult dataclass."""
    
    def test_annotation_result_success(self):
        """Test successful annotation result."""
        # Create test annotation record with all required fields
        span_annotation = SpanAnnotation(
            start_pos=0,
            end_pos=4,
            xbar_class="N",
            confidence=0.85,
            linguistic_features={"extracted_text": "Test"}
        )
        
        annotation_record = AnnotationRecord(
            sequence_id=1,
            raw="Test text",
            embedding_chunk_id=0,
            total_positions=10,
            span_annotations=[span_annotation]
        )
        
        result = AnnotationResult(
            sequence_id=1,
            annotation_record=annotation_record,
            success=True,
            processing_time=1.5,
            turns_used=3
        )
        
        self.assertEqual(result.sequence_id, 1)
        self.assertIsNotNone(result.annotation_record)
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(result.turns_used, 3)
    
    def test_annotation_result_failure(self):
        """Test failed annotation result."""
        result = AnnotationResult(
            sequence_id=1,
            annotation_record=None,
            success=False,
            error_message="Model timeout",
            processing_time=180.0,
            turns_used=0
        )
        
        self.assertEqual(result.sequence_id, 1)
        self.assertIsNone(result.annotation_record)
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Model timeout")
        self.assertEqual(result.processing_time, 180.0)
        self.assertEqual(result.turns_used, 0)


class TestSpanAnnotatorSession(unittest.TestCase):
    """Test the SpanAnnotatorSession class."""
    
    def setUp(self):
        """Set up test session."""
        self.session = SpanAnnotatorSession(
            model_name="test-model",
            max_retries=2,
            conversation_timeout=30.0,
            temperature=0.1,
            max_spans_per_sequence=32
        )
        
        # Create test data
        self.test_record = PretrainRecord(
            raw="The quick brown fox jumps over the lazy dog.",
            type="natural",
            meta=RecordMeta(
                sequence_number=1,
                doc_language="en",
                extracted_by="test",
                confidence=0.9,
                source_file="test.txt",
                notes="test record",
                timestamp="2025-01-01",
                source="test"
            )
        )
    
    def test_session_initialization(self):
        """Test session initialization."""
        self.assertEqual(self.session.model_name, "test-model")
        self.assertEqual(self.session.max_retries, 2)
        self.assertEqual(self.session.conversation_timeout, 30.0)
        self.assertEqual(self.session.temperature, 0.1)
        self.assertEqual(self.session.max_spans_per_sequence, 32)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        stats = self.session.get_statistics()
        
        # Check actual statistics fields from the implementation
        self.assertIn("total_processed", stats)
        self.assertIn("successful", stats)
        self.assertIn("failed", stats)
        self.assertIn("total_spans", stats)
        self.assertIn("total_turns", stats)
        self.assertIn("total_time", stats)
        
        # Initial values should be zero
        self.assertEqual(stats["total_processed"], 0)
        self.assertEqual(stats["successful"], 0)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["total_spans"], 0)
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Reset statistics
        self.session.reset_statistics()
        
        # Verify statistics are reset
        stats = self.session.get_statistics()
        self.assertEqual(stats["total_processed"], 0)
        self.assertEqual(stats["successful"], 0)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["total_spans"], 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test session."""
        self.session = SpanAnnotatorSession(
            model_name="test-model",
            max_retries=1,
            conversation_timeout=1.0  # Short timeout for testing
        )
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        async def run_test():
            batch = await self.session.annotate_batch([])
            
            self.assertIsInstance(batch, AnnotationBatch)
            self.assertEqual(len(batch.records), 0)
            # Check that batch has basic structure
            self.assertIsNotNone(batch.batch_id)
            self.assertIsInstance(batch.embedding_chunk_ids, list)
            self.assertIsInstance(batch.batch_metadata, dict)
        
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
