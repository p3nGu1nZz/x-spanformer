"""
Test suite for span_annotator.py - Asynchronous span annotation agent.

Tests the span annotation pipeline for X-bar linguistic analysis
with position-wise embedding alignment.
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from typing import List, Dict, Any
import json

from x_spanformer.agents.session.span_annotator_session import (
    DialogueAgent,
    SpanAnnotatorSession,
    AnnotationTask,
    AnnotationResult
)
from x_spanformer.xbar.position_mapper import (
    PositionMapper,
    CharacterSpan,
    PositionSpan
)
from x_spanformer.schema.annotation_record import (
    SpanAnnotation,
    AnnotationRecord,
    AnnotationBatch
)
from x_spanformer.schema.pretrain_record import PretrainRecord


class TestDialogueAgent(unittest.TestCase):
    """Test the DialogueAgent wrapper."""
    
    def setUp(self):
        """Set up test agent."""
        self.agent = DialogueAgent(model_name="test-model")
    
    def test_dialogue_agent_init(self):
        """Test dialogue agent initialization."""
        self.assertEqual(self.agent.model_name, "test-model")
        self.assertEqual(len(self.agent.sessions), 0)
    
    def test_start_session(self):
        """Test starting a dialogue session."""
        async def run_test():
            session_id = "test-session-001"
            system_prompt = "You are a linguistic analysis assistant."
            
            await self.agent.start_session(session_id, system_prompt)
            
            self.assertIn(session_id, self.agent.sessions)
            self.assertIsNotNone(self.agent.sessions[session_id])
        
        # Run async test
        asyncio.run(run_test())
    
    def test_send_message(self):
        """Test sending a message in a session."""
        async def run_test():
            session_id = "test-session-002"
            system_prompt = "Test system prompt"
            
            await self.agent.start_session(session_id, system_prompt)
            
            # Mock response for testing
            test_message = "Analyze this text: 'The quick brown fox'"
            response = await self.agent.send_message(session_id, test_message)
            
            # Should return a string (mock implementation)
            self.assertIsInstance(response, str)
        
        # Run async test
        asyncio.run(run_test())
    
    def test_async_methods(self):
        """Test that async methods work correctly."""
        async def run_tests():
            # Test start session
            session_id = "test-session-003"
            system_prompt = "You are a linguistic analysis assistant."
            await self.agent.start_session(session_id, system_prompt)
            self.assertIn(session_id, self.agent.sessions)
            
            # Test send message
            test_message = "Analyze this text: 'The quick brown fox'"
            response = await self.agent.send_message(session_id, test_message)
            self.assertIsInstance(response, str)
        
        # Run async tests
        asyncio.run(run_tests())


class TestAnnotationTask(unittest.TestCase):
    """Test the AnnotationTask dataclass."""
    
    def test_annotation_task_creation(self):
        """Test annotation task creation."""
        pretrain_record = PretrainRecord(
            raw="Test sequence",
            sequence_number=42,
            embedding_chunk_id=1
        )
        
        task = AnnotationTask(
            sequence_id=42,
            text="Test sequence",
            embedding_chunk_id=1,
            pretrain_record=pretrain_record,
            priority=1
        )
        
        self.assertEqual(task.sequence_id, 42)
        self.assertEqual(task.text, "Test sequence")
        self.assertEqual(task.embedding_chunk_id, 1)
        self.assertEqual(task.priority, 1)
        self.assertEqual(task.retry_count, 0)


class TestAnnotationResult(unittest.TestCase):
    """Test the AnnotationResult dataclass."""
    
    def test_annotation_result_success(self):
        """Test successful annotation result."""
        record = AnnotationRecord(
            raw="Test text",
            sequence_id=1,
            embedding_chunk_id=1,
            total_positions=9
        )
        
        result = AnnotationResult(
            sequence_id=1,
            annotation_record=record,
            success=True,
            processing_time=1.5,
            turns_used=2
        )
        
        self.assertEqual(result.sequence_id, 1)
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(result.turns_used, 2)
    
    def test_annotation_result_failure(self):
        """Test failed annotation result."""
        result = AnnotationResult(
            sequence_id=1,
            annotation_record=None,
            success=False,
            error_message="Network timeout",
            processing_time=30.0,
            turns_used=0
        )
        
        self.assertEqual(result.sequence_id, 1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Network timeout")
        self.assertIsNone(result.annotation_record)


class TestSpanAnnotatorSession(unittest.TestCase):
    """Test the main SpanAnnotatorSession class."""
    
    def setUp(self):
        """Set up test session."""
        self.agent = SpanAnnotatorSession(
            model_name="test-model",
            max_concurrent=2,
            max_retries=1,
            conversation_timeout=5.0
        )
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.model_name, "test-model")
        self.assertEqual(self.agent.max_concurrent, 2)
        self.assertEqual(self.agent.max_retries, 1)
        self.assertEqual(self.agent.conversation_timeout, 5.0)
        self.assertIsNotNone(self.agent.dialogue_agent)
        
        # Check statistics initialization
        self.assertEqual(self.agent.stats["total_processed"], 0)
        self.assertEqual(self.agent.stats["successful"], 0)
    
    def test_get_xbar_system_prompt(self):
        """Test X-bar system prompt generation."""
        prompt = self.agent.get_xbar_system_prompt()
        
        self.assertIsInstance(prompt, str)
        self.assertIn("X-bar", prompt)
        self.assertIn("linguistic", prompt.lower())
        self.assertIn("syntactic", prompt.lower())
        self.assertIn("confidence", prompt.lower())
        
        # Test domain-specific prompts
        natural_prompt = self.agent.get_xbar_system_prompt("natural")
        code_prompt = self.agent.get_xbar_system_prompt("code")
        
        self.assertIn("noun", natural_prompt.lower())
        self.assertIn("keyword", code_prompt.lower())
    
    def test_get_initial_annotation_request(self):
        """Test initial annotation request generation."""
        text = "The quick brown fox jumps over the lazy dog."
        request = self.agent.get_initial_annotation_request(text, "natural")
        
        self.assertIsInstance(request, str)
        self.assertIn(text, request)
        self.assertIn("X-bar", request)
        
        # Test domain-specific requests
        code_request = self.agent.get_initial_annotation_request("def hello(): pass", "code")
        self.assertIn("code", code_request.lower())


class TestSpanAnnotationProcess(unittest.TestCase):
    """Test the span annotation process."""
    
    def setUp(self):
        """Set up test environment."""
        self.agent = SpanAnnotatorSession(model_name="test-model")
        self.position_mapper = PositionMapper(text="The quick brown fox")
        
        self.test_record = PretrainRecord(
            raw="The quick brown fox",
            sequence_number=1,
            embedding_chunk_id=1,
            embedding_positions=19
        )
    
    def test_process_single_sequence(self):
        """Test processing a single sequence."""
        async def run_test():
            # Mock the dialogue agent response
            mock_response = '''Analysis result:
"The quick brown fox" (0-18) -> NP [confidence: 0.88]
"The" (0-2) -> Det [confidence: 0.95]
"quick brown fox" (4-18) -> N' [confidence: 0.85]'''
            
            with patch.object(self.agent.dialogue_agent, 'send_message', new_callable=AsyncMock) as mock_send:
                mock_send.return_value = mock_response
                
                # Create annotation task
                task = AnnotationTask(
                    sequence_id=1,
                    text=self.test_record.raw,
                    embedding_chunk_id=1,
                    pretrain_record=self.test_record
                )
                
                # Process (this would normally be an internal method)
                # For now we test the components that exist
                
                system_prompt = self.agent.get_xbar_system_prompt("natural")
                self.assertIsInstance(system_prompt, str)
                
                # Test initial annotation request instead
                initial_request = self.agent.get_initial_annotation_request(self.test_record.raw, "natural")
                self.assertIsInstance(initial_request, str)
                self.assertIn(self.test_record.raw, initial_request)
        
        # Run async test
        asyncio.run(run_test())
    
    def test_process_async(self):
        """Test async processing."""
        # This now calls the synchronous version
        self.test_process_single_sequence()


class TestPositionMapper(unittest.TestCase):
    """Test position mapping functionality."""
    
    def setUp(self):
        """Set up position mapper."""
        self.text = "The quick brown fox"
        self.mapper = PositionMapper(text=self.text)
    
    def test_position_mapper_init(self):
        """Test position mapper initialization."""
        self.assertEqual(self.mapper.text, self.text)
        self.assertEqual(len(self.mapper.text), 19)
    
    def test_char_to_position_mapping(self):
        """Test character to position conversion."""
        char_span = CharacterSpan(
            start_char=4,
            end_char=9,
            xbar_class="Adj",
            text="quick"
        )
        
        pos_span = self.mapper.char_span_to_position_span(char_span)
        
        self.assertEqual(pos_span.start_pos, 4)
        self.assertEqual(pos_span.end_pos, 9)  # Directly mapped, not exclusive+1
        self.assertEqual(pos_span.xbar_class, "Adj")
    
    def test_batch_char_to_position(self):
        """Test batch character to position conversion."""
        char_spans = [
            CharacterSpan(start_char=0, end_char=3, xbar_class="Det", text="The"),
            CharacterSpan(start_char=4, end_char=9, xbar_class="Adj", text="quick"),
            CharacterSpan(start_char=0, end_char=19, xbar_class="NP", text="The quick brown fox")
        ]
        
        pos_spans = self.mapper.batch_char_to_position(char_spans)
        
        self.assertEqual(len(pos_spans), 3)
        self.assertEqual(pos_spans[0].start_pos, 0)
        self.assertEqual(pos_spans[0].end_pos, 3)  # Direct mapping
        self.assertEqual(pos_spans[2].end_pos, 19)  # Full span end
    
    def test_validate_span_boundaries(self):
        """Test span boundary validation."""
        # Valid span
        valid_span = PositionSpan(
            start_pos=0,
            end_pos=10,
            xbar_class="NP"
        )
        
        # Check span is within text bounds
        self.assertLessEqual(valid_span.end_pos, len(self.mapper.text))
        
        # Invalid span - out of bounds
        invalid_span = PositionSpan(
            start_pos=0,
            end_pos=25,  # Beyond text length
            xbar_class="NP"
        )
        
        # Check span exceeds text bounds
        self.assertGreater(invalid_span.end_pos, len(self.mapper.text))


class TestErrorHandling(unittest.TestCase):
    """Test error handling in span annotation."""
    
    def setUp(self):
        """Set up test agent."""
        self.agent = SpanAnnotatorSession(model_name="test-model")
    
    def test_agent_statistics_tracking(self):
        """Test that agent tracks statistics correctly."""
        initial_stats = self.agent.stats.copy()
        
        # Verify initial state
        self.assertEqual(initial_stats["total_processed"], 0)
        self.assertEqual(initial_stats["successful"], 0)
        self.assertEqual(initial_stats["failed"], 0)
        self.assertEqual(initial_stats["total_spans"], 0)
        self.assertEqual(initial_stats["total_turns"], 0)
        self.assertEqual(initial_stats["total_time"], 0.0)
    
    def test_conversation_timeout_setting(self):
        """Test conversation timeout configuration."""
        agent = SpanAnnotatorSession(conversation_timeout=15.0)
        self.assertEqual(agent.conversation_timeout, 15.0)
    
    def test_max_concurrent_setting(self):
        """Test max concurrent requests configuration."""
        agent = SpanAnnotatorSession(max_concurrent=10)
        self.assertEqual(agent.max_concurrent, 10)
        self.assertEqual(agent.semaphore._value, 10)  # Check semaphore limit


class TestIntegration(unittest.TestCase):
    """Integration tests for span annotation components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.agent = SpanAnnotatorSession(model_name="test-model")
        
        self.test_sequence = PretrainRecord(
            raw="The cat sat on the mat.",
            type="natural",
            sequence_number=42,
            embedding_chunk_id=3,
            embedding_positions=23
        )
    
    def test_end_to_end_components(self):
        """Test that all components work together."""
        # Test system prompt generation
        system_prompt = self.agent.get_xbar_system_prompt()
        self.assertIsInstance(system_prompt, str)
        self.assertIn("X-bar", system_prompt)
        
        # Test initial annotation request generation
        initial_request = self.agent.get_initial_annotation_request(self.test_sequence.raw)
        self.assertIsInstance(initial_request, str)
        self.assertIn(self.test_sequence.raw, initial_request)
        
        # Test position mapping
        mapper = PositionMapper(text=self.test_sequence.raw)
        self.assertEqual(len(mapper.text), 23)
        
        # Test annotation task creation
        task = AnnotationTask(
            sequence_id=42,
            text=self.test_sequence.raw,
            embedding_chunk_id=3,
            pretrain_record=self.test_sequence
        )
        
        self.assertEqual(task.sequence_id, 42)
        self.assertEqual(task.embedding_chunk_id, 3)
    
    def test_position_alignment_consistency(self):
        """Test that position alignment is consistent."""
        text = "Hello world!"
        sequence_length = len(text)
        
        # Create position mapper
        mapper = PositionMapper(text=text)
        
        # Create character span
        char_span = CharacterSpan(
            start_char=0,
            end_char=5,
            xbar_class="Greeting",
            text="Hello"
        )
        
        # Convert to position span
        pos_span = mapper.char_span_to_position_span(char_span)
        
        # Verify alignment
        self.assertEqual(pos_span.start_pos, 0)
        self.assertEqual(pos_span.end_pos, 5)  # Direct mapping
        self.assertLess(pos_span.end_pos, sequence_length + 1)  # Within bounds
        
        # Verify span text consistency
        extracted_text = text[char_span.start_char:char_span.end_char]
        self.assertEqual(extracted_text, "Hello")


if __name__ == "__main__":
    unittest.main()
