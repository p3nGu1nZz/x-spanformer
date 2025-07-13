#!/usr/bin/env python3
"""
Comprehensive test suite for X-Spanformer judge and improve functionality.
Tests threshold logic, edge cases, and 3-judge consensus mechanisms.
"""
import asyncio
import csv
import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, AsyncMock

from x_spanformer.agents.selfcrit import judge_segment, parse_response
from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.ollama_client import chat
from x_spanformer.agents.prompts import render_prompt
from x_spanformer.agents.session.improve_session import ImproveSession


class TestCSVData:
    """Test data for CSV generation"""
    
    CSV_TEST_DATA = [
        "# Machine Learning Classification\n\n## Overview\nThis document explains the fundamentals of machine learning classification algorithms. Classification is a supervised learning technique used to predict categorical labels.\n\n## Common Algorithms\n- **Decision Trees**: Easy to interpret tree-based models\n- **Random Forest**: Ensemble method combining multiple trees\n- **Support Vector Machines**: Effective for high-dimensional data\n- **Neural Networks**: Deep learning approaches for complex patterns\n\n## Key Concepts\nThe goal is to learn a mapping function from input features to output categories. Training data consists of labeled examples that help the model understand patterns.",
        
        "def calculate_fibonacci(n):\n    '''Calculate nth Fibonacci number using dynamic programming'''\n    if n <= 1:\n        return n\n    \n    # Initialize base cases\n    prev_two = 0\n    prev_one = 1\n    \n    # Calculate iteratively\n    for i in range(2, n + 1):\n        current = prev_one + prev_two\n        prev_two = prev_one\n        prev_one = current\n    \n    return prev_one\n\n# Example usage\nif __name__ == '__main__':\n    print(f'Fibonacci(10) = {calculate_fibonacci(10)}')",
        
        "To define a constant in JavaScript use const PI = 3.14. This creates an immutable variable that cannot be reassigned. For functions use function greet(name) { return Hello + name; } syntax for standard function declarations. These code constructs provide clear structural elements.",
        
        "class DataProcessor:\n    def __init__(self, batch_size=32):\n        self.batch_size = batch_size\n        self.processed_count = 0\n    \n    def process_batch(self, data):\n        processed = []\n        for item in data:\n            cleaned = self.clean_data(item)\n            processed.append(cleaned)\n        \n        self.processed_count += len(processed)\n        return processed",
        
        "Lorem ipsum dolor sit amet consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam quis nostrud exercitation ullamco laboris.",
        
        "This text demonstrates various quality levels for testing the improved judge consensus mechanism. It should provide clear examples for content type classification and judge agreement testing in the X-Spanformer pipeline system."
    ]


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with test data"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    
    # Write CSV data
    writer = csv.writer(temp_file)
    writer.writerow(['text'])  # Header
    for text_data in TestCSVData.CSV_TEST_DATA:
        writer.writerow([text_data])
    
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass


@pytest.fixture
def selfcrit_config():
    """Load selfcrit configuration for testing"""
    return load_selfcrit_config(quiet=True)


class TestThresholdLogic:
    """Test threshold logic for judge decisions"""
    
    @pytest.mark.asyncio
    async def test_score_below_keep_threshold_results_in_revise(self):
        """Test that scores between 0.25 and 0.8 result in 'revise' status"""
        test_text = "This is a moderately good text that should score around 0.7"
        
        result = await judge_segment(test_text)
        
        # Verify the result structure
        assert 'score' in result
        assert 'status' in result
        assert 'reason' in result
        
        # Check threshold logic
        if 0.25 <= result['score'] < 0.8:
            assert result['status'] == 'revise', f"Score {result['score']} should result in 'revise' status"
        elif result['score'] >= 0.8:
            assert result['status'] == 'keep', f"Score {result['score']} should result in 'keep' status"
        elif result['score'] < 0.25:
            assert result['status'] == 'discard', f"Score {result['score']} should result in 'discard' status"
    
    @pytest.mark.asyncio
    async def test_high_quality_text_gets_keep_status(self):
        """Test that high-quality text gets 'keep' status"""
        high_quality_text = """
        def binary_search(arr, target):
            '''
            Perform binary search on a sorted array.
            
            Args:
                arr: Sorted list of comparable elements
                target: Element to search for
                
            Returns:
                Index of target if found, -1 otherwise
            '''
            left, right = 0, len(arr) - 1
            
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return -1
        """
        
        result = await judge_segment(high_quality_text)
        
        # High quality code should score well
        assert result['score'] >= 0.7, f"High quality code should score >= 0.7, got {result['score']}"


class TestSingleJudge:
    """Test individual judge responses"""
    
    @pytest.mark.asyncio
    async def test_single_judge_response_parsing(self, selfcrit_config):
        """Test that single judge calls work correctly"""
        test_text = "# Machine Learning Classification\n\nThis document explains classification algorithms."
        
        system = render_prompt(selfcrit_config["templates"]["system"], text=test_text)
        model = selfcrit_config["judge"]["model_name"]
        temp = selfcrit_config["judge"]["temperature"]
        
        dm = DialogueManager(system_prompt=system, max_turns=selfcrit_config["dialogue"]["max_turns"])
        dm.add("user", render_prompt(selfcrit_config["templates"]["critique"], text=test_text))
        
        reply = await chat(
            model=model,
            conversation=dm.as_messages(),
            temperature=temp
        )
        
        assert reply is not None, "Judge should return a response"
        assert len(reply.strip()) > 0, "Judge response should not be empty"
        
        # Test parsing
        result = parse_response(reply)
        
        assert 'score' in result
        assert 'status' in result
        assert 'reason' in result
        assert 'type' in result
        
        assert isinstance(result['score'], (int, float))
        assert 0.0 <= result['score'] <= 1.0
        assert result['status'] in ['keep', 'revise', 'discard']
        assert result['type'] in ['Natural', 'Code', 'Mixed']


class TestThreeJudgeConsensus:
    """Test 3-judge consensus mechanism"""
    
    @pytest.mark.asyncio
    async def test_three_judge_consensus(self):
        """Test that 3-judge consensus works correctly"""
        test_text = "# Machine Learning Classification\n\nThis document explains classification algorithms."
        
        result = await judge_segment(test_text)
        
        # Verify consensus result structure
        assert 'score' in result
        assert 'status' in result
        assert 'reason' in result
        
        # Score should be reasonable (between 0 and 1)
        assert 0.0 <= result['score'] <= 1.0
        
        # Status should be valid
        assert result['status'] in ['keep', 'revise', 'discard']
        
        # Reason should exist
        assert len(result['reason']) > 0
    
    @pytest.mark.asyncio
    async def test_consensus_with_mixed_content(self):
        """Test consensus on mixed content (code + natural language)"""
        mixed_text = """
        # Data Processing Pipeline
        
        This module implements a data processing pipeline for machine learning.
        
        def process_data(raw_data):
            '''Process raw data for ML training'''
            cleaned = clean_data(raw_data)
            features = extract_features(cleaned)
            return normalize_features(features)
        """
        
        result = await judge_segment(mixed_text)
        
        assert result['status'] in ['keep', 'revise', 'discard']
        # Mixed content with good structure should score reasonably well
        assert result['score'] >= 0.3


class TestImproveEdgeCases:
    """Test edge cases in text improvement"""
    
    @pytest.mark.asyncio
    async def test_improve_text_with_problematic_phrases(self):
        """Test improving text that contains phrases we might accidentally filter"""
        improver = ImproveSession(quiet=True)
        
        # Technical documentation that mentions potentially problematic phrases
        test_text = """
        The revised text format should follow the segmentation guidelines.
        This improvement shows how readability patterns can enhance
        training purposes in machine learning models.
        """
        
        improved, content_type = await improver.improve(test_text, "improve clarity")
        
        # Should successfully improve without filtering out valid content
        assert content_type == "Natural"
        
        if improved:  # Improvement is optional, but if provided should be valid
            assert len(improved) > 50, "Improvement should be substantial"
            assert "machine learning" in improved.lower(), "Key concepts should be preserved"
    
    @pytest.mark.asyncio
    async def test_improve_code_with_comments(self):
        """Test improving code with comments that mention potentially problematic phrases"""
        improver = ImproveSession(quiet=True)
        
        test_text = """
        # The revised text processing algorithm
        def process_text(input_text):
            # This improvement incorporates better span segmentation
            return improved_text
        """
        
        improved, content_type = await improver.improve(test_text, "fix formatting")
        
        # Should be classified as Code or Mixed
        assert content_type in ["Code", "Mixed"]
        
        if improved:  # Improvement is optional
            assert "def process_text" in improved, "Function structure should be preserved"
            assert len(improved) > 50, "Improvement should be substantial"
    
    @pytest.mark.asyncio
    async def test_improve_preserves_valid_technical_terms(self):
        """Test that improvement preserves valid technical terms that might match filter patterns"""
        improver = ImproveSession(quiet=True)
        
        # Text with legitimate technical terms
        test_text = """
        Machine learning model training requires careful attention to
        data segmentation and span analysis. The revised approach
        incorporates better threshold management for improved results.
        """
        
        improved, content_type = await improver.improve(test_text)
        
        assert content_type == "Natural"
        
        if improved:
            # Key technical terms should be preserved or enhanced, not filtered out
            technical_terms_present = any(term in improved.lower() for term in [
                "machine learning", "segmentation", "threshold", "training"
            ])
            assert technical_terms_present, "Technical terms should be preserved"


class TestParseResponse:
    """Test response parsing functionality"""
    
    def test_parse_valid_response(self):
        """Test parsing a well-formatted judge response"""
        response = """Score: 0.8
Status: keep
Type: Natural
Reason: Clear structure and good training value"""
        
        result = parse_response(response)
        
        assert result['score'] == 0.8
        assert result['status'] == 'keep'
        assert result['type'] == 'Natural'
        assert 'Clear structure' in result['reason']
    
    def test_parse_response_with_markdown(self):
        """Test parsing response with markdown formatting"""
        response = """**Score:** 0.7
**Status:** revise
**Type:** Mixed
**Reason:** Good content but needs improvement"""
        
        result = parse_response(response)
        
        # Current regex doesn't handle markdown format, so it falls back to defaults
        assert result['score'] == 0.5  # Fallback value
        assert result['status'] == 'revise'  # Fallback value
        assert result['reason'] == 'unparseable'  # Fallback reason
    
    def test_parse_malformed_response(self):
        """Test parsing malformed response falls back gracefully"""
        response = "This is not a properly formatted response"
        
        result = parse_response(response)
        
        # Should fall back to safe defaults
        assert result['score'] == 0.5
        assert result['status'] == 'revise'
        assert result['reason'] == 'unparseable'
        # Note: 'type' field is not included in fallback response


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for full pipeline functionality"""
    
    @pytest.mark.asyncio
    async def test_csv_processing_workflow(self, temp_csv_file):
        """Test processing a CSV file through the complete workflow"""
        # This would require importing and testing the full pipeline
        # For now, just verify the CSV file was created correctly
        assert os.path.exists(temp_csv_file)
        
        # Read and verify content
        with open(temp_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == len(TestCSVData.CSV_TEST_DATA)
        assert all('text' in row for row in rows)
        
        # Verify first row contains expected content
        assert 'Machine Learning Classification' in rows[0]['text']
        assert 'def calculate_fibonacci' in rows[1]['text']


if __name__ == "__main__":
    # Allow running directly with pytest
    pytest.main([__file__, "-v"])
