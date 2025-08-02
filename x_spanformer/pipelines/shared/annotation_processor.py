"""
Shared annotation processing utilities for X-Spanformer pipelines.

Provides utilities for processing LLM annotation responses, validating
span boundaries, and converting between different annotation formats.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from x_spanformer.xbar.position_mapper import (
    CharacterSpan,
    PositionSpan,
    parse_character_spans_from_agent_response
)

logger = logging.getLogger(__name__)


class AnnotationProcessor:
    """Shared utilities for annotation pipelines."""
    
    def parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse and validate JSON response from LLM.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            List of parsed annotation dictionaries
        """
        annotations = []
        
        # Try to extract JSON from response
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',  # JSON code block
            r'```\s*(\[.*?\])\s*```',      # Generic code block
            r'(\[.*?\])',                   # Direct JSON array
            r'(\{.*?\})',                   # Single JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, list):
                        annotations.extend(data)
                    elif isinstance(data, dict):
                        annotations.append(data)
                except json.JSONDecodeError:
                    continue
        
        return annotations
    
    def validate_span_boundaries(
        self, 
        sequence: str, 
        start: int, 
        end: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate span boundaries are within sequence bounds.
        
        Args:
            sequence: Original text sequence
            start: Start position (inclusive)
            end: End position (exclusive)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if start < 0:
            issues.append(f"Start position {start} is negative")
        
        if end > len(sequence):
            issues.append(f"End position {end} exceeds sequence length {len(sequence)}")
        
        if start >= end:
            issues.append(f"Start position {start} >= end position {end}")
        
        if end - start > len(sequence):
            issues.append(f"Span length {end - start} exceeds sequence length {len(sequence)}")
        
        return len(issues) == 0, issues
    
    def normalize_xbar_class(self, xbar_class: str) -> str:
        """
        Normalize X-bar class label to standard format.
        
        Args:
            xbar_class: Raw X-bar class from LLM
            
        Returns:
            Normalized X-bar class
        """
        # Remove extra whitespace and convert to standard case
        normalized = xbar_class.strip()
        
        # Common normalizations
        normalization_map = {
            "noun phrase": "NP",
            "verb phrase": "VP", 
            "adjective phrase": "AP",
            "prepositional phrase": "PP",
            "determiner phrase": "DP",
            "complementizer phrase": "CP",
            "noun": "N",
            "verb": "V",
            "adjective": "A",
            "adverb": "Adv",
            "determiner": "D",
            "preposition": "P",
            "pronoun": "Pro",
            "conjunction": "Conj",
        }
        
        # Try exact match first
        if normalized.lower() in normalization_map:
            return normalization_map[normalized.lower()]
        
        # Try partial matches
        for key, value in normalization_map.items():
            if key in normalized.lower():
                return value
        
        return normalized
    
    def consolidate_working_files(self, working_dir: Path, output_file: Path):
        """
        Consolidate individual working files into training JSONL.
        
        Args:
            working_dir: Directory containing individual working files
            output_file: Output JSONL file path
        """
        logger.info(f"Consolidating working files from {working_dir} to {output_file}")
        
        successful_annotations = []
        failed_count = 0
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process all working files
        for working_file in working_dir.glob("corpus-seq-*.json"):
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if annotation was successful
                annotation_status = data.get("annotation_session", {}).get("annotation_status")
                
                if annotation_status == "completed":
                    annotation_result = data.get("annotation_result")
                    if annotation_result:
                        successful_annotations.append(annotation_result)
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process working file {working_file}: {e}")
                failed_count += 1
        
        # Write consolidated annotations
        with open(output_file, 'w', encoding='utf-8') as f:
            for annotation in successful_annotations:
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        logger.info(f"Consolidated {len(successful_annotations)} successful annotations")
        logger.info(f"Skipped {failed_count} failed annotations")
    
    def update_global_metadata(self, metadata_file: Path, updates: Dict[str, Any]):
        """
        Thread-safe global metadata updates.
        
        Args:
            metadata_file: Path to global metadata JSON file
            updates: Dictionary of updates to apply
        """
        # Load existing metadata
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {}
        
        # Apply updates
        metadata.update(updates)
        metadata["last_updated"] = updates.get("last_updated", "unknown")
        
        # Write back atomically
        temp_file = metadata_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        temp_file.replace(metadata_file)
    
    def extract_spans_from_comprehensive_response(
        self,
        response: str,
        text: str,
        expected_classifiers: List[str]
    ) -> List[CharacterSpan]:
        """
        Extract spans from comprehensive LLM response covering multiple classifiers.
        
        Args:
            response: LLM response containing comprehensive annotations
            text: Original text sequence
            expected_classifiers: List of expected classifier types
            
        Returns:
            List of extracted character spans
        """
        spans = []
        
        # First try standard character span parsing
        standard_spans = parse_character_spans_from_agent_response(response, text)
        spans.extend(standard_spans)
        
        # Try JSON parsing for structured responses
        json_annotations = self.parse_json_response(response)
        
        for annotation in json_annotations:
            try:
                # Extract span information
                span_text = annotation.get("text", "")
                start_char = annotation.get("start", annotation.get("char_start", annotation.get("start_char")))
                end_char = annotation.get("end", annotation.get("char_end", annotation.get("end_char")))
                xbar_class = annotation.get("label", annotation.get("xbar_class", annotation.get("type")))
                confidence = annotation.get("confidence", 1.0)
                
                if start_char is not None and end_char is not None and xbar_class:
                    # Validate span
                    is_valid, issues = self.validate_span_boundaries(text, start_char, end_char)
                    
                    if is_valid:
                        spans.append(CharacterSpan(
                            start_char=start_char,
                            end_char=end_char,
                            xbar_class=self.normalize_xbar_class(xbar_class),
                            confidence=float(confidence),
                            text=span_text or text[start_char:end_char]
                        ))
                    else:
                        logger.warning(f"Invalid span boundaries: {issues}")
                        
            except Exception as e:
                logger.warning(f"Failed to parse annotation: {e}")
                continue
        
        return spans
    
    def align_with_position_embeddings(
        self, 
        spans: List[CharacterSpan], 
        text: str
    ) -> List[PositionSpan]:
        """
        Align character spans with position-wise embeddings using PositionMapper.
        
        Args:
            spans: List of character-level spans
            text: Original text sequence
            
        Returns:
            List of position-aligned spans
        """
        from x_spanformer.xbar.position_mapper import PositionMapper
        
        mapper = PositionMapper(text=text)
        return mapper.batch_char_to_position(spans)
