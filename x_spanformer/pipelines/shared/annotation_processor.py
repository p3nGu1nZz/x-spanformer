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
    
    def extract_text_boundaries(self, text: str, target_text: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of target_text in the source text and return their boundaries.
        
        Args:
            text: Source text to search in
            target_text: Text snippet to find
            
        Returns:
            List of (start_pos, end_pos) tuples for all occurrences
        """
        boundaries = []
        if not target_text or not target_text.strip():
            return boundaries
            
        # Clean target text
        target_clean = target_text.strip()
        
        # Find all occurrences using regex with proper escaping
        escaped_target = re.escape(target_clean)
        
        # Use finditer to get all matches with positions
        for match in re.finditer(escaped_target, text):
            start_pos = match.start()
            end_pos = match.end()
            boundaries.append((start_pos, end_pos))
            
        # If no exact matches found, try fuzzy matching for partial text
        if not boundaries and len(target_clean) > 2:
            # Try finding the target as a substring (case insensitive)
            text_lower = text.lower()
            target_lower = target_clean.lower()
            start = 0
            while True:
                pos = text_lower.find(target_lower, start)
                if pos == -1:
                    break
                boundaries.append((pos, pos + len(target_clean)))
                start = pos + 1
                
        return boundaries
    
    def parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse and validate JSON response from LLM with enhanced error recovery.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            List of parsed annotation dictionaries
        """
        annotations = []
        
        # Try to extract JSON from response with more comprehensive patterns
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',       # JSON code block with array
            r'```json\s*(\{.*?\})\s*```',       # JSON code block with object
            r'```\s*(\[.*?\])\s*```',           # Generic code block with array
            r'```\s*(\{.*?\})\s*```',           # Generic code block with object
            r'(\[(?:\s*\{[^}]*\},?\s*)*\])',    # JSON arrays with objects
            r'(\{[^{}]*"text"[^{}]*\})',        # Objects containing "text" field
            r'(\{[^{}]*"start_char"[^{}]*\})',  # Objects containing position fields
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    # Clean up match before parsing
                    cleaned_match = match.strip()
                    if not cleaned_match:
                        continue
                    
                    # Try to fix common JSON issues
                    cleaned_match = self._fix_malformed_json(cleaned_match)
                        
                    data = json.loads(cleaned_match)
                    if isinstance(data, list):
                        annotations.extend(data)
                    elif isinstance(data, dict):
                        annotations.append(data)
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON: {e} for match: {match[:100]}...")
                    # Try alternative parsing strategies
                    recovered = self._recover_malformed_json(match)
                    annotations.extend(recovered)
                    continue
                except Exception as e:
                    logger.debug(f"Unexpected error parsing JSON: {e}")
                    continue
        
        # Deduplicate annotations based on key fields
        seen_annotations = set()
        unique_annotations = []
        
        for annotation in annotations:
            if isinstance(annotation, dict):
                # Create key for deduplication
                key_fields = (
                    annotation.get("text", ""),
                    annotation.get("start_char", annotation.get("start", -1)),
                    annotation.get("end_char", annotation.get("end", -1)),
                    annotation.get("xbar_class", annotation.get("type", ""))
                )
                
                if key_fields not in seen_annotations:
                    unique_annotations.append(annotation)
                    seen_annotations.add(key_fields)
        
        logger.debug(f"Parsed {len(unique_annotations)} unique annotations from {len(annotations)} total found")
        return unique_annotations
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)
        
        # Fix missing commas between objects in arrays
        json_str = re.sub(r'}\s*{', '}, {', json_str)
        
        return json_str
    
    def _recover_malformed_json(self, malformed_str: str) -> List[Dict[str, Any]]:
        """Attempt to recover data from malformed JSON using pattern matching."""
        recovered = []
        
        try:
            # Look for text/class pairs using regex
            text_pattern = r'"text":\s*"([^"]*)"'
            class_pattern = r'"xbar_class":\s*"([^"]*)"'
            conf_pattern = r'"confidence":\s*([0-9.]+)'
            
            text_matches = re.findall(text_pattern, malformed_str)
            class_matches = re.findall(class_pattern, malformed_str)
            conf_matches = re.findall(conf_pattern, malformed_str)
            
            # Try to pair them up
            for i, text in enumerate(text_matches):
                if i < len(class_matches):
                    annotation = {
                        "text": text,
                        "xbar_class": class_matches[i],
                        "confidence": float(conf_matches[i]) if i < len(conf_matches) else 1.0
                    }
                    recovered.append(annotation)
                    
        except Exception as e:
            logger.debug(f"Recovery attempt failed: {e}")
            
        return recovered
    
    def validate_span_boundaries(
        self, 
        sequence: str, 
        start: int, 
        end: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate span boundaries are within sequence bounds with automatic correction.
        
        Args:
            sequence: Original text sequence
            start: Start position (inclusive)
            end: End position (exclusive)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        original_start, original_end = start, end
        
        # Fix negative start positions
        if start < 0:
            issues.append(f"Start position {start} was negative, correcting to 0")
            start = 0
        
        # Fix end positions beyond sequence length
        if end > len(sequence):
            issues.append(f"End position {end} exceeded sequence length {len(sequence)}, correcting to {len(sequence)}")
            end = len(sequence)
        
        # Fix inverted positions
        if start >= end:
            if original_start < len(sequence):
                # Try to fix by extending end position slightly
                end = min(start + 1, len(sequence))
                issues.append(f"Start position {start} >= end position {original_end}, correcting end to {end}")
            else:
                issues.append(f"Start position {start} >= end position {end} and cannot be corrected")
                return False, issues
        
        # Check for reasonable span length (not entire text unless it's a sentence-level span)
        span_length = end - start
        if span_length > len(sequence) * 0.8 and span_length < len(sequence):
            issues.append(f"Large span length {span_length} (likely sentence/clause level)")
        
        # Return corrected positions are valid
        return True, issues
    
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
        
        # Preserve detailed classifier names first, only fall back to abbreviations if needed
        detailed_map = {
            # Keep full names for better interpretability
            "determiner": "determiner",
            "noun": "noun", 
            "verb": "verb",
            "adjective": "adjective",
            "adverb": "adverb",
            "preposition": "preposition",
            "pronoun": "pronoun",
            "conjunction": "conjunction",
            "punctuation": "punctuation",
            "noun_phrase": "noun_phrase",
            "verb_phrase": "verb_phrase",
            "adjective_phrase": "adjective_phrase",
            "adverb_phrase": "adverb_phrase", 
            "prepositional_phrase": "prepositional_phrase",
            "main_clause": "main_clause",
            "subordinate_clause": "subordinate_clause",
            "relative_clause": "relative_clause",
            "simple_sentence": "simple_sentence",
            "compound_sentence": "compound_sentence",
            "complex_sentence": "complex_sentence",
            "head": "head",
            "specifier": "specifier",
            "modifier": "modifier", 
            "complement": "complement",
            "adjunct": "adjunct",
        }
        
        # Try exact match first (case insensitive)
        for key, value in detailed_map.items():
            if normalized.lower() == key.lower():
                return value
        
        # Legacy abbreviation fallbacks (only if no detailed match)
        abbreviation_map = {
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
        
        # Try partial matches for abbreviations
        for key, value in abbreviation_map.items():
            if key in normalized.lower():
                return value
        
        # Return original if no mapping found
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
        Uses text-based boundary extraction for accurate positioning.
        
        Args:
            response: LLM response containing comprehensive annotations
            text: Original text sequence
            expected_classifiers: List of expected classifier types
            
        Returns:
            List of extracted character spans
        """
        spans = []
        seen_spans = set()  # Track duplicates
        
        # Parse JSON annotations (now text-based, no position data)
        json_annotations = self.parse_json_response(response)
        
        for annotation in json_annotations:
            try:
                # Extract span information (no position fields needed)
                span_text = annotation.get("text", "")
                xbar_class = annotation.get("xbar_class", 
                    annotation.get("label", 
                        annotation.get("type", 
                            annotation.get("class", 
                                annotation.get("category", "")))))
                confidence = annotation.get("confidence", 1.0)
                
                if span_text and xbar_class:
                    # Find all occurrences of this text in the source
                    boundaries = self.extract_text_boundaries(text, span_text)
                    
                    for start_char, end_char in boundaries:
                        # Validate and correct boundaries if needed
                        is_valid, issues = self.validate_span_boundaries(text, start_char, end_char)
                        
                        if is_valid:
                            # Use corrected boundaries
                            corrected_start = max(0, start_char)
                            corrected_end = min(len(text), end_char)
                            
                            # Extract actual text from corrected boundaries
                            actual_text = text[corrected_start:corrected_end]
                            
                            # Preserve original classifier name
                            normalized_class = xbar_class.strip()
                            
                            # Create span key for deduplication
                            span_key = (corrected_start, corrected_end, normalized_class)
                            
                            if span_key not in seen_spans:
                                spans.append(CharacterSpan(
                                    start_char=corrected_start,
                                    end_char=corrected_end,
                                    xbar_class=normalized_class,
                                    confidence=float(confidence),
                                    text=actual_text  # Use actual extracted text
                                ))
                                seen_spans.add(span_key)
                                
                            if issues:  # Log corrections, not errors
                                logger.info(f"Corrected span boundaries: {issues}")
                        else:
                            logger.warning(f"Invalid span boundaries for '{span_text}': {issues}")
                            
            except Exception as e:
                logger.warning(f"Failed to parse annotation: {e}")
                continue
        
        logger.info(f"Extracted {len(spans)} unique spans from text-based response")
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
