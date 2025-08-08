#!/usr/bin/env python3
"""
X-bar Annotator for X-Spanformer

Comprehensive X-bar theory span annotator that integrates with position mapping,
validation, and classifier mapping for robust hierarchical span extraction.
"""

import logging
import asyncio
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord, SpanAnnotation
from x_spanformer.xbar.position_mapper import PositionMapper, CharacterSpan, PositionSpan
from x_spanformer.xbar.xbar_map import XBarClassifierMap, DomainType
from x_spanformer.xbar.span_validator import SpanValidator

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the X-bar annotation model."""
    name: str = "llama3.2:3b"
    temperature: float = 0.2
    timeout: float = 180.0


class XBarAnnotator:
    """
    Comprehensive X-bar theory span annotator.
    
    Integrates position mapping, validation, and hierarchical X-bar classification
    for robust span extraction across natural language, code, and mixed domains.
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.validator = SpanValidator()
    
    def _detect_domain_from_record(self, pretrain_record: PretrainRecord) -> DomainType:
        """Detect domain type from the pretrain record."""
        # Use the domain from the record (set by upstream judging agent)
        domain_type = getattr(pretrain_record, 'type', 'natural')
        if isinstance(domain_type, str):
            try:
                domain = DomainType(domain_type.lower())
            except ValueError:
                domain = DomainType.NATURAL
        else:
            domain = domain_type if isinstance(domain_type, DomainType) else DomainType.NATURAL
        
        return domain
    
    def _build_system_prompt(self, domain: DomainType) -> str:
        """Build comprehensive system prompt for domain-specific annotation."""
        return XBarClassifierMap.build_system_prompt(domain)
    
    async def _extract_spans_via_dialogue(
        self, 
        text: str, 
        domain: DomainType,
        turn_focus: str
    ) -> List[CharacterSpan]:
        """
        Extract spans via dialogue with LLM using focused turn strategy.
        
        Args:
            text: Text to annotate
            domain: Domain type for classifier selection
            turn_focus: Focus for this turn (word_level, phrase_level, clause_level)
            
        Returns:
            List of character-level spans from LLM
        """
        try:
            # Import chat function locally to avoid circular imports
            from x_spanformer.agents.ollama_client import chat
            
            # Build focused prompt for this turn
            system_prompt = self._build_system_prompt(domain)
            user_prompt = f"""
Please analyze the following text for {turn_focus} X-bar spans:

Text: "{text}"

Focus on {turn_focus} structures and provide comprehensive span annotations.
Return a JSON array with all identified spans using the format specified in the system prompt.
"""
            
            # Get response from ollama
            conversation = [{"role": "user", "content": user_prompt}]
            response = await chat(
                model=self.model_config.name,
                conversation=conversation,
                system=system_prompt,
                temperature=self.model_config.temperature
            )
            
            # Parse character spans from response
            char_spans = self._parse_spans_from_response(response, text)
            
            logger.info(f"Extracted {len(char_spans)} spans for {turn_focus} from dialogue")
            return char_spans
            
        except Exception as e:
            logger.error(f"Failed to extract spans via dialogue: {e}")
            return []
    
    def _parse_spans_from_response(self, response: str, text: str) -> List[CharacterSpan]:
        """
        Parse character spans from LLM response with robust error recovery.
        
        Args:
            response: Raw LLM response
            text: Original text for validation
            
        Returns:
            List of parsed and validated character spans
        """
        spans = []
        
        # First try robust JSON parsing with multiple patterns
        json_annotations = self._parse_json_response(response)
        
        for annotation in json_annotations:
            try:
                # Extract required fields with flexible field names
                span_text = annotation.get('text', '')
                xbar_class = annotation.get('label', annotation.get('xbar_class', annotation.get('class', '')))
                confidence = float(annotation.get('confidence', 1.0))
                
                # Try to get positions if available
                start_char = annotation.get('start', annotation.get('start_char'))
                end_char = annotation.get('end', annotation.get('end_char'))
                
                # If positions available, use them directly
                if start_char is not None and end_char is not None:
                    start_char = int(start_char)
                    end_char = int(end_char)
                    
                    # Validate bounds and text match
                    is_valid, corrected_start, corrected_end = self._validate_span_boundaries(
                        text, start_char, end_char, span_text
                    )
                    
                    if is_valid:
                        char_span = CharacterSpan(
                            start_char=corrected_start,
                            end_char=corrected_end,
                            xbar_class=self._normalize_xbar_class(xbar_class),
                            confidence=confidence,
                            text=span_text
                        )
                        spans.append(char_span)
                
                # If no positions, use text-based boundary extraction
                elif span_text and span_text.strip():
                    boundaries = self._extract_text_boundaries(text, span_text)
                    for start_char, end_char in boundaries:
                        char_span = CharacterSpan(
                            start_char=start_char,
                            end_char=end_char - 1,  # Convert to inclusive end
                            xbar_class=self._normalize_xbar_class(xbar_class),
                            confidence=confidence,
                            text=span_text
                        )
                        spans.append(char_span)
                        break  # Use first occurrence
                        
            except Exception as e:
                logger.debug(f"Failed to parse annotation: {annotation}, error: {e}")
                continue
        
        # Fallback: parse from text format if no JSON found
        if not spans:
            pattern = r'"([^"]*?)"\s*\((\d+)-(\d+)\)\s*->\s*(\w+)(?:\s*\[confidence:\s*([\d.]+)\])?'
            matches = re.finditer(pattern, response)
            
            for match in matches:
                try:
                    span_text = match.group(1)
                    start_char = int(match.group(2))
                    end_char_inclusive = int(match.group(3))
                    xbar_class = match.group(4)
                    confidence = float(match.group(5)) if match.group(5) else 1.0
                    
                    is_valid, corrected_start, corrected_end = self._validate_span_boundaries(
                        text, start_char, end_char_inclusive, span_text
                    )
                    
                    if is_valid:
                        spans.append(CharacterSpan(
                            start_char=corrected_start,
                            end_char=corrected_end,
                            xbar_class=self._normalize_xbar_class(xbar_class),
                            confidence=confidence,
                            text=span_text
                        ))
                except Exception as e:
                    logger.debug(f"Failed to parse text format span: {match.group(0)}, error: {e}")
                    continue
        
        # Remove duplicates based on position and class
        unique_spans = []
        seen_spans = set()
        
        for span in spans:
            span_key = (span.start_char, span.end_char, span.xbar_class)
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                unique_spans.append(span)
        
        logger.info(f"Parsed {len(unique_spans)} unique spans from {len(spans)} total found")
        return unique_spans
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse and validate JSON response from LLM with enhanced error recovery.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            List of parsed annotation dictionaries
        """
        annotations = []
        
        # Try to extract JSON from response with comprehensive patterns
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',       # JSON code block with array
            r'```json\s*(\{.*?\})\s*```',       # JSON code block with object
            r'```\s*(\[.*?\])\s*```',           # Generic code block with array
            r'```\s*(\{.*?\})\s*```',           # Generic code block with object
            r'(\[(?:\s*\{[^}]*\},?\s*)*\])',    # JSON arrays with objects
            r'(\{[^{}]*"text"[^{}]*\})',        # Objects containing "text" field
            r'(\{[^{}]*"start"[^{}]*\})',       # Objects containing position fields
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    # Try direct parsing first
                    parsed_data = json.loads(match)
                    if isinstance(parsed_data, list):
                        annotations.extend(parsed_data)
                    elif isinstance(parsed_data, dict):
                        annotations.append(parsed_data)
                except json.JSONDecodeError:
                    # Try fixing malformed JSON
                    try:
                        fixed_json = self._fix_malformed_json(match)
                        parsed_data = json.loads(fixed_json)
                        if isinstance(parsed_data, list):
                            annotations.extend(parsed_data)
                        elif isinstance(parsed_data, dict):
                            annotations.append(parsed_data)
                    except json.JSONDecodeError:
                        # Try regex recovery as last resort
                        recovered_annotations = self._recover_malformed_json(match)
                        annotations.extend(recovered_annotations)
        
        # Deduplicate annotations based on key fields
        seen_annotations = set()
        unique_annotations = []
        
        for annotation in annotations:
            if isinstance(annotation, dict):
                # Create key for deduplication
                text = annotation.get('text', '')
                xbar_class = annotation.get('label', annotation.get('xbar_class', ''))
                key = (text.strip(), xbar_class.strip())
                
                if key not in seen_annotations and text.strip():
                    seen_annotations.add(key)
                    unique_annotations.append(annotation)
        
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
            class_pattern = r'"(?:label|xbar_class|class)":\s*"([^"]*)"'
            conf_pattern = r'"confidence":\s*([0-9.]+)'
            start_pattern = r'"(?:start|start_char)":\s*(\d+)'
            end_pattern = r'"(?:end|end_char)":\s*(\d+)'
            
            text_matches = re.findall(text_pattern, malformed_str)
            class_matches = re.findall(class_pattern, malformed_str)
            conf_matches = re.findall(conf_pattern, malformed_str)
            start_matches = re.findall(start_pattern, malformed_str)
            end_matches = re.findall(end_pattern, malformed_str)
            
            # Try to pair them up
            for i, text in enumerate(text_matches):
                annotation = {'text': text}
                
                if i < len(class_matches):
                    annotation['label'] = class_matches[i]
                if i < len(conf_matches):
                    annotation['confidence'] = float(conf_matches[i])
                if i < len(start_matches):
                    annotation['start'] = int(start_matches[i])
                if i < len(end_matches):
                    annotation['end'] = int(end_matches[i])
                
                if 'label' in annotation:  # Only add if we have both text and label
                    recovered.append(annotation)
                    
        except Exception as e:
            logger.debug(f"Recovery attempt failed: {e}")
            
        return recovered
    
    def _validate_span_boundaries(
        self, 
        text: str, 
        start: int, 
        end: int,
        span_text: str = ""
    ) -> Tuple[bool, int, int]:
        """
        Validate and correct span boundaries.
        
        Args:
            text: Original text sequence
            start: Start position (inclusive)
            end: End position (inclusive for this method)
            span_text: Expected text at this position
            
        Returns:
            Tuple of (is_valid, corrected_start, corrected_end)
        """
        original_start, original_end = start, end
        
        # Fix negative start positions
        if start < 0:
            start = 0
        
        # Fix end positions beyond sequence length
        if end >= len(text):
            end = len(text) - 1
        
        # Fix inverted positions
        if start >= end:
            if original_start < len(text):
                end = min(original_start + max(1, len(span_text)), len(text) - 1)
            else:
                return False, start, end
        
        # Validate against expected text if provided
        if span_text and span_text.strip():
            actual_text = text[start:end + 1]
            if actual_text.strip() != span_text.strip():
                # Try to find the text nearby
                boundaries = self._extract_text_boundaries(text, span_text.strip())
                if boundaries:
                    best_start, best_end = boundaries[0]
                    return True, best_start, best_end - 1  # Convert to inclusive
        
        # Final validation: ensure we have a valid span length
        if end <= start:
            return False, start, end
        
        return True, start, end
    
    def _extract_text_boundaries(self, text: str, target_text: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of target_text in the source text.
        
        Args:
            text: Source text to search in
            target_text: Text snippet to find
            
        Returns:
            List of (start_pos, end_pos) tuples for all occurrences (end_pos is exclusive)
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
    
    def _normalize_xbar_class(self, xbar_class: str) -> str:
        """
        Normalize X-bar class label to standard format.
        
        Args:
            xbar_class: Raw X-bar class from LLM
            
        Returns:
            Normalized X-bar class
        """
        if not xbar_class:
            return "unknown"
            
        # Remove extra whitespace and convert to standard case
        normalized = xbar_class.strip()
        
        # Preserve detailed classifier names first, only fall back to abbreviations if needed
        detailed_map = {
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
            "noun phrase": "noun_phrase",
            "verb phrase": "verb_phrase", 
            "adjective phrase": "adjective_phrase",
            "prepositional phrase": "prepositional_phrase",
            "determiner phrase": "determiner_phrase",
            "complementizer phrase": "complementizer_phrase",  
            "np": "noun_phrase",
            "vp": "verb_phrase",
            "ap": "adjective_phrase",
            "pp": "prepositional_phrase",
            "dp": "determiner_phrase",
            "cp": "complementizer_phrase",
            "n": "noun",
            "v": "verb",
            "a": "adjective",
            "adv": "adverb",
            "d": "determiner",
            "p": "preposition",
            "pro": "pronoun",
            "conj": "conjunction",
        }
        
        # Try partial matches for abbreviations
        for key, value in abbreviation_map.items():
            if normalized.lower() == key.lower():
                return value
        
        # Return original if no mapping found
        return normalized
    
    def _validate_and_filter_spans(
        self, 
        char_spans: List[CharacterSpan], 
        text: str
    ) -> List[CharacterSpan]:
        """
        Validate and filter character spans using span validator.
        
        Args:
            char_spans: Raw character spans from LLM
            text: Original text
            
        Returns:
            Validated and filtered character spans
        """
        valid_spans = []
        
        for char_span in char_spans:
            # Create annotation record for validation
            annotation = {
                'span_annotation': {
                    'text': char_span.text,
                    'start_pos': char_span.start_char,
                    'end_pos': char_span.end_char,
                    'length': char_span.end_char - char_span.start_char + 1,
                    'xbar_class': char_span.xbar_class
                },
                'raw': text
            }
            
            is_valid, reason = self.validator.validate_span(annotation)
            if is_valid:
                valid_spans.append(char_span)
            else:
                logger.debug(f"Filtered invalid span '{char_span.text}': {reason}")
        
        logger.info(f"Validated {len(valid_spans)}/{len(char_spans)} spans")
        return valid_spans
    
    def _convert_to_position_spans(
        self, 
        char_spans: List[CharacterSpan], 
        position_mapper: PositionMapper
    ) -> List[SpanAnnotation]:
        """
        Convert character spans to position spans and create SpanAnnotation objects.
        
        Args:
            char_spans: Validated character spans
            position_mapper: Position mapper for the text
            
        Returns:
            List of SpanAnnotation objects for annotation record
        """
        span_annotations = []
        
        for char_span in char_spans:
            # Convert to position span
            pos_span = position_mapper.char_span_to_position_span(char_span)
            
            # Create SpanAnnotation object
            span_annotation = SpanAnnotation(
                start_pos=pos_span.start_pos,
                end_pos=pos_span.end_pos,
                xbar_class=pos_span.xbar_class,
                confidence=pos_span.confidence,
                linguistic_features={
                    'extracted_text': char_span.text,
                    'character_span': {
                        'start_char': char_span.start_char,
                        'end_char': char_span.end_char
                    },
                    'position_span': {
                        'start_pos': pos_span.start_pos,
                        'end_pos': pos_span.end_pos,
                        'positions': pos_span.positions
                    }
                }
            )
            
            span_annotations.append(span_annotation)
        
        return span_annotations
    
    async def annotate_sequence(self, pretrain_record: PretrainRecord) -> Optional[AnnotationRecord]:
        """
        Annotate a sequence using comprehensive X-bar theory analysis.
        
        This method implements a three-turn strategy:
        1. Word-level annotation (nouns, verbs, etc.)
        2. Phrase-level annotation (noun phrases, verb phrases, etc.)
        3. Clause-level annotation (main clauses, subordinate clauses, etc.)
        
        Args:
            pretrain_record: PretrainRecord to annotate
            
        Returns:
            AnnotationRecord with comprehensive X-bar spans or None if failed
        """
        try:
            text = pretrain_record.raw
            domain = self._detect_domain_from_record(pretrain_record)
            
            logger.info(f"Starting X-bar annotation for domain: {domain.value}")
            logger.info(f"Text length: {len(text)} characters")
            
            # Initialize position mapper
            position_mapper = PositionMapper(text)
            
            # Three-turn annotation strategy
            all_char_spans = []
            
            # Turn 1: Word-level spans
            logger.info("Turn 1: Extracting word-level spans")
            word_spans = await self._extract_spans_via_dialogue(text, domain, "word_level")
            all_char_spans.extend(word_spans)
            
            # Turn 2: Phrase-level spans
            logger.info("Turn 2: Extracting phrase-level spans")
            phrase_spans = await self._extract_spans_via_dialogue(text, domain, "phrase_level")
            all_char_spans.extend(phrase_spans)
            
            # Turn 3: Clause-level spans
            logger.info("Turn 3: Extracting clause-level spans")
            clause_spans = await self._extract_spans_via_dialogue(text, domain, "clause_level")
            all_char_spans.extend(clause_spans)
            
            # Validate and filter spans
            logger.info(f"Validating {len(all_char_spans)} total spans")
            valid_char_spans = self._validate_and_filter_spans(all_char_spans, text)
            
            # Convert to position spans
            span_annotations = self._convert_to_position_spans(valid_char_spans, position_mapper)
            
            # Create annotation record
            annotation_record = AnnotationRecord(
                raw=text,
                sequence_number=getattr(pretrain_record.meta, 'sequence_number', 0) if pretrain_record.meta else 0,
                total_positions=position_mapper.get_text_length(),
                span_annotations=span_annotations,
                agent_metadata={
                    "strategy": "three_turn_xbar",
                    "model": self.model_config.name,
                    "domain": domain.value,
                    "total_turns": 3,
                    "word_spans": len(word_spans),
                    "phrase_spans": len(phrase_spans),
                    "clause_spans": len(clause_spans),
                    "validated_spans": len(valid_char_spans),
                    "final_spans": len(span_annotations)
                }
            )
            
            logger.info(f"Successfully annotated sequence with {len(span_annotations)} spans")
            return annotation_record
            
        except Exception as e:
            logger.error(f"Failed to annotate sequence: {e}", exc_info=True)
            return None
