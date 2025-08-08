#!/usr/bin/env python3
"""
X-bar Annotator for X-Spanformer

Comprehensive X-bar theory span annotator that integrates with position mapping,
validation, and classifier mapping for robust hierarchical span extraction.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
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
        Parse character spans from LLM response.
        
        Args:
            response: Raw LLM response
            text: Original text for validation
            
        Returns:
            List of parsed and validated character spans
        """
        import json
        import re
        
        spans = []
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith('['):
                span_data = json.loads(response.strip())
                for item in span_data:
                    if all(key in item for key in ['text', 'start', 'end', 'label']):
                        char_span = CharacterSpan(
                            start_char=item['start'],
                            end_char=item['end'],
                            xbar_class=item['label'],
                            confidence=item.get('confidence', 1.0),
                            text=item['text']
                        )
                        
                        # Validate span bounds and text match
                        if (0 <= char_span.start_char < len(text) and 
                            char_span.start_char < char_span.end_char <= len(text)):
                            actual_text = text[char_span.start_char:char_span.end_char + 1]
                            if (char_span.text and actual_text and 
                                actual_text.strip() == char_span.text.strip()):
                                spans.append(char_span)
            
            # Fallback: parse from text format
            else:
                pattern = r'"([^"]*?)"\s*\((\d+)-(\d+)\)\s*->\s*(\w+)(?:\s*\[confidence:\s*([\d.]+)\])?'
                matches = re.finditer(pattern, response)
                
                for match in matches:
                    span_text = match.group(1)
                    start_char = int(match.group(2))
                    end_char_inclusive = int(match.group(3))
                    xbar_class = match.group(4)
                    confidence = float(match.group(5)) if match.group(5) else 1.0
                    
                    if start_char >= 0 and end_char_inclusive < len(text):
                        actual_text = text[start_char:end_char_inclusive + 1]
                        if actual_text.strip() == span_text.strip():
                            spans.append(CharacterSpan(
                                start_char=start_char,
                                end_char=end_char_inclusive,
                                xbar_class=xbar_class,
                                confidence=confidence,
                                text=span_text
                            ))
        
        except Exception as e:
            logger.warning(f"Failed to parse spans from response: {e}")
        
        return spans
    
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
                sequence_id=getattr(pretrain_record.meta, 'sequence_number', 0) if pretrain_record.meta else 0,
                embedding_chunk_id=getattr(pretrain_record, 'embedding_chunk_id', 1),
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
