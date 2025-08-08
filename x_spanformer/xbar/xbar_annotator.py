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
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass

from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord, SpanAnnotation
from x_spanformer.schema.span import SpanLabel
from x_spanformer.xbar.xbar_map import XBarLabelMap, DomainType

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
    
    def _detect_domain_from_record(self, pretrain_record: PretrainRecord) -> DomainType:
        """Detect domain type from the pretrain record."""
        domain_type = getattr(pretrain_record, 'type', 'natural')
        if isinstance(domain_type, str):
            try:
                return DomainType(domain_type.lower())
            except ValueError:
                return DomainType.NATURAL
        return domain_type if isinstance(domain_type, DomainType) else DomainType.NATURAL
    
    def _build_system_prompt(self, domain: DomainType) -> str:
        """Build comprehensive system prompt for domain-specific annotation."""
        labels = XBarLabelMap.get_labels_for_domain(domain)
        label_descriptions = [f"- {label}: {description}" for label, description in labels.items()]
        
        return f"""You are a linguistic annotator specializing in {domain.value} text analysis.

Available labels for {domain.value} domain:
{chr(10).join(label_descriptions)}

Extract spans and classify them using only the labels above. Focus on accuracy and consistency."""
    
    def _build_domain_specific_prompts(
        self, 
        domain: DomainType, 
        turn_focus: str, 
        text_snippet: str
    ) -> Tuple[str, str]:
        """Build domain-specific system and user prompts for a given turn."""
        all_labels = XBarLabelMap.get_labels_for_domain(domain)
        
        # Define label sets and descriptions for each turn focus and domain
        label_configs = {
            "word_level": {
                DomainType.NATURAL: {
                    "labels": ["noun", "verb", "adjective", "adverb", "preposition", "determiner", "pronoun", "conjunction", "punctuation"],
                    "description": "individual WORDS and their grammatical classes",
                    "examples": '"word" -> noun, "runs" -> verb, "quickly" -> adverb'
                },
                DomainType.CODE: {
                    "labels": ["keyword", "identifier", "operator", "literal", "delimiter", "type_name", "comment"],
                    "description": "individual CODE TOKENS and their syntactic types",
                    "examples": '"if" -> keyword, "variable" -> identifier, "+" -> operator'
                },
                DomainType.MIXED: {
                    "labels": ["noun", "verb", "adjective", "adverb", "preposition", "determiner", "pronoun", "conjunction", "keyword", "identifier", "operator", "literal", "inline_code"],
                    "description": "individual WORDS/TOKENS from both natural language and code",
                    "examples": '"function" -> noun (or keyword in code context), "variable" -> identifier'
                }
            },
            "phrase_level": {
                DomainType.NATURAL: {
                    "labels": ["noun_phrase", "verb_phrase", "adjective_phrase", "adverb_phrase", "prepositional_phrase"],
                    "description": "PHRASES (groups of related words)",
                    "examples": '"the red car" -> noun_phrase, "is running quickly" -> verb_phrase'
                },
                DomainType.CODE: {
                    "labels": ["expression", "function_call", "assignment", "parameter_list", "argument_list"],
                    "description": "CODE EXPRESSIONS and structured constructs",
                    "examples": '"x + y" -> expression, "func(a, b)" -> function_call'
                },
                DomainType.MIXED: {
                    "labels": ["noun_phrase", "verb_phrase", "expression", "function_call", "code_block", "documentation_comment"],
                    "description": "PHRASES and CODE EXPRESSIONS from mixed content",
                    "examples": '"the function call" -> noun_phrase, "func(x)" -> function_call'
                }
            },
            "clause_level": {
                DomainType.NATURAL: {
                    "labels": ["main_clause", "subordinate_clause", "relative_clause"],
                    "description": "CLAUSES and major syntactic structures",
                    "examples": '"She runs fast" -> main_clause, "because it was late" -> subordinate_clause'
                },
                DomainType.CODE: {
                    "labels": ["if_statement", "loop_statement", "function_definition", "class_definition", "import_statement", "return_statement"],
                    "description": "CODE STATEMENTS and control structures",
                    "examples": '"if x > 0:" -> if_statement, "def func():" -> function_definition'
                },
                DomainType.MIXED: {
                    "labels": ["main_clause", "subordinate_clause", "if_statement", "loop_statement", "function_definition"],
                    "description": "CLAUSES and CODE STATEMENTS from mixed content",
                    "examples": '"The function returns" -> main_clause, "if condition:" -> if_statement'
                }
            }
        }
        
        config = label_configs[turn_focus][domain]
        relevant_labels = {k: v for k, v in all_labels.items() if k in config["labels"]}
        label_names = list(relevant_labels.keys())
        label_descriptions = [f"{k}: {v}" for k, v in relevant_labels.items()]
        
        system_prompt = f"""You are a linguistic annotator specializing in {domain.value} domain {turn_focus.replace('_', '-')} analysis.

Domain: {domain.value.upper()}
Focus: {config["description"]}

Available labels:
{chr(10).join(f"- {desc}" for desc in label_descriptions)}

Extract accurate spans using ONLY these labels. Be precise and consistent."""

        user_prompt = f"""Analyze this {domain.value} text and identify {config["description"]}:
"{text_snippet}"

Return ONLY a JSON array with this exact format. Do not include any explanations, notes, or additional text:
[{{"text":"extracted_text","xbar_label":"label_name"}}]

Examples: {config["examples"]}

Use these labels: {", ".join(label_names)}"""

        return system_prompt, user_prompt
    
    async def _extract_spans_via_dialogue(
        self, 
        text: str, 
        domain: DomainType,
        turn_focus: str,
        pretrain_record: PretrainRecord
    ) -> List[SpanLabel]:
        """Extract spans via dialogue with LLM using focused turn strategy."""
        try:
            from x_spanformer.agents.ollama_client import chat
            from x_spanformer.agents.dialogue import DialogueManager
            
            domain = self._detect_domain_from_record(pretrain_record)
            text_snippet = text[:200] if len(text) > 200 else text
            
            system_prompt, user_prompt = self._build_domain_specific_prompts(domain, turn_focus, text_snippet)
            
            dm = DialogueManager(system_prompt=system_prompt, max_turns=1)
            dm.add("user", user_prompt)
            
            response = await chat(
                model=self.model_config.name,
                conversation=dm.as_messages(),
                temperature=0.1,
                timeout=90.0
            )
            
            dm.add("assistant", response)
            spans = self._parse_spans_from_response(response, text)
            
            logger.info(f"Extracted {len(spans)} spans for {turn_focus} from dialogue")
            return spans
            
        except ValueError as e:
            logger.error(f"Failed to extract spans via dialogue: {e}")
            raise  # Let ValueError bubble up to sequence level
        except Exception as e:
            logger.error(f"Failed to extract spans via dialogue: {e}")
            return []
    
    def _parse_spans_from_response(self, response: str, text: str) -> List[SpanLabel]:
        """Parse spans from LLM response using regex-based text matching."""
        spans = []
        json_annotations = self._parse_json_response(response)
        
        if json_annotations:
            logger.debug(f"Successfully parsed {len(json_annotations)} JSON annotations, now attempting text matching")
        else:
            logger.debug("No JSON annotations parsed from response")
        
        for annotation in json_annotations:
            try:
                span_text = (annotation.get('text', '') or '').strip()
                xbar_label_raw = (annotation.get('xbar_label') or annotation.get('label') or 
                                annotation.get('xbar_class') or annotation.get('class') or '')
                xbar_label = xbar_label_raw.strip() if xbar_label_raw else ''
                
                if not span_text or not xbar_label:
                    logger.debug(f"Skipping annotation with empty text='{span_text}' or label='{xbar_label}'")
                    continue
                
                # Find matches using regex
                escaped_text = re.escape(span_text)
                matches = list(re.finditer(escaped_text, text, re.IGNORECASE))
                
                if matches:
                    best_match = self._select_best_match(matches, span_text, text)
                    start_pos, end_pos = best_match.start(), best_match.end() - 1
                    actual_text = text[start_pos:end_pos + 1]
                    
                    if actual_text.lower() == span_text.lower():
                        span_label = SpanLabel(span=(start_pos, end_pos), xbar_label=xbar_label, text=actual_text)
                        spans.append(span_label)
                    else:
                        logger.debug(f"Text mismatch: expected '{span_text}', got '{actual_text}' at {start_pos}-{end_pos}")
                elif len(span_text) > 10:  # Try fuzzy matching for longer phrases
                    fuzzy_span = self._try_fuzzy_match(span_text, text, xbar_label)
                    if fuzzy_span:
                        spans.append(fuzzy_span)
                else:
                    logger.debug(f"Could not find text '{span_text}' in source text")
                    
            except Exception as e:
                logger.debug(f"Failed to parse annotation: {annotation}, error: {e}")
                continue
        
        # Fallback: parse from text format if no JSON found
        if not spans:
            spans = self._parse_text_format_fallback(response)
        
        return self._deduplicate_spans(spans)
    
    def _try_fuzzy_match(self, span_text: str, text: str, xbar_label: str) -> Optional[SpanLabel]:
        """Try fuzzy matching for phrases that may not match exactly."""
        words = span_text.split()
        if len(words) < 2:
            return None
            
        first_words = ' '.join(words[:3]) if len(words) >= 3 else ' '.join(words[:2])
        first_escaped = re.escape(first_words)
        first_matches = list(re.finditer(first_escaped, text, re.IGNORECASE))
        
        if first_matches:
            start_pos = first_matches[0].start()
            search_end = min(start_pos + len(span_text) * 2, len(text))
            extended_text = text[start_pos:search_end]
            
            # Find natural break point
            for break_char in ['.', ';', ':', '\n', '  ']:
                break_pos = extended_text.find(break_char, len(first_words))
                if break_pos > 0:
                    end_pos = start_pos + break_pos - 1
                    if end_pos > start_pos and end_pos < len(text):
                        actual_text = text[start_pos:end_pos + 1].strip()
                        if actual_text:
                            logger.debug(f"Fuzzy matched phrase: '{span_text}' -> '{actual_text[:50]}...'")
                            return SpanLabel(span=(start_pos, end_pos), xbar_label=xbar_label, text=actual_text)
        return None
    
    def _parse_text_format_fallback(self, response: str) -> List[SpanLabel]:
        """Parse spans from text format as fallback."""
        spans = []
        pattern = r'"([^"]*?)"\s*\((\d+)-(\d+)\)\s*->\s*(\w+)'
        for match in re.finditer(pattern, response):
            try:
                span_text = match.group(1)
                start_char = int(match.group(2))
                end_char_inclusive = int(match.group(3))
                xbar_label = match.group(4)
                spans.append(SpanLabel(span=(start_char, end_char_inclusive), xbar_label=xbar_label, text=span_text))
            except Exception as e:
                logger.debug(f"Failed to parse text format span: {match.group(0)}, error: {e}")
        return spans
    
    def _deduplicate_spans(self, spans: List[SpanLabel]) -> List[SpanLabel]:
        """Remove duplicate spans based on position and label."""
        unique_spans = []
        seen_spans = set()
        
        for span in spans:
            start_pos, end_pos = span.span
            span_key = (start_pos, end_pos, span.xbar_label)
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                unique_spans.append(span)
        
        logger.info(f"Parsed {len(unique_spans)} unique spans from {len(spans)} total found")
        return unique_spans
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response with simple, fast error handling."""
        response_stripped = response.strip()
        if not response_stripped:
            return []
        
        # JSON extraction patterns
        patterns = [
            r'```json\s*(\[.*?\])\s*```',       # JSON code block
            r'```\s*(\[.*?\])\s*```',           # Generic code block  
            r'(\[.*?\])',                       # Simple array pattern
        ]
        
        annotations = []
        for pattern in patterns:
            for match in re.findall(pattern, response_stripped, re.DOTALL):
                try:
                    parsed_data = json.loads(match)
                    if isinstance(parsed_data, list):
                        annotations.extend(parsed_data)
                    elif isinstance(parsed_data, dict):
                        annotations.append(parsed_data)
                    logger.debug(f"Successfully parsed {len(annotations)} annotations directly")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    logger.error(f"Malformed JSON content: {match[:200]}...")
                    raise ValueError(f"Failed to parse JSON annotation. Pipeline will exit to prevent hanging on malformed JSON. Error: {e}")
            if annotations:
                break
        
        return self._filter_valid_annotations(annotations)
    
    def _filter_valid_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and deduplicate annotations."""
        seen_annotations = set()
        unique_annotations = []
        empty_count = 0
        
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
                
            text = (annotation.get('text', '') or '').strip()
            xbar_label = (annotation.get('xbar_label', '') or annotation.get('label', '') or 
                         annotation.get('xbar_class', '') or annotation.get('class', '') or '').strip()
            
            if not text or not xbar_label:
                empty_count += 1
                continue
            
            # Filter out literal field names
            if text in ['text', 'label', 'xbar_label'] or xbar_label in ['text', 'label', 'xbar_label']:
                continue
            
            key = (text, xbar_label)
            if key not in seen_annotations:
                seen_annotations.add(key)
                unique_annotations.append(annotation)
        
        if empty_count > 10:
            logger.warning(f"Detected {empty_count} empty/invalid JSON objects - possible LLM output quality issue")
        
        logger.debug(f"Parsed {len(unique_annotations)} unique annotations from {len(annotations)} total found")
        return unique_annotations
    




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
    
    def _select_best_match(self, matches: List[re.Match], span_text: str, full_text: str) -> re.Match:
        """
        Select the best match from multiple occurrences of the same text.
        
        Prioritizes matches that:
        1. Are at word boundaries
        2. Are not nested within other matches
        3. Appear earlier in the text (for deterministic results)
        
        Args:
            matches: List of regex matches for the same text
            span_text: The text being matched
            full_text: Full source text
            
        Returns:
            The best match object
        """
        if len(matches) == 1:
            return matches[0]
        
        # Score each match
        scored_matches = []
        for match in matches:
            score = 0
            start_pos = match.start()
            end_pos = match.end()
            
            # Prefer word boundaries (not in middle of words)
            if start_pos == 0 or not full_text[start_pos - 1].isalnum():
                score += 2
            if end_pos >= len(full_text) or not full_text[end_pos].isalnum():
                score += 2
                
            # Prefer earlier positions for consistency
            score -= start_pos / len(full_text)
            
            scored_matches.append((score, match))
        
        # Return the highest scored match
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        return scored_matches[0][1]
    
    def _normalize_xbar_class(self, xbar_class: str) -> str:
        """
        Normalize XBar class label using the mapping from XBarLabelMap.
        
        Args:
            xbar_class: Input label (can be abbreviation or full name)
            
        Returns:
            Normalized full label name
        """
        return XBarLabelMap.normalize_xbar_class(xbar_class)
    
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
            
            # Three-turn annotation strategy
            all_spans = []
            
            # Turn 1: Word-level spans
            logger.info("Turn 1: Extracting word-level spans")
            word_spans = await self._extract_spans_via_dialogue(text, domain, "word_level", pretrain_record)
            all_spans.extend(word_spans)
            
            # Turn 2: Phrase-level spans
            logger.info("Turn 2: Extracting phrase-level spans")
            phrase_spans = await self._extract_spans_via_dialogue(text, domain, "phrase_level", pretrain_record)
            all_spans.extend(phrase_spans)
            
            # Turn 3: Clause-level spans
            logger.info("Turn 3: Extracting clause-level spans")
            clause_spans = await self._extract_spans_via_dialogue(text, domain, "clause_level", pretrain_record)
            all_spans.extend(clause_spans)
            
            # Validate and filter spans
            logger.info(f"Validating {len(all_spans)} total spans")
            valid_spans = self._validate_and_filter_span_labels(all_spans, text)
            
            # Convert to SpanAnnotation objects
            span_annotations = self._convert_span_labels_to_annotations(valid_spans, text)
            
            # Create annotation record
            annotation_record = AnnotationRecord(
                raw=text,
                sequence_number=pretrain_record.sequence_number or 0,
                total_positions=len(text),
                span_annotations=span_annotations,
                agent_metadata={
                    "strategy": "three_turn_xbar",
                    "model": self.model_config.name,
                    "domain": domain.value,
                    "total_turns": 3,
                    "word_spans": len(word_spans),
                    "phrase_spans": len(phrase_spans),
                    "clause_spans": len(clause_spans),
                    "total_valid_spans": len(valid_spans)
                }
            )
            
            logger.info(f"Successfully annotated sequence with {len(span_annotations)} spans")
            return annotation_record
            
        except Exception as e:
            logger.error(f"Failed to annotate sequence: {e}", exc_info=True)
            return None
    
    def _validate_and_filter_span_labels(self, spans: List[SpanLabel], text: str) -> List[SpanLabel]:
        """Validate and filter span labels removing duplicates and invalid spans."""
        valid_spans = []
        seen_spans = set()
        
        for span in spans:
            try:
                start_pos, end_pos = span.span
                span_text = (span.text or "").strip()
                xbar_label = (span.xbar_label or "").strip()
                
                # Basic validation
                if (start_pos < 0 or end_pos >= len(text) or start_pos > end_pos or 
                    not span_text or not xbar_label or
                    span_text in ['text', 'label', 'xbar_label'] or 
                    xbar_label in ['text', 'label', 'xbar_label']):
                    continue
                
                # Check for duplicates
                span_key = (start_pos, end_pos, xbar_label)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    valid_spans.append(span)
                
            except Exception as e:
                logger.debug(f"Error validating span {span}: {e}")
                continue
        
        logger.info(f"Validated {len(valid_spans)}/{len(spans)} spans")
        return valid_spans
    
    def _convert_span_labels_to_annotations(self, spans: List[SpanLabel], text: str) -> List[SpanAnnotation]:
        """Convert SpanLabel objects to SpanAnnotation objects."""
        annotations = []
        
        for span in spans:
            try:
                start_pos, end_pos = span.span
                span_text = span.text or ""
                actual_text = text[start_pos:end_pos + 1]
                
                if actual_text != span_text:
                    logger.debug(f"Skipping span with mismatched text: expected '{span_text}', got '{actual_text}'")
                    continue
                
                annotation = SpanAnnotation(
                    start_pos=start_pos,
                    end_pos=end_pos + 1,  # Convert to exclusive end for final output
                    xbar_label=span.xbar_label,
                    linguistic_features={
                        'extracted_text': span_text,
                        'character_span': {'start_char': start_pos, 'end_char': end_pos}
                    }
                )
                annotations.append(annotation)
                
            except Exception as e:
                logger.debug(f"Error converting span {span}: {e}")
                continue
        
        return annotations
