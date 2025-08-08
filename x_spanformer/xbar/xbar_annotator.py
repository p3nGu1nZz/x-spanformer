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
        labels = XBarLabelMap.get_labels_for_domain(domain)
        label_descriptions = []
        
        for label, description in labels.items():
            label_descriptions.append(f"- {label}: {description}")
        
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
        
        # Get domain-specific labels
        all_labels = XBarLabelMap.get_labels_for_domain(domain)
        
        # Filter labels by turn focus
        if turn_focus == "word_level":
            # Word-level labels for each domain
            if domain == DomainType.NATURAL:
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["noun", "verb", "adjective", "adverb", "preposition", 
                                         "determiner", "pronoun", "conjunction", "punctuation"]}
                focus_description = "individual WORDS and their grammatical classes"
                examples = '"word" -> noun, "runs" -> verb, "quickly" -> adverb'
            elif domain == DomainType.CODE:
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["keyword", "identifier", "operator", "literal", 
                                         "delimiter", "type_name", "comment"]}
                focus_description = "individual CODE TOKENS and their syntactic types"
                examples = '"if" -> keyword, "variable" -> identifier, "+" -> operator'
            else:  # MIXED
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["noun", "verb", "adjective", "adverb", "preposition",
                                         "determiner", "pronoun", "conjunction", "keyword", 
                                         "identifier", "operator", "literal", "inline_code"]}
                focus_description = "individual WORDS/TOKENS from both natural language and code"
                examples = '"function" -> noun (or keyword in code context), "variable" -> identifier'
                
        elif turn_focus == "phrase_level":
            # Phrase-level labels
            if domain == DomainType.NATURAL:
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["noun_phrase", "verb_phrase", "adjective_phrase", 
                                         "adverb_phrase", "prepositional_phrase"]}
                focus_description = "PHRASES (groups of related words)"
                examples = '"the red car" -> noun_phrase, "is running quickly" -> verb_phrase'
            elif domain == DomainType.CODE:
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["expression", "function_call", "assignment", 
                                         "parameter_list", "argument_list"]}
                focus_description = "CODE EXPRESSIONS and structured constructs"
                examples = '"x + y" -> expression, "func(a, b)" -> function_call'
            else:  # MIXED
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["noun_phrase", "verb_phrase", "expression", "function_call", 
                                         "code_block", "documentation_comment"]}
                focus_description = "PHRASES and CODE EXPRESSIONS from mixed content"
                examples = '"the function call" -> noun_phrase, "func(x)" -> function_call'
                
        else:  # clause_level
            # Clause-level labels
            if domain == DomainType.NATURAL:
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["main_clause", "subordinate_clause", "relative_clause"]}
                focus_description = "CLAUSES and major syntactic structures"
                examples = '"She runs fast" -> main_clause, "because it was late" -> subordinate_clause'
            elif domain == DomainType.CODE:
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["if_statement", "loop_statement", "function_definition", 
                                         "class_definition", "import_statement", "return_statement"]}
                focus_description = "CODE STATEMENTS and control structures"
                examples = '"if x > 0:" -> if_statement, "def func():" -> function_definition'
            else:  # MIXED
                relevant_labels = {k: v for k, v in all_labels.items() 
                                 if k in ["main_clause", "subordinate_clause", "if_statement", 
                                         "loop_statement", "function_definition"]}
                focus_description = "CLAUSES and CODE STATEMENTS from mixed content"
                examples = '"The function returns" -> main_clause, "if condition:" -> if_statement'
        
        # Build label list for prompt
        label_names = list(relevant_labels.keys())
        label_descriptions = [f"{k}: {v}" for k, v in relevant_labels.items()]
        
        # Build system prompt
        system_prompt = f"""You are a linguistic annotator specializing in {domain.value} domain {turn_focus.replace('_', '-')} analysis.

Domain: {domain.value.upper()}
Focus: {focus_description}

Available labels:
{chr(10).join(f"- {desc}" for desc in label_descriptions)}

Extract accurate spans using ONLY these labels. Be precise and consistent."""

        # Build user prompt
        user_prompt = f"""Analyze this {domain.value} text and identify {focus_description}:
"{text_snippet}"

Return ONLY a JSON array with this exact format. Do not include any explanations, notes, or additional text:
[{{"text":"extracted_text","xbar_label":"label_name"}}]

Examples: {examples}

Use these labels: {", ".join(label_names)}"""

        return system_prompt, user_prompt
    
    async def _extract_spans_via_dialogue(
        self, 
        text: str, 
        domain: DomainType,
        turn_focus: str,
        pretrain_record: PretrainRecord
    ) -> List[SpanLabel]:
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
            # Import chat function and DialogueManager locally to avoid circular imports
            from x_spanformer.agents.ollama_client import chat
            from x_spanformer.agents.dialogue import DialogueManager
            
            # Detect domain for this sequence
            domain = self._detect_domain_from_record(pretrain_record)
            
            # Build focused prompt for this turn - use larger text window for better context
            text_snippet = text[:200] if len(text) > 200 else text
            
            # Create domain-specific turn prompts
            system_prompt, user_prompt = self._build_domain_specific_prompts(
                domain, turn_focus, text_snippet
            )
            
            # Use DialogueManager for proper conversation handling
            dm = DialogueManager(system_prompt=system_prompt, max_turns=1)
            dm.add("user", user_prompt)
            
            # Get response from ollama with increased timeout for complex text
            response = await chat(
                model=self.model_config.name,
                conversation=dm.as_messages(),
                temperature=0.1,
                timeout=90.0
            )
            
            # Add assistant response to dialogue for logging
            dm.add("assistant", response)
            
            # Parse spans from response
            spans = self._parse_spans_from_response(response, text)
            
            logger.info(f"Extracted {len(spans)} spans for {turn_focus} from dialogue")
            return spans
            
        except Exception as e:
            logger.error(f"Failed to extract spans via dialogue: {e}")
            return []
    
    def _parse_spans_from_response(self, response: str, text: str) -> List[SpanLabel]:
        """
        Parse spans from LLM response using regex-based text matching.
        
        Args:
            response: Raw LLM response
            text: Original text for validation
            
        Returns:
            List of parsed and validated span labels
        """
        spans = []
        
        # Parse JSON annotations from response
        json_annotations = self._parse_json_response(response)
        
        for annotation in json_annotations:
            try:
                # Extract required fields
                span_text = annotation.get('text', '').strip()
                xbar_label = annotation.get('xbar_label', annotation.get('label', annotation.get('xbar_class', annotation.get('class', '')))).strip()
                
                if not span_text or not xbar_label:
                    continue
                
                # Use regex to find all occurrences of this text in the original text
                # Escape special regex characters in the span text and clean whitespace
                span_text = span_text.strip()  # Remove leading/trailing whitespace
                if not span_text:  # Skip empty spans
                    continue
                    
                escaped_text = re.escape(span_text)
                
                # Find all matches (case-insensitive for all input segments)
                matches = list(re.finditer(escaped_text, text, re.IGNORECASE))
                
                if matches:
                    # Choose the best match based on context and position
                    best_match = self._select_best_match(matches, span_text, text)
                    start_pos = best_match.start()
                    end_pos = best_match.end() - 1  # Convert to inclusive end for internal storage
                    
                    # Validate that extracted text matches (case-insensitive for all input segments)
                    actual_text = text[start_pos:end_pos + 1]
                    # For case-insensitive matches, compare lowercased versions
                    if actual_text.lower() != span_text.lower():
                        logger.debug(f"Text mismatch (case-insensitive): expected '{span_text}', got '{actual_text}' at {start_pos}-{end_pos}")
                        continue
                    # Use the actual text from the source for the span
                    span_text_to_use = actual_text
                    
                    # Create SpanLabel object
                    span_label = SpanLabel(
                        span=(start_pos, end_pos),
                        xbar_label=xbar_label,
                        text=span_text_to_use
                    )
                    spans.append(span_label)
                else:
                    logger.debug(f"Could not find text '{span_text}' in source text")
                    
            except Exception as e:
                logger.debug(f"Failed to parse annotation: {annotation}, error: {e}")
                continue
        
        # Fallback: parse from text format if no JSON found
        if not spans:
            pattern = r'"([^"]*?)"\s*\((\d+)-(\d+)\)\s*->\s*(\w+)'
            matches = re.finditer(pattern, response)
            
            for match in matches:
                try:
                    span_text = match.group(1)
                    start_char = int(match.group(2))
                    end_char_inclusive = int(match.group(3))
                    xbar_label = match.group(4)
                    
                    # Create SpanLabel object
                    span_label = SpanLabel(
                        span=(start_char, end_char_inclusive),
                        xbar_label=xbar_label,
                        text=span_text
                    )
                    spans.append(span_label)
                except Exception as e:
                    logger.debug(f"Failed to parse text format span: {match.group(0)}, error: {e}")
                    continue
        
        # Remove duplicates based on position and label
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
            start_pattern = r'"(?:start|start_char)":\s*(\d+)'
            end_pattern = r'"(?:end|end_char)":\s*(\d+)'
            
            text_matches = re.findall(text_pattern, malformed_str)
            class_matches = re.findall(class_pattern, malformed_str)
            start_matches = re.findall(start_pattern, malformed_str)
            end_matches = re.findall(end_pattern, malformed_str)
            
            # Try to pair them up
            for i, text in enumerate(text_matches):
                annotation = {'text': text}
                
                if i < len(class_matches):
                    annotation['label'] = class_matches[i]
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
        """
        Validate and filter span labels removing duplicates and invalid spans.
        
        Args:
            spans: List of SpanLabel objects to validate
            text: Original text for validation
            
        Returns:
            List of valid, deduplicated SpanLabel objects
        """
        valid_spans = []
        seen_spans = set()
        
        for span in spans:
            try:
                start_pos, end_pos = span.span
                span_text = span.text or ""
                xbar_label = span.xbar_label
                
                # Basic validation
                if start_pos < 0 or end_pos >= len(text) or start_pos > end_pos:
                    logger.debug(f"Invalid span positions: {span.span} for text length {len(text)}")
                    continue
                
                if not span_text or not xbar_label:
                    logger.debug(f"Missing text or label: {span}")
                    continue
                
                # Check for duplicates
                span_key = (start_pos, end_pos, xbar_label)
                if span_key in seen_spans:
                    continue
                
                seen_spans.add(span_key)
                valid_spans.append(span)
                
            except Exception as e:
                logger.debug(f"Error validating span {span}: {e}")
                continue
        
        logger.info(f"Validated {len(valid_spans)}/{len(spans)} spans")
        return valid_spans
    
    def _convert_span_labels_to_annotations(self, spans: List[SpanLabel], text: str) -> List[SpanAnnotation]:
        """
        Convert SpanLabel objects to SpanAnnotation objects.
        
        Args:
            spans: List of validated SpanLabel objects
            text: Original text
            
        Returns:
            List of SpanAnnotation objects
        """
        annotations = []
        
        for span in spans:
            try:
                start_pos, end_pos = span.span
                span_text = span.text or ""
                
                # Validate span boundaries one more time
                actual_text = text[start_pos:end_pos + 1]
                if actual_text != span_text:
                    logger.debug(f"Skipping span with mismatched text: expected '{span_text}', got '{actual_text}' at {start_pos}-{end_pos}")
                    continue
                
                # Create SpanAnnotation with correct end position (keep inclusive internally)
                annotation = SpanAnnotation(
                    start_pos=start_pos,
                    end_pos=end_pos + 1,  # Convert to exclusive end only for final output
                    xbar_label=span.xbar_label,
                    linguistic_features={
                        'extracted_text': span_text,
                        'character_span': {
                            'start_char': start_pos,
                            'end_char': end_pos  # Keep original inclusive end in metadata
                        }
                    }
                )
                annotations.append(annotation)
                
            except Exception as e:
                logger.debug(f"Error converting span {span}: {e}")
                continue
        
        return annotations
