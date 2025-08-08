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
        
        This method implements efficient error recovery aligned with the mathematical
        rigor described in Section 3.3 (Factorized Pointer Networks) of the 
        X-Spanformer architecture paper.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            List of parsed annotation dictionaries
        """
        annotations = []
        
        # Check for obviously truncated responses
        if response.strip().endswith(('"}', '"}}', '"]')) and not response.strip().endswith(']}'):
            logger.warning("Detected truncated response - attempting recovery")
            # Try to append missing closing bracket
            response = response.strip() + ']'
        
        # Try to extract JSON from response with comprehensive patterns
        # Priority ordered: try most likely patterns first for efficiency
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',       # JSON code block with array (highest priority)
            r'```\s*(\[.*?\])\s*```',           # Generic code block with array
            r'```json\s*(\[.*?)\s*```',         # JSON code block with truncated array
            r'```\s*(\[.*?)\s*```',             # Generic code block with truncated array  
            r'(\[(?:\s*\{[^}]*\},?\s*)*\])',    # JSON arrays with objects
            r'```json\s*(\{.*?\})\s*```',       # JSON code block with object
            r'```\s*(\{.*?\})\s*```',           # Generic code block with object
        ]
        
        # Track attempted fixes to prevent infinite loops
        attempted_fixes = set()
        successful_parse = False
        
        for pattern in json_patterns:
            if successful_parse:
                break  # Exit early if we successfully parsed complete JSON
                
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            for match in matches:
                # Skip if we've already attempted to fix this exact text
                match_hash = hash(match)
                if match_hash in attempted_fixes:
                    logger.debug(f"Skipping duplicate fix attempt for JSON fragment")
                    continue
                    
                attempted_fixes.add(match_hash)
                
                try:
                    # Try direct parsing first
                    parsed_data = json.loads(match)
                    if isinstance(parsed_data, list):
                        annotations.extend(parsed_data)
                        successful_parse = True
                    elif isinstance(parsed_data, dict):
                        annotations.append(parsed_data)
                    logger.debug(f"Successfully parsed JSON directly")
                    break  # Move to next pattern if successful
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed: {e}")
                    logger.debug(f"Problematic JSON snippet: {match[:100]}...")  # Reduced log size
                    
                    # Apply circuit breaker: only attempt fix if error is recoverable
                    if self._is_recoverable_json_error(e, match):
                        try:
                            fixed_json = self._fix_malformed_json(match)
                            parsed_data = json.loads(fixed_json)
                            if isinstance(parsed_data, list):
                                annotations.extend(parsed_data)
                                successful_parse = True
                            elif isinstance(parsed_data, dict):
                                annotations.append(parsed_data)
                            logger.debug(f"Successfully fixed and parsed JSON")
                            break  # Move to next pattern if successful
                            
                        except json.JSONDecodeError as e2:
                            logger.debug(f"JSON fix failed: {e2}")
                            # Try regex recovery when fix fails
                            recovered_annotations = self._recover_malformed_json(match)
                            if recovered_annotations:
                                logger.debug(f"Recovered {len(recovered_annotations)} annotations via regex")
                                annotations.extend(recovered_annotations)
                            # Continue to next match rather than trying same pattern repeatedly
                    else:
                        logger.debug(f"JSON error not recoverable, trying regex recovery")
                        recovered_annotations = self._recover_malformed_json(match)
                        if recovered_annotations:
                            logger.debug(f"Recovered {len(recovered_annotations)} annotations via regex")
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
                
                # Filter out invalid/junk annotations
                if (key not in seen_annotations and 
                    text.strip() and 
                    text.strip() not in ['text', 'label', 'xbar_label'] and  # Filter literal field names
                    len(text.strip()) > 0):
                    seen_annotations.add(key)
                    unique_annotations.append(annotation)
        
        logger.debug(f"Parsed {len(unique_annotations)} unique annotations from {len(annotations)} total found")
        return unique_annotations
    
    def _is_recoverable_json_error(self, error_or_json: Union[json.JSONDecodeError, str], json_str: Optional[str] = None) -> bool:
        """
        Determine if a JSON parsing error is likely recoverable through fixing.
        
        This implements efficient error classification aligned with the architectural
        principle of mathematical rigor from Section 3.3.
        
        Args:
            error_or_json: Either a JSON decode error or JSON string
            json_str: The malformed JSON string (optional, used when first arg is error)
            
        Returns:
            True if error appears recoverable, False otherwise
        """
        # Handle both call patterns: (error, json_str) and (json_str)
        if isinstance(error_or_json, str):
            json_content = error_or_json
            # Try to detect recoverable patterns in the JSON content
            recoverable_patterns = [
                r'"[^"]*"\s*"[^"]*"',              # Missing colon: "text" "value"
                r'"[^"]*"\s*:\s*"[^"]*"\s*"[^"]*"', # Missing comma: "text":"val" "other"
                r'\w+\s*:\s*"[^"]*"',              # Missing quotes on property name
                r'\{[^}]*$',                       # Incomplete object
                r'\[[^\]]*$',                      # Incomplete array
            ]
            return any(re.search(pattern, json_content) for pattern in recoverable_patterns)
        
        # Original error-based checking
        error = error_or_json
        
        # Recoverable error types (based on our test patterns)
        recoverable_messages = [
            "Expecting ':' delimiter",      # Missing colon after property name
            "Expecting ',' delimiter",      # Missing comma between properties
            "Expecting property name",      # Missing quotes around property names
            "Extra data",                   # Extra content after valid JSON
            "Unterminated string",          # Missing closing quote
        ]
        
        # Check if error message indicates a recoverable pattern
        for recoverable_msg in recoverable_messages:
            if recoverable_msg in error.msg:
                return True
        
        # Additional heuristics for recoverability
        # When called with string format, json_str parameter should be used for context if available
        json_content = json_str if json_str is not None else ""
        
        # If the JSON content contains expected field names, it's likely recoverable
        if json_content and any(field in json_content for field in ['text', 'xbar_label', 'label']):
            return True
            
        # If it's a very short string with basic structure, try to recover
        if json_content and len(json_content) < 100 and ('{' in json_content or '[' in json_content):
            return True
            
        return False
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """
        Fix common JSON formatting issues with comprehensive pattern matching.
        
        This method implements the error recovery patterns identified from production
        logs and validated through our comprehensive test suite.
        """
        original_str = json_str
        
        # Early return for already valid JSON or hopeless cases
        if not json_str.strip():
            return json_str
            
        # Handle truncated JSON - if it ends with incomplete objects
        if json_str.strip().endswith(('"}', '"}}')):
            if not json_str.strip().endswith((']}', ')]')):
                # Try to close incomplete arrays
                json_str = json_str.strip() + ']'
        
        # Handle incomplete object at end
        if json_str.strip().endswith(','):
            json_str = json_str.strip()[:-1]  # Remove trailing comma
        
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # CRITICAL FIX: Handle the specific production error pattern
        # {"text","xbar_label":"literal"} - missing colon after "text"
        # This is the exact pattern causing the repetitive failures in production
        
        # Pattern 1: Fix missing colon after "text" specifically
        # {"text","xbar_label":"value"} -> {"text":"","xbar_label":"value"}
        json_str = re.sub(r'\{"text",', r'{"text":"",', json_str)
        
        # Pattern 2: Fix specific missing colon after known field names
        # Only target known field names that should have colons
        json_str = re.sub(r'"(text|label|xbar_label|xbar_class|class|start|end|start_pos|end_pos)"\s*,\s*"([^"]+)"\s*:', r'"\1":"", "\2":', json_str)
        
        # Pattern 3: Standard missing colon patterns
        # "text" "value" -> "text": "value" (most common)
        json_str = re.sub(r'"(text|label|xbar_label|xbar_class|class|start|end|start_pos|end_pos)"\s+"([^"]*)"', r'"\1": "\2"', json_str)
        
        # Pattern 4: More general quoted field followed by quoted value
        json_str = re.sub(r'"([^"]+)"\s+"([^"]*)"(?=\s*[,}\]])', r'"\1": "\2"', json_str)
        
        # Pattern 5: Unquoted field names followed by quoted values
        json_str = re.sub(r'(\w+)\s+"([^"]*)"', r'"\1": "\2"', json_str)
        
        # Pattern 6: Handle multiple missing colons in sequence
        # "text" "value", "label" "type" -> "text": "value", "label": "type"
        json_str = re.sub(r'"([^"]+)"\s+"([^"]*)",\s*"([^"]+)"\s+"([^"]*)"', r'"\1": "\2", "\3": "\4"', json_str)
        
        # ENHANCED: Fix missing commas between key-value pairs
        
        # Pattern 7: "key": "value" "key2": "value2" -> "key": "value", "key2": "value2"
        json_str = re.sub(r'"\s+"([^"]+)"\s*:', r'", "\1":', json_str)
        
        # Pattern 8: Missing comma between objects in arrays
        json_str = re.sub(r'}\s*{', '}, {', json_str)
        
        # Pattern 9: Missing comma after string values before next field
        # "value" "field": -> "value", "field":
        json_str = re.sub(r'"\s+"([^"]+)"\s*:', r'", "\1":', json_str)
        
        # ENHANCED: Fix property name issues
        
        # Pattern 10: Missing quotes around property names (only at start of line or after {,)
        # Avoid matching inside quoted strings by requiring word boundaries
        json_str = re.sub(r'(?<=[\{\s,])(\w+)(?=\s*:)', r'"\1"', json_str)
        
        # Pattern 11: Fix incomplete objects and arrays
        
        # Handle cases where opening brace/bracket exists but object is malformed
        if '{' in json_str and '}' not in json_str:
            # Try to complete the object
            if '"' in json_str:
                json_str += '"}'
            else:
                json_str += '}'
                
        if '[' in json_str and ']' not in json_str:
            # Try to complete the array
            json_str += ']'
        
        # Pattern 12: Fix extra data issues - remove text after valid JSON
        # Look for pattern like: [{"text":"word"}] extra text
        # Use greedy matching to get the full array content
        match = re.search(r'(\[.*\]|\{.*\})', json_str, re.DOTALL)
        if match:
            potential_json = match.group(1)
            try:
                # Test if this parses successfully
                import json
                json.loads(potential_json)
                json_str = potential_json  # Use only the valid JSON part
            except:
                pass  # Continue with other fixes
        
        # Pattern 13: Fix quotes around string values when missing
        # "field": value -> "field": "value" (where value isn't a number/boolean)
        json_str = re.sub(r'"([^"]+)":\s*([^",\s{}\[\]]+)(?=[,}\]])', r'"\1": "\2"', json_str)
        
        # Pattern 14: Handle malformed arrays of objects
        # Fix patterns like: [{"text":"word""label":"noun"}] -> [{"text":"word","label":"noun"}]
        json_str = re.sub(r'""', '","', json_str)
        
        # Pattern 15: Fix concatenated JSON objects
        # {"text":"word"}{"label":"noun"} -> {"text":"word","label":"noun"}
        json_str = re.sub(r'}\s*{', ',', json_str)
        
        # If we made changes, log for debugging (but limit log size)
        if original_str != json_str and len(original_str) < 200:
            logger.debug(f"JSON transformation: {original_str} -> {json_str}")
        elif original_str != json_str:
            logger.debug(f"JSON transformation: {original_str[:50]}... -> {json_str[:50]}...")
        
        return json_str
    
    def _recover_malformed_json(self, malformed_str: str) -> List[Dict[str, Any]]:
        """
        Optimized recovery from malformed JSON focusing on production error patterns.
        
        This method implements the specific pattern {"text","xbar_label":"literal"}
        and other production error cases with early termination to prevent repetitive processing.
        """
        recovered = []
        
        try:
            # PRIORITY RECOVERY: Handle the exact production error pattern first
            # {"text","xbar_label":"literal"} - this is the most common production failure
            production_pattern = r'\{"text"\s*,\s*"xbar_label"\s*:\s*"([^"]*)"'
            production_matches = re.finditer(production_pattern, malformed_str, re.IGNORECASE)
            
            for match in production_matches:
                label_value = match.group(1).strip()
                # Filter out literal field names to prevent 'text': 'text' patterns
                if label_value and label_value.lower() not in ['text', 'literal', 'xbar_label', 'label']:
                    recovered.append({
                        'text': label_value,  # Use label as text since original text is malformed
                        'xbar_label': label_value,
                        'start': 0,
                        'end': len(label_value)
                    })
                    logger.debug(f"Recovered from production pattern: {label_value}")
            
            # Early return if production pattern recovery succeeded
            if recovered:
                logger.debug(f"Production pattern recovery successful: {len(recovered)} items")
                return recovered
            
            # STANDARD RECOVERY PATTERNS (priority ordered)
            
            # Pattern 1: Standard missing colon patterns
            text_patterns = [
                r'"text"\s*:\s*"([^"]*)"',           # Standard: "text": "value"
                r'"text"\s+"([^"]*)"',               # Missing colon: "text" "value"
                r'text\s*:\s*"([^"]*)"',             # Missing quotes on key: text: "value"
            ]
            
            label_patterns = [
                r'"(?:label|xbar_label|xbar_class|class)"\s*:\s*"([^"]*)"',    # Standard
                r'"(?:label|xbar_label|xbar_class|class)"\s+"([^"]*)"',        # Missing colon
                r'(?:label|xbar_label|xbar_class|class)\s*:\s*"([^"]*)"',      # Missing quotes on key
            ]
            
            # Extract text and label values
            text_matches = []
            for pattern in text_patterns:
                matches = re.findall(pattern, malformed_str, re.IGNORECASE)
                text_matches.extend([m.strip() for m in matches if m.strip()])
            
            label_matches = []
            for pattern in label_patterns:
                matches = re.findall(pattern, malformed_str, re.IGNORECASE)
                label_matches.extend([m.strip() for m in matches if m.strip()])
            
            # Pattern 2: Space-separated key-value pairs
            # "text" "word" "label" "noun" -> {"text": "word", "label": "noun"}
            if not text_matches and not label_matches:
                quoted_parts = re.findall(r'"([^"]*)"', malformed_str)
                if len(quoted_parts) >= 2:
                    for i in range(0, len(quoted_parts) - 1, 2):
                        key = quoted_parts[i].strip().lower()
                        value = quoted_parts[i + 1].strip()
                        
                        if key == 'text' and value:
                            text_matches.append(value)
                        elif key in ['label', 'xbar_label', 'class', 'xbar_class'] and value:
                            label_matches.append(value)
            
            # Combine text and label matches
            if text_matches and label_matches:
                max_pairs = min(len(text_matches), len(label_matches))
                for i in range(max_pairs):
                    text_val = text_matches[i]
                    label_val = label_matches[i]
                    if text_val and label_val:
                        recovered.append({
                            'text': text_val,
                            'xbar_label': label_val,
                            'start': 0,
                            'end': len(text_val)
                        })
            
            # Pattern 3: Single field extraction (fallback)
            elif text_matches:
                for text_val in text_matches[:3]:  # Limit to 3 items
                    if text_val:
                        recovered.append({
                            'text': text_val,
                            'xbar_label': 'extracted',
                            'start': 0,
                            'end': len(text_val)
                        })
            
            elif label_matches:
                for label_val in label_matches[:3]:  # Limit to 3 items
                    if label_val:
                        recovered.append({
                            'text': label_val,
                            'xbar_label': label_val,
                            'start': 0,
                            'end': len(label_val)
                        })
            
            # Pattern 4: Last resort - extract meaningful words
            if not recovered:
                # Look for words that aren't field names
                word_matches = re.findall(r'\b[A-Za-z]{2,}\b', malformed_str)
                field_names = {'text', 'label', 'xbar_label', 'class', 'start', 'end', 'pos', 'literal'}
                
                meaningful_words = [
                    word for word in word_matches[:5] 
                    if word.lower() not in field_names and not word.isdigit()
                ]
                
                for word in meaningful_words[:2]:  # Limit to 2 words max
                    recovered.append({
                        'text': word,
                        'xbar_label': 'recovered',
                        'start': 0,
                        'end': len(word)
                    })
            
            # Remove duplicates while preserving order
            unique_recovered = []
            seen_texts = set()
            for annotation in recovered:
                if isinstance(annotation, dict) and 'text' in annotation:
                    text = annotation['text']
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        unique_recovered.append(annotation)
            
            recovered = unique_recovered
            
        except Exception as e:
            logger.debug(f"Recovery attempt failed: {e}")
            
        logger.debug(f"Recovered {len(recovered)} annotations from malformed JSON")
        return recovered[:5]  # Limit to max 5 annotations
    
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
