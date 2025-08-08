#!/usr/bin/env python3
"""
X-bar JSON Parser and Repair Module

Handles JSON parsing and repair for LLM-generated annotation responses.
Provides robust error handling and repair patterns for common LLM JSON malformations.
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class XBarJsonParser:
    """
    Comprehensive JSON parser and repair system for X-bar annotation responses.
    
    Handles common LLM JSON generation errors including:
    - Missing quotes around property names/values
    - Truncated responses
    - Malformed object structures
    - Repetitive garbage output
    - Encoding issues
    """
    
    def __init__(self):
        """Initialize the JSON parser with repair patterns."""
        self.repair_patterns = self._initialize_repair_patterns()
    
    def _initialize_repair_patterns(self) -> List[tuple]:
        """Initialize the comprehensive repair patterns for LLM JSON errors."""
        return [
            # NULL VALUE FIXES FIRST - must come before any general patterns
            (r'"xbar_label"\s*:\s*null(?=\s*[,}])', r'"xbar_label":"unknown"', 'fix_null_xbar_label'),
            (r'("text"\s*:\s*"[^"]*")\s*,\s*null(?=\s*})', r'\1,"xbar_label":"unknown"', 'fix_null_after_text'),
            (r'(\{"text"\s*:\s*"[^"]*")\s*,\s*null\s*}', r'\1,"xbar_label":"unknown"}', 'fix_null_terminating_object'),
            (r':\s*null(?=\s*[,}])', r':"unknown"', 'fix_unquoted_null_values'),
            (r':\s*(true|false|undefined|none)(?=\s*[,}])', r':"\1"', 'fix_unquoted_literals'),
            
            # SPECIFIC FIXES for failing unit tests - handle the exact patterns first
            (r'(\{"text"),("[^"]*"),("xbar_label":"[^"]*"\})', r'\1:\2,\3', 'fix_comma_instead_of_colon_exact'),
            (r'\{"text",,"xbar_label":"([^"]*)"\}', r'{"text":"","xbar_label":"\1"}', 'fix_double_comma_pattern'),
            
            # Early detection patterns for massive repetitive blocks
            (r'(\{"text"\s*,\s*"xbar_label"\s*:\s*"[^"]*"\s*,?\s*\}[\s,]*){5,}', '', 'remove_repetitive_blocks'),
            (r'(\{"text","xbar_label":"",\}[\s,]*){3,}', '', 'remove_empty_repetitive_blocks'),
            
            # Core malformation patterns (order matters)
            (r'\{"text"\s*,\s*"xbar_label"\s*:\s*"[^"]*"\s*,?\s*\}', r'{"text":"","xbar_label":"unknown"}', 'fix_comma_instead_colon'),
            (r'\{"text"\s*,\s*("[^"]*")\s*,\s*"xbar_label"\s*:\s*("[^"]*")\s*\}', r'{"text":\1,"xbar_label":\2}', 'fix_text_comma_value_specific'),
            (r'(\{"text")\s*,\s*("[^"]*")\s*,\s*("xbar_label")', r'\1:\2,\3', 'fix_text_comma_value'),
            (r'(\{"text")\s*,\s*"([^"]*)""\s*,\s*("xbar_label")', r'\1:"\2",\3', 'fix_quoted_text_with_extra_quote'),
            (r'(\{"text")\s*,\s*"([^"]*)}"\s*,\s*("xbar_label")', r'\1:"\2",\3', 'fix_text_with_brace_in_quote'),
            (r'(\{"text")\s*,\s*"([^"]*)"?\s+("xbar_label")', r'\1:"\2",\3', 'fix_text_missing_comma'),
            (r'(\{"text")\s+([^,}]+)\s*,\s*("xbar_label")', r'\1:\2,\3', 'fix_text_missing_colon'),
            (r'\{"text"\s*,\s*("xbar_label"\s*:\s*"[^"]*")\s*\}', r'{"text":"",\1}', 'fix_standalone_property'),
            (r'\{"text"\s*,\s*("xbar_label"\s*:\s*"[^"]*")\s*,\s*\}', r'{"text":"",\1}', 'fix_text_missing_value_trailing_comma'),
            
            # Sequence 7-9 specific patterns
            (r'(\{"text"\s*:\s*"[^"]*")\s*,\s*("[^"]*")(?=\s*[,}])', r'\1,"xbar_label":\2', 'fix_missing_xbar_label_key'),
            (r'(\{"text"\s*:\s*"[^"]*")\s*,\s*("(?:noun|verb|adjective|adverb|determiner|preposition|pronoun|conjunction|punctuation|keyword|identifier|operator|literal|inline_code)")(?=\s*[,}])', r'\1,"xbar_label":\2', 'fix_bare_label_value'),
            (r'\{"text"\s*,\s*"([^"]*?)"\s*,\s*"xbar_label"\s*:\s*"([^"]*?)"\s*\}', r'{"text":"\1","xbar_label":"\2"}', 'fix_comma_instead_colon_in_text'),
            (r'\{"text"\s*,\s*""\s*,\s*"xbar_label"\s*:\s*"([^"]*?)"\s*\}', r'{"text":"","xbar_label":"\1"}', 'fix_empty_text_with_comma'),
            (r'\{"text"\s*,\s*"([^"]*?)"\s*\}', r'{"text":"","xbar_label":"\1"}', 'fix_single_word_labels'),
            (r'(\{"text")([^:,]*?)\s*,\s*("xbar_label"\s*:\s*"[^"]*")\s*\}', r'\1:"\2",\3}', 'fix_unquoted_text_special_chars'),
            (r'("xbar_label"\s*:\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*[,}])', r'\1"\2"', 'fix_unquoted_xbar_label_values'),
            
            # Critical pattern fixes
            (r'\{"text"\s*:\s*"([^"]*)"\s*,\s*xbar_label\s*:\s*"([^"]*)"\s*\}', r'{"text":"\1","xbar_label":"\2"}', 'fix_unquoted_property_name'),
            (r'\{"text",\s*"([^"]*?)"\s*,\s*("xbar_label"\s*:\s*"[^"]*")\s*\}', r'{"text":"\1",\2}', 'fix_comma_instead_colon_pattern'),
            (r':\s*"([^"]*?)"\s*,\s*xbar_label\s*:', r': "\1", "xbar_label":', 'fix_missing_quotes_around_property'),
            (r',\s*xbar_label\s*:', r', "xbar_label":', 'fix_unquoted_property_general'),
            (r':\s*"\(\s*",\s*xbar_label\s*:\s*"([^"]*)"', r': "(", "xbar_label": "\1"', 'fix_specific_char_358_pattern'),
            
            # Empty text removal patterns
            (r',?\s*\{"text"\s*:\s*""\s*,\s*"xbar_label"\s*:\s*"[^"]*"\s*\}\s*,?', r'', 'remove_empty_text_entries'),
            
            # Property name and value fixes
            (r'([{,]\s*)(text|xbar_label|label)(\s*:)', r'\1"\2"\3', 'fix_unquoted_property_names'),
            (r'("(?:text|xbar_label|label)"\s*:\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*[,}])', r'\1"\2"', 'fix_unquoted_property_values'),
            (r'("(?:text|xbar_label|label)")\s+("[^"]*")(?=\s*[,}])', r'\1:\2', 'fix_missing_colon_between_property_value'),
            (r'"xbar_label"\s*:\s*"proper\s+noun"', r'"xbar_label":"noun"', 'fix_proper_noun_label'),
            (r'"xbar_label"\s*:\s*"([^"]*\s+[^"]*)"', r'"xbar_label":"\1"', 'fix_multi_word_labels_placeholder'),
            
            # Structural fixes
            (r',\s*([}\]])', r'\1', 'fix_trailing_commas'),
            (r',\s*,+', r',', 'fix_double_commas'),
            (r'\[\s*,', r'[', 'fix_leading_comma_in_array'),
            (r'}\s*{', r'},{', 'fix_missing_comma_between_objects'),
            (r':\s*(?!null|unknown|true|false|undefined|none)([^",\[\]{}\s]+)(?=\s*[,}\]])', r':"\1"', 'fix_unquoted_values_general'),
            
            # Final cleanup patterns
            (r'(\{"text":"","xbar_label":"[^"]*"\}\s*,?\s*){3,}', r'{"text":"","xbar_label":"unknown"},', 'remove_consecutive_empty_entries'),
            (r'(\{"text":"","xbar_label":""\}\s*,?\s*)+', r'', 'remove_completely_empty_entries'),
            (r'{"text","xbar_label":"([^"]*)",}', r'{"text":"","xbar_label":"\1"}', 'fix_specific_malformed_pattern'),
            (r'("text")\s*,\s*("xbar_label")(\s*:)', r'\1:"",\2\3', 'fix_incomplete_property_names'),
        ]
    
    def parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response with comprehensive error handling and repair."""
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
                    logger.warning(f"JSON parsing failed, attempting repair: {e}")
                    
                    # Debug: Show context around the error position
                    if hasattr(e, 'pos'):
                        pos = e.pos
                        start = max(0, pos - 50)
                        end = min(len(match), pos + 50)
                        context = match[start:end]
                        logger.debug(f"JSON error at position {pos}: '{context}'")
                        
                        # Show the actual characters around the error
                        if pos < len(match):
                            logger.debug(f"Character at error position: '{match[pos]}' (ascii: {ord(match[pos]) if pos < len(match) else 'EOF'})")
                            logger.debug(f"Characters around error: {[match[i] for i in range(max(0, pos-5), min(len(match), pos+5))]}")
                    
                    # Try JSON repair
                    repaired_json = self.attempt_json_repair(match)
                    if repaired_json != match:  # Only try if repair actually changed something
                        try:
                            parsed_data = json.loads(repaired_json)
                            if isinstance(parsed_data, list):
                                annotations.extend(parsed_data)
                            elif isinstance(parsed_data, dict):
                                annotations.append(parsed_data)
                            logger.info(f"Successfully repaired and parsed JSON with {len(annotations)} annotations")
                            break
                        except json.JSONDecodeError as repair_error:
                            logger.warning(f"JSON repair attempt failed: {repair_error}")
                            logger.error(f"Original JSON content: {match[:200]}...")
                            logger.error(f"Repaired JSON content: {repaired_json[:200]}...")
                            raise ValueError(f"Failed to parse JSON annotation even after repair. Original error: {e}, Repair error: {repair_error}")
                    
                    logger.error(f"JSON parsing failed and no repair was attempted: {e}")
                    logger.error(f"Malformed JSON content: {match[:200]}...")
                    raise ValueError(f"Failed to parse JSON annotation. Pipeline will exit to prevent hanging on malformed JSON. Error: {e}")
            if annotations:
                break
        
        return self.filter_valid_annotations(annotations)
    
    def attempt_json_repair(self, json_str: str) -> str:
        """Attempt comprehensive JSON repair using predefined patterns."""
        try:
            # Debug: log the original malformed JSON
            logger.debug(f"REPAIR INPUT: {repr(json_str)}")
            
            repaired = json_str.strip()
            
            # Apply all repair patterns in order
            for pattern, replacement, description in self.repair_patterns:
                if description == 'fix_multi_word_labels_placeholder':
                    # Special handling for multi-word labels
                    repaired = re.sub(pattern, lambda m: f'"xbar_label":"{m.group(1).replace(" ", "_")}"', repaired)
                else:
                    repaired = re.sub(pattern, replacement, repaired)
            
            # Handle truncated responses
            repaired = self._repair_truncated_json(repaired)
            
            # Handle unterminated strings
            repaired = self._repair_unterminated_strings(repaired)
            
            # Final structural repairs
            repaired = self._repair_final_structure(repaired)
            
            # Debug: log the repaired JSON
            logger.debug(f"REPAIR OUTPUT: {repr(repaired)}")
            
            return repaired
            
        except Exception as e:
            logger.debug(f"REPAIR EXCEPTION: {e}")
            return json_str  # Return original if repair fails
    
    def _repair_truncated_json(self, json_str: str) -> str:
        """Handle truncated JSON responses."""
        if json_str.startswith('[') and not json_str.endswith(']'):
            # Check for unclosed objects
            open_braces = json_str.count('{') - json_str.count('}')
            
            # Handle specific truncation patterns
            if open_braces > 0:
                # Pattern 1: Incomplete entry at end like: {"text":"wo
                truncated_entry_pattern = r',\s*\{"text":"([^"]*)"?\s*$'
                match = re.search(truncated_entry_pattern, json_str)
                if match:
                    # Complete the truncated entry
                    text_value = match.group(1)
                    # Remove the incomplete entry
                    json_str = re.sub(truncated_entry_pattern, '', json_str)
                    # Add completed entry
                    if not json_str.rstrip().endswith(','):
                        json_str += ','
                    json_str += f'{{"text":"{text_value}","xbar_label":"unknown"}}'
                else:
                    # Pattern 2: Complete object missing closing brace
                    # Like: [{"text":"form","xbar_label":"noun"
                    # Just add the missing closing braces
                    json_str += '}' * open_braces
            
            # Handle odd number of quotes (unclosed strings)
            if json_str.count('"') % 2 != 0:
                json_str += '"'
                
            # Close the array if not already closed
            if not json_str.endswith(']'):
                json_str += ']'
        
        return json_str
    
    def _repair_unterminated_strings(self, json_str: str) -> str:
        """Fix unterminated string patterns."""
        lines = json_str.split('\n')
        for i, line in enumerate(lines):
            # Look for unterminated strings in JSON object lines
            if '{"text"' in line and line.count('"') % 2 != 0:
                # If line ends without proper termination, try to fix it
                line = line.rstrip()
                if not line.endswith(('"}', '",', '"}')):
                    # Add closing quote if it looks like it's missing
                    if line.endswith(',') or line.endswith('}'):
                        line = line[:-1] + '"' + line[-1]
                    else:
                        line += '"'
                lines[i] = line
        
        return '\n'.join(lines)
    
    def _repair_final_structure(self, json_str: str) -> str:
        """Final structural repairs and cleanup."""
        # Handle odd number of quotes
        if json_str.count('"') % 2 != 0:
            # Find the last incomplete string and try to close it
            last_quote_pos = json_str.rfind('"')
            if last_quote_pos > 0:
                # Look for incomplete patterns after the last quote
                after_quote = json_str[last_quote_pos + 1:].strip()
                if after_quote and not after_quote.endswith(('"}', '",', '"}')):
                    # Truncated in middle of value, close it
                    json_str = json_str[:last_quote_pos + 1] + '"'
                    # Add closing brace/bracket if needed
                    if json_str.count('{') > json_str.count('}'):
                        json_str += '}'
                    if json_str.startswith('[') and not json_str.endswith(']'):
                        json_str += ']'
        
        # Handle unclosed objects (only if not already handled by truncated repair)
        # Skip if this is an array that was already processed
        if not (json_str.startswith('[') and json_str.endswith(']')):
            open_braces = json_str.count('{') - json_str.count('}')
            if open_braces > 0:
                json_str += '}' * open_braces
        
        # Final array closure - only add if genuinely missing and not already handled
        json_str = json_str.strip()  # Remove any whitespace that might confuse endswith
        if (json_str.startswith('[') and 
            not json_str.endswith(']') and 
            json_str.count('[') > json_str.count(']')):
            json_str += ']'
        
        # Final cleanup patterns
        json_str = re.sub(r',\s*,', r',', json_str)
        json_str = re.sub(r'\[\s*,', r'[', json_str)
        json_str = re.sub(r',\s*\]', r']', json_str)
        json_str = re.sub(r',\s*}', r'}', json_str)
        
        return json_str
    
    def filter_valid_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and deduplicate annotations while preserving legitimate overlapping spans."""
        seen_annotations = set()
        unique_annotations = []
        empty_count = 0
        
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
                
            text = (annotation.get('text', '') or '').strip()
            xbar_label = (annotation.get('xbar_label', '') or annotation.get('label', '') or 
                         annotation.get('xbar_class', '') or annotation.get('class', '') or '').strip()
            
            # Skip completely empty or malformed entries
            if not text or not xbar_label:
                empty_count += 1
                continue
                
            # Skip placeholder/garbage values
            if (text in ['', 'text', 'label', 'xbar_label', 'unknown', 'null'] or 
                xbar_label in ['', 'text', 'label', 'xbar_label', 'unknown', 'null']):
                empty_count += 1
                continue
            
            # Skip single-character punctuation that's not meaningful
            if len(text) == 1 and text in [',', ';', ':', '.', '"', "'", '(', ')', '[', ']', '{', '}']:
                continue
                
            # Skip obvious artifacts (repeated characters)
            if len(set(text)) == 1 and len(text) > 3:  # Like ",,,,," or "    "
                continue
            
            # For legitimate spans, allow overlaps and multiple labels
            # Only deduplicate exact duplicates (same text + same label)
            key = (text.lower(), xbar_label.lower())
            if key not in seen_annotations:
                seen_annotations.add(key)
                unique_annotations.append({'text': text, 'xbar_label': xbar_label})
        
        if empty_count > 10:
            logger.warning(f"Filtered out {empty_count} empty/garbage annotations from LLM output")
        
        logger.debug(f"Parsed {len(unique_annotations)} unique annotations from {len(annotations)} total found (filtered {empty_count} empty/garbage)")
        return unique_annotations
    
    def add_repair_pattern(self, pattern: str, replacement: str, description: str):
        """Add a new repair pattern to the parser."""
        self.repair_patterns.append((pattern, replacement, description))
    
    def remove_repair_pattern(self, description: str):
        """Remove a repair pattern by description."""
        self.repair_patterns = [p for p in self.repair_patterns if p[2] != description]


# Create a global instance for convenience
default_parser = XBarJsonParser()

# Convenience functions for backward compatibility
def parse_json_response(response: str) -> List[Dict[str, Any]]:
    """Parse JSON response using the default parser."""
    return default_parser.parse_json_response(response)

def attempt_json_repair(json_str: str) -> str:
    """Attempt JSON repair using the default parser."""
    return default_parser.attempt_json_repair(json_str)

def filter_valid_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter valid annotations using the default parser."""
    return default_parser.filter_valid_annotations(annotations)
