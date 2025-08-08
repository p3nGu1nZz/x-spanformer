#!/usr/bin/env python3
"""
X-bar JSON Parser Module

Simple JSON parser for LLM-generated annotation responses.
Skips sequences that cannot be parsed as valid JSON.
"""
import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class XBarJsonParser:
    """
    Simple JSON parser for X-bar annotation responses.
    
    Parses valid JSON or skips malformed sequences.
    No repair functionality - sequences with malformed JSON will be skipped.
    """
    
    def __init__(self):
        """Initialize the JSON parser."""
        pass
    
    def parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response - skip if malformed."""
        response_stripped = response.strip()
        if not response_stripped:
            return []
        
        # Pre-process response to fix common LLM JSON generation errors
        response_cleaned = self._clean_malformed_json(response_stripped)
        
        # JSON extraction patterns - try multiple approaches
        patterns = [
            r'```json\s*(\[.*?\])\s*```',        # JSON array in code block
            r'```json\s*(\{.*?\})\s*```',        # JSON object in code block
            r'```\s*(\[.*?\])\s*```',            # Array in generic code block  
            r'```\s*(\{.*?\})\s*```',            # Object in generic code block
            r'(\[.*?\])',                        # Simple array pattern
            r'(\{[^{}]*"text"[^{}]*"xbar_label"[^{}]*\})',  # Simple object pattern with required fields
        ]
        
        # Try standard JSON parsing first
        for pattern in patterns:
            for match in re.findall(pattern, response_cleaned, re.DOTALL):
                try:
                    # Try direct parsing - no repair
                    parsed_data = json.loads(match)
                    if isinstance(parsed_data, list):
                        annotations = parsed_data
                    elif isinstance(parsed_data, dict):
                        annotations = [parsed_data]
                    else:
                        continue
                        
                    logger.debug(f"Successfully parsed {len(annotations)} annotations")
                    filtered_annotations = self.filter_valid_annotations(annotations)
                    # Remove duplicates at JSON parse level before passing to annotator
                    deduplicated_annotations = self._remove_duplicates(filtered_annotations)
                    logger.debug(f"Removed {len(filtered_annotations) - len(deduplicated_annotations)} duplicates at JSON parse level")
                    return deduplicated_annotations
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed for match '{match[:100]}...': {e}")
                    continue
        
        # If standard JSON parsing fails, try regex-based extraction
        # This handles cases where quotes inside text break JSON structure
        logger.debug("Standard JSON parsing failed, trying regex extraction")
        annotations = self._extract_annotations_with_regex(response_stripped)
        if annotations:
            logger.debug(f"Regex extraction found {len(annotations)} annotations")
            filtered_annotations = self.filter_valid_annotations(annotations)
            deduplicated_annotations = self._remove_duplicates(filtered_annotations)
            logger.debug(f"Removed {len(filtered_annotations) - len(deduplicated_annotations)} duplicates at JSON parse level")
            return deduplicated_annotations
        
        logger.warning("No valid JSON found in response, skipping sequence")
        return []
    
    def _extract_annotations_with_regex(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract annotations using regex patterns when JSON parsing fails.
        
        This handles cases where the LLM generates valid-looking JSON that's actually
        broken due to unescaped quotes or other issues. The key is to extract the
        text content exactly as the LLM intended, preserving quotes and special characters.
        """
        annotations = []
        
        # Pattern to extract complete JSON-like objects, being flexible about quote handling
        # This captures the full object structure and extracts fields separately
        
        # Primary pattern: match complete objects with text and xbar_label
        object_patterns = [
            # Standard order: {"text":"...", "xbar_label":"..."}
            r'\{\s*"text"\s*:\s*"([^"]*(?:"[^"]*)*?)"\s*,\s*"xbar_label"\s*:\s*"([^"]+)"\s*\}',
            
            # Reversed order: {"xbar_label":"...", "text":"..."}  
            r'\{\s*"xbar_label"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]*(?:"[^"]*)*?)"\s*\}',
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                if len(match) == 2:
                    # Check if this is the reversed pattern
                    if '"xbar_label"' in pattern and pattern.index('"xbar_label"') < pattern.index('"text"'):
                        # Reversed order: (label, text)
                        label, text = match
                    else:
                        # Normal order: (text, label)
                        text, label = match
                    
                    # Clean up the extracted content
                    text = text.strip()
                    label = label.strip()
                    
                    if text and label:
                        annotations.append({
                            'text': text,
                            'xbar_label': label
                        })
        
        return annotations
    
    def _remove_duplicates(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate annotations while preserving order."""
        seen = set()
        unique_annotations = []
        for ann in annotations:
            key = (ann['text'], ann['xbar_label'])
            if key not in seen:
                seen.add(key)
                unique_annotations.append(ann)
        return unique_annotations
    
    def _clean_malformed_json(self, response: str) -> str:
        """
        Clean common LLM-generated JSON malformations.
        
        Focus on simple, reliable fixes that don't over-process the text.
        The goal is to make the JSON parseable while preserving the original text content.
        """
        cleaned = response
        
        # Remove any obvious formatting issues
        cleaned = cleaned.strip()
        
        # Fix unquoted property names - add quotes around common property names
        # Match property names that are not already quoted
        cleaned = re.sub(r'\b(text|xbar_label|label|class|xbar_class)\s*:', r'"\1":', cleaned)
        
        # Fix common trailing comma issues
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # Fix missing commas between objects
        cleaned = re.sub(r'}\s*{', '},{', cleaned)
        
        return cleaned
    
    def filter_valid_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter annotations for basic validity - no deduplication at this level."""
        
        # Only do basic filtering - deduplication happens at sequence level
        valid_annotations = []
        
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
                
            # Extract text and label, handling different key names and None values
            text = annotation.get('text', '')
            xbar_label = (
                annotation.get('xbar_label') or 
                annotation.get('label') or 
                annotation.get('xbar_class') or 
                annotation.get('class')
            )
            
            # Handle None values from null in JSON - filter out completely
            if text is None:
                text = ''
            if xbar_label is None:
                continue  # Skip null labels entirely
            
            # Convert to string and strip
            text = str(text).strip()
            xbar_label = str(xbar_label).strip()
            
            # Skip empty entries and unknown labels
            if not text or not xbar_label or xbar_label.lower() == 'unknown':
                continue
                
            # Skip placeholder/garbage values
            if (text in ['text', 'label', 'xbar_label', 'unknown'] or 
                xbar_label in ['text', 'label', 'xbar_label', 'unknown']):
                continue
            
            # Skip obvious artifacts (repeated characters)
            if len(set(text)) == 1 and len(text) > 3:
                continue
                
            # Skip repetitive punctuation patterns (LLM hallucination)
            if len(text) > 1 and all(c in '.,;:!?-_()[]{}' for c in text):
                continue
            
            # Keep all single-character punctuation with proper labels
            # Important punctuation like (), {}, [], etc. should be preserved
            # Only skip if it's clearly mislabeled (e.g., punctuation labeled as 'noun')
            if (len(text) == 1 and text in [',', ';', ':', '.', '"', "'"] and 
                xbar_label not in ['operator', 'punctuation', 'delimiter', 'bracket', 'conjunction']):
                continue
                
            # Skip very short non-meaningful text
            if len(text) == 1 and text.isspace():
                continue
            
            valid_annotations.append({'text': text, 'xbar_label': xbar_label})
        
        logger.debug(f"Filtered to {len(valid_annotations)} valid annotations (no deduplication at turn level)")
        return valid_annotations
# Create a global instance for convenience
default_parser = XBarJsonParser()

# Convenience functions for backward compatibility
def parse_json_response(response: str) -> List[Dict[str, Any]]:
    """Parse JSON response using the default parser."""
    return default_parser.parse_json_response(response)

def filter_valid_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter valid annotations using the default parser."""
    return default_parser.filter_valid_annotations(annotations)
