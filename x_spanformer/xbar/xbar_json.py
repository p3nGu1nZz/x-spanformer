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
        
        # JSON extraction patterns
        patterns = [
            r'```json\s*(\[.*?\])\s*```',        # JSON array in code block
            r'```json\s*(\{.*?\})\s*```',        # JSON object in code block
            r'```\s*(\[.*?\])\s*```',            # Array in generic code block  
            r'```\s*(\{.*?\})\s*```',            # Object in generic code block
            r'(\[.*?\])',                        # Simple array pattern
            r'(\{[^{}]*"text"[^{}]*"xbar_label"[^{}]*\})',  # Simple object pattern with required fields
        ]
        
        for pattern in patterns:
            for match in re.findall(pattern, response_stripped, re.DOTALL):
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
                    return self.filter_valid_annotations(annotations)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed, skipping sequence: {e}")
                    continue
        
        logger.warning("No valid JSON found in response, skipping sequence")
        return []
    
    def filter_valid_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and deduplicate annotations with greedy label selection."""
        from collections import Counter
        
        # First pass: collect all valid annotations
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
        
        # Second pass: implement greedy deduplication for same text with different labels
        # Count label frequency for greedy selection
        label_counts = Counter(ann['xbar_label'] for ann in valid_annotations)
        
        # Group by text (case-insensitive)
        text_groups = {}
        for i, annotation in enumerate(valid_annotations):
            text_key = annotation['text'].lower()
            if text_key not in text_groups:
                text_groups[text_key] = []
            text_groups[text_key].append((annotation, i))  # Store annotation with original index
        
        # Apply greedy selection for each text group
        unique_annotations = []
        for text_key, annotations_with_indices in text_groups.items():
            if len(annotations_with_indices) == 1:
                # Only one annotation for this text, keep it
                unique_annotations.append(annotations_with_indices[0][0])
            else:
                # Multiple labels for same text - apply greedy selection
                # Sort by: 1) label frequency (descending), 2) original order (ascending)
                def greedy_sort_key(ann_with_index):
                    annotation, original_index = ann_with_index
                    label_freq = label_counts[annotation['xbar_label']]
                    return (-label_freq, original_index)  # Negative for descending frequency
                
                sorted_annotations = sorted(annotations_with_indices, key=greedy_sort_key)
                winner = sorted_annotations[0][0]  # Take the annotation, not the index
                unique_annotations.append(winner)
                
                if len(annotations_with_indices) > 1:
                    logger.debug(f"Greedy selection for '{winner['text']}': kept '{winner['xbar_label']}' (freq: {label_counts[winner['xbar_label']]}), removed {len(annotations_with_indices) - 1} alternatives")
        
        logger.debug(f"Filtered to {len(unique_annotations)} unique valid annotations")
        return unique_annotations


# Create a global instance for convenience
default_parser = XBarJsonParser()

# Convenience functions for backward compatibility
def parse_json_response(response: str) -> List[Dict[str, Any]]:
    """Parse JSON response using the default parser."""
    return default_parser.parse_json_response(response)

def filter_valid_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter valid annotations using the default parser."""
    return default_parser.filter_valid_annotations(annotations)
