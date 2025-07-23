"""
Shared text processing utilities for pipelines.

Contains common text splitting, concatenation, and length management functions
used across different pipeline implementations (pdf2jsonl, repo2jsonl, etc.)
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from x_spanformer.agents.rich_utils import console

logger = logging.getLogger(__name__)


def split_long_text(text: str, max_length: int = 512) -> List[str]:
    """
    Split text that exceeds max_length into smaller chunks, ensuring no chunk is longer than max_length.
    It prioritizes splitting at sentence boundaries, then word boundaries, and finally at the character level
    for very long, unbroken strings of text.
    
    Args:
        text: The text to split
        max_length: Maximum length per chunk (default: 512)
        
    Returns:
        List of text chunks, each <= max_length characters
    """
    if len(text) <= max_length:
        return [text]

    # Use a simple regex for sentence splitting to avoid heavy dependencies like spaCy
    # This regex looks for sentence-ending punctuation followed by a space or the end of the string.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    
    for sent in sentences:
        if not sent.strip():
            continue
            
        if len(sent) <= max_length:
            chunks.append(sent)
        else:
            # The sentence itself is too long, so we need to split it further.
            # First, try splitting by words.
            words = sent.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_length:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
            
            if current_chunk:
                chunks.append(current_chunk)

    # Final check: If any chunk is still too long (e.g., a very long word or token),
    # we must split it at the character level.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            for i in range(0, len(chunk), max_length):
                final_chunks.append(chunk[i:i+max_length])
        else:
            final_chunks.append(chunk)
            
    # If, after all that, we have no chunks, it means the original text was probably just whitespace.
    # Return an empty list in that case.
    if not final_chunks and not text.strip():
        return []
        
    return final_chunks


def concatenate_small_segments(spans: List[str], source_mapping: List[str], 
                             min_length: int = 64, max_length: int = 512) -> Tuple[List[str], List[str]]:
    """
    Iteratively concatenate small segments within the same document until reaching acceptable length.
    Respects max_length boundary - never exceeds it during concatenation.
    
    Args:
        spans: List of text segments  
        source_mapping: List of source files corresponding to each segment
        min_length: Minimum segment length to keep standalone (default: 64)
        max_length: Maximum length after concatenation (default: 512)
    
    Returns:
        Tuple of (concatenated_spans, updated_source_mapping)
    """
    if not spans:
        return spans, source_mapping
    
    concatenated_spans = []
    concatenated_sources = []
    concatenated_count = 0
    
    i = 0
    while i < len(spans):
        current_text = spans[i].strip()
        current_source = source_mapping[i]
        
        # Skip empty or whitespace-only segments
        if not current_text:
            i += 1
            continue
        
        # If segment is too small, try iterative concatenation
        if len(current_text) < min_length and i < len(spans) - 1:
            combined_text = current_text
            segments_used = 1
            j = i + 1
            
            # Iteratively add subsequent segments from same document
            while j < len(spans) and source_mapping[j] == current_source:
                next_segment = spans[j].strip()
                
                # Skip empty segments
                if not next_segment:
                    j += 1
                    continue
                
                # Don't concatenate with segments that are already long enough by themselves
                if len(next_segment) >= min_length:
                    break
                
                potential_combined = combined_text + " " + next_segment
                
                # RESPECT MAX LENGTH - don't exceed during concatenation
                if len(potential_combined) <= max_length:
                    combined_text = potential_combined
                    segments_used += 1
                    j += 1
                    
                    # Stop if we've reached a good length
                    if len(combined_text) >= min_length:
                        break
                else:
                    # Would exceed max_length, stop concatenation
                    break
            
            concatenated_spans.append(combined_text)
            concatenated_sources.append(current_source)
            
            if segments_used > 1:
                concatenated_count += segments_used - 1  # Count extra segments merged
            
            i = j  # Skip the segments we just concatenated
        else:
            # Segment is long enough or is last segment - keep as-is
            concatenated_spans.append(current_text)
            concatenated_sources.append(current_source)
            i += 1
    
    if concatenated_count > 0:
        console.print(f"[cyan]🔗 Concatenated {concatenated_count} small segments (min: {min_length} chars, max: {max_length} chars)[/cyan]")
    
    return concatenated_spans, concatenated_sources


def split_long_text_for_code(text: str, max_length: int = 512) -> List[str]:
    """
    Split text that exceeds max_length into smaller chunks, optimized for code content.
    Prioritizes splitting on newlines to preserve code structure.
    
    Args:
        text: The text to split (typically code)
        max_length: Maximum length per chunk (default: 512)
        
    Returns:
        List of text chunks, each <= max_length characters
    """
    if len(text) <= max_length:
        return [text]

    # Split on newlines first for code
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # If any chunk is still too long, split by characters
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            for i in range(0, len(chunk), max_length):
                final_chunks.append(chunk[i:i+max_length])
        else:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else [""]


def normalize_text_segments(spans: List[str], source_mapping: List[str], 
                          max_length: int = 512, min_length: int = 64,
                          content_type: str = "natural") -> Tuple[List[str], List[str]]:
    """
    Normalize text segments by applying appropriate splitting and concatenation.
    
    This is the main entry point that combines splitting and concatenation logic
    based on content type and length constraints.
    
    Args:
        spans: List of text segments
        source_mapping: List of source files corresponding to each segment
        max_length: Maximum length per segment (default: 512)
        min_length: Minimum length to avoid concatenation (default: 64)
        content_type: Type of content ("natural", "code", "mixed")
        
    Returns:
        Tuple of (normalized_spans, updated_source_mapping)
    """
    if not spans:
        return spans, source_mapping
    
    # Step 1: Split long texts
    expanded_spans = []
    expanded_source_mapping = []
    
    for text, source_file in zip(spans, source_mapping):
        if len(text) > max_length:
            if content_type == "code":
                text_chunks = split_long_text_for_code(text, max_length)
            else:
                text_chunks = split_long_text(text, max_length)
        else:
            text_chunks = [text]
        
        for chunk in text_chunks:
            if chunk.strip():  # Only add non-empty chunks
                expanded_spans.append(chunk)
                expanded_source_mapping.append(source_file)
    
    # Step 2: Concatenate small segments (only for natural text)
    if content_type == "natural" and min_length > 0:
        return concatenate_small_segments(expanded_spans, expanded_source_mapping, min_length, max_length)
    
    return expanded_spans, expanded_source_mapping
