"""
Position mapping utilities for character-level to position-level conversion.

Handles the mapping between LLM agent character-level span annotations
and the position-wise contextual embeddings used by X-Spanformer.
"""

from typing import List, Tuple, Dict, Optional
import re
from dataclasses import dataclass


@dataclass
class CharacterSpan:
    """Character-level span from LLM agent."""
    start_char: int
    end_char: int
    xbar_class: str
    confidence: float = 1.0
    text: Optional[str] = None


@dataclass 
class PositionSpan:
    """Position-level span aligned with embeddings."""
    start_pos: int
    end_pos: int
    xbar_class: str
    confidence: float = 1.0
    positions: Optional[List[int]] = None


class PositionMapper:
    """
    Maps character-level annotations to position-level indices.
    
    Handles the conversion from LLM agent character-based spans to 
    position-based spans that align with contextual embeddings.
    
    IMPORTANT: In X-Spanformer's architecture, position embeddings H[t] represent
    contextual character representations (512-dim vectors), NOT span embeddings.
    Spans are detected through boundary prediction, not by averaging position embeddings.
    
    Each H[t] contains:
    - Character-level information at position t
    - Contextual information from the entire sequence  
    - Compositional patterns from multi-scale dilated convolutions
    
    Span representation works through boundary detection:
    - Start boundary head: W_start @ H[t] -> P(start boundary at position t)
    - End boundary head: W_end @ H[t] -> P(end boundary at position t)
    - Training targets: Binary labels at start/end positions
    """
    
    def __init__(self, text: str):
        """
        Initialize mapper with text sequence.
        
        Args:
            text: The raw Unicode text sequence
        """
        self.text = text
        self.char_to_pos = self._build_char_to_pos_mapping()
        self.pos_to_char = self._build_pos_to_char_mapping()
    
    def _build_char_to_pos_mapping(self) -> Dict[int, int]:
        """
        Build mapping from character indices to position indices.
        
        For tokenizer-free architecture, we use character-level positions
        where each Unicode character corresponds to one embedding position.
        
        Returns:
            Dictionary mapping character index -> position index
        """
        return {i: i for i in range(len(self.text))}
    
    def _build_pos_to_char_mapping(self) -> Dict[int, int]:
        """
        Build mapping from position indices to character indices.
        
        Returns:
            Dictionary mapping position index -> character index
        """
        return {i: i for i in range(len(self.text))}
    
    def char_span_to_position_span(self, char_span: CharacterSpan) -> PositionSpan:
        """
        Convert character-level span to position-level span.
        
        Args:
            char_span: Character-level span from LLM agent
            
        Returns:
            Position-level span for embedding alignment
        """
        # For tokenizer-free architecture, character indices map directly to positions
        start_pos = self.char_to_pos.get(char_span.start_char, char_span.start_char)
        end_pos = self.char_to_pos.get(char_span.end_char - 1, char_span.end_char - 1) + 1
        
        # Ensure bounds are valid
        start_pos = max(0, min(start_pos, len(self.text)))
        end_pos = max(start_pos, min(end_pos, len(self.text)))
        
        return PositionSpan(
            start_pos=start_pos,
            end_pos=end_pos,
            xbar_class=char_span.xbar_class,
            confidence=char_span.confidence,
            positions=list(range(start_pos, end_pos))
        )
    
    def position_span_to_char_span(self, pos_span: PositionSpan) -> CharacterSpan:
        """
        Convert position-level span to character-level span.
        
        Args:
            pos_span: Position-level span from embeddings
            
        Returns:
            Character-level span for text analysis
        """
        start_char = self.pos_to_char.get(pos_span.start_pos, pos_span.start_pos)
        end_char = self.pos_to_char.get(pos_span.end_pos - 1, pos_span.end_pos - 1) + 1
        
        # Ensure bounds are valid
        start_char = max(0, min(start_char, len(self.text)))
        end_char = max(start_char, min(end_char, len(self.text)))
        
        return CharacterSpan(
            start_char=start_char,
            end_char=end_char,
            xbar_class=pos_span.xbar_class,
            confidence=pos_span.confidence,
            text=self.text[start_char:end_char] if start_char < end_char else ""
        )
    
    def batch_char_to_position(self, char_spans: List[CharacterSpan]) -> List[PositionSpan]:
        """
        Convert multiple character spans to position spans.
        
        Args:
            char_spans: List of character-level spans
            
        Returns:
            List of position-level spans
        """
        return [self.char_span_to_position_span(span) for span in char_spans]
    
    def batch_position_to_char(self, pos_spans: List[PositionSpan]) -> List[CharacterSpan]:
        """
        Convert multiple position spans to character spans.
        
        Args:
            pos_spans: List of position-level spans
            
        Returns:
            List of character-level spans
        """
        return [self.position_span_to_char_span(span) for span in pos_spans]
    
    def validate_position_spans(self, pos_spans: List[PositionSpan]) -> List[Tuple[PositionSpan, List[str]]]:
        """
        Validate position spans for consistency and bounds.
        
        Args:
            pos_spans: List of position-level spans to validate
            
        Returns:
            List of (span, issues) tuples where issues is list of problem descriptions
        """
        results = []
        text_length = len(self.text)
        
        for span in pos_spans:
            issues = []
            
            # Check bounds
            if span.start_pos < 0:
                issues.append(f"Start position {span.start_pos} is negative")
            if span.end_pos > text_length:
                issues.append(f"End position {span.end_pos} exceeds text length {text_length}")
            if span.start_pos >= span.end_pos:
                issues.append(f"Start position {span.start_pos} >= end position {span.end_pos}")
            
            # Check confidence
            if not (0.0 <= span.confidence <= 1.0):
                issues.append(f"Confidence {span.confidence} not in range [0.0, 1.0]")
            
            # Check X-bar class format
            if not span.xbar_class or not isinstance(span.xbar_class, str):
                issues.append(f"Invalid X-bar class: {span.xbar_class}")
            
            results.append((span, issues))
        
        return results
    
    def get_text_length(self) -> int:
        """Get the total length of the text sequence."""
        return len(self.text)
    
    def get_position_text(self, start_pos: int, end_pos: int) -> str:
        """
        Get text substring for a position range.
        
        Args:
            start_pos: Start position index
            end_pos: End position index (exclusive)
            
        Returns:
            Text substring for the position range
        """
        start_char = self.pos_to_char.get(start_pos, start_pos)
        end_char = self.pos_to_char.get(end_pos - 1, end_pos - 1) + 1 if end_pos > 0 else 0
        
        return self.text[start_char:end_char]


def parse_character_spans_from_agent_response(
    agent_response: str, 
    text: str
) -> List[CharacterSpan]:
    """
    Parse character-level spans from LLM agent response.
    
    Expects response format with spans like:
    "The quick brown fox" (0-18) -> NP [confidence: 0.88]
    
    Note: The response uses inclusive end positions (0-18 means characters 0 through 18),
    but Python slicing is exclusive, so we use text[0:19].
    
    Args:
        agent_response: Raw response from LLM agent
        text: Original text sequence
        
    Returns:
        List of parsed character spans
    """
    spans = []
    
    # Regex pattern to match span annotations
    # Pattern: "text" (start-end) -> XBar [confidence: score]
    pattern = r'"([^"]*?)"\s*\((\d+)-(\d+)\)\s*->\s*(\w+)(?:\s*\[confidence:\s*([\d.]+)\])?'
    
    matches = re.finditer(pattern, agent_response)
    
    for match in matches:
        span_text = match.group(1)
        start_char = int(match.group(2))
        end_char_inclusive = int(match.group(3))  # This is inclusive in the response
        end_char = end_char_inclusive + 1  # Convert to exclusive for Python slicing
        xbar_class = match.group(4)
        confidence = float(match.group(5)) if match.group(5) else 1.0
        
        # Validate against original text
        if start_char >= 0 and end_char <= len(text):
            actual_text = text[start_char:end_char]
            if actual_text.strip() == span_text.strip():
                spans.append(CharacterSpan(
                    start_char=start_char,
                    end_char=end_char,  # Store as exclusive for consistency
                    xbar_class=xbar_class,
                    confidence=confidence,
                    text=span_text
                ))
    
    return spans
