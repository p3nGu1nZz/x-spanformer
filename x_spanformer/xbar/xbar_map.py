"""
XBar label mapping for comprehensive span annotation.

Provides unified X-bar label definitions for all domains
(natural language, code, mixed) with descriptions for LLM agents.
"""

from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Content domain types."""
    NATURAL = "natural"
    CODE = "code"  
    MIXED = "mixed"


class XBarLabelMap:
    """Unified X-bar label definitions for all domains."""
    
    # Natural language labels
    NATURAL_LABELS = {
        "noun": "Individual nouns: people, places, things, concepts (e.g., 'transformer', 'attention', 'model', 'layer')",
        "verb": "Individual verbs: actions, states, processes (e.g., 'computes', 'processes', 'encodes', 'learns')", 
        "adjective": "Individual adjectives: descriptive words, properties (e.g., 'neural', 'deep', 'efficient', 'complex')",
        "adverb": "Individual adverbs: manner, time, degree modifiers (e.g., 'efficiently', 'quickly', 'automatically')",
        "determiner": "Determiners: articles, quantifiers, possessives (e.g., 'the', 'a', 'each', 'this', 'our')",
        "preposition": "Prepositions: spatial, temporal, logical relations (e.g., 'in', 'through', 'during', 'via')",
        "pronoun": "Pronouns: personal, demonstrative, relative (e.g., 'it', 'they', 'which', 'that')",
        "conjunction": "Coordinating and subordinating conjunctions (e.g., 'and', 'but', 'because', 'while')",
        "punctuation": "Sentence-level punctuation with syntactic significance (e.g., '.', ',', ';', ':', '(', ')')",
        "noun_phrase": "Complete noun phrases with modifiers (e.g., 'the attention mechanism', 'multi-head self-attention')",
        "verb_phrase": "Complete verb phrases with auxiliaries and complements (e.g., 'computes attention weights', 'has been trained')",
        "adjective_phrase": "Adjective phrases with modifiers (e.g., 'computationally efficient', 'very deep')",
        "adverb_phrase": "Adverb phrases with intensifiers (e.g., 'quite efficiently', 'much more quickly')",
        "prepositional_phrase": "Prepositional phrases with objects (e.g., 'in the transformer', 'through multiple layers')",
        "main_clause": "Independent clauses that express complete thoughts (e.g., 'The model processes sequences', 'Attention mechanisms enable parallelization')",
        "subordinate_clause": "Dependent clauses with subordinating conjunctions (e.g., 'because it allows parallel computation', 'when training deep networks')",
        "relative_clause": "Relative clauses modifying noun phrases (e.g., 'which computes attention weights', 'that processes the input')"
    }
    
    # Code labels
    CODE_LABELS = {
        "keyword": "Programming language keywords and reserved words (e.g., 'def', 'class', 'if', 'for', 'import', 'return')",
        "identifier": "Variable, function, class, and module names (e.g., 'attention_weights', 'forward', 'TransformerModel')",
        "operator": "Arithmetic, logical, comparison, and assignment operators (e.g., '+', '==', '&&', '=', '->')",
        "literal": "String, numeric, boolean literals and constants (e.g., '\"hello\"', '0.1', 'True', 'None')",
        "delimiter": "Structural delimiters and separators (e.g., '(', ')', '[', ']', '{', '}', ';', ',')",
        "type_name": "Built-in and user-defined type names (e.g., 'int', 'str', 'List', 'torch.Tensor')",
        "comment": "Single-line and multi-line comments (e.g., '# This computes attention', '/* Multi-line comment */')",
        "expression": "Mathematical and logical expressions (e.g., 'x + y * 2', 'hidden_dim > 0', 'torch.matmul(q, k)')",
        "function_call": "Function invocations with arguments (e.g., 'torch.nn.Linear(512, 256)', 'model.forward(x)')",
        "assignment": "Variable assignments and parameter bindings (e.g., 'x = torch.zeros()', 'hidden_size=512')",
        "parameter_list": "Function parameter definitions (e.g., '(self, x, mask=None)', '(input_dim: int, output_dim: int)')",
        "argument_list": "Function call arguments (e.g., '(x, attention_mask)', '(hidden_states, key_padding_mask=mask)')",
        "if_statement": "Conditional statements and branches (e.g., 'if mask is not None:', 'elif hidden_dim > 0:')",
        "loop_statement": "Iteration constructs (e.g., 'for layer in self.layers:', 'while epoch < max_epochs:')",
        "function_definition": "Complete function definitions (e.g., 'def forward(self, x):', 'async def train_model():')",
        "class_definition": "Complete class definitions (e.g., 'class TransformerEncoder(nn.Module):', 'class Config:')",
        "import_statement": "Import and include statements (e.g., 'import torch', 'from transformers import BertModel')",
        "return_statement": "Return statements in functions (e.g., 'return output', 'return self.layer_norm(hidden_states)')"
    }
    
    # Mixed domain labels
    MIXED_LABELS = {
        "inline_code": "Inline code snippets within natural language (e.g., '`torch.nn.Module`', '`self.attention()`')",
        "code_block": "Multi-line code blocks or examples (e.g., '```python\\nmodel = TransformerModel()\\n```')",
        "natural_instruction": "Natural language instructions about code (e.g., 'Initialize the model with default parameters')",
        "documentation_comment": "Structured documentation and docstrings (e.g., '\"\"\"Computes multi-head attention\"\"\"')",
        "api_reference": "References to APIs, classes, or functions (e.g., 'torch.nn.TransformerEncoder', 'the forward() method')",
        "error_message": "Error messages and exception text (e.g., 'RuntimeError: Expected tensor on cuda:0', 'ValueError: Invalid input shape')"
    }
    
    # Abbreviation mappings for efficiency
    ABBREVIATION_MAP = {
        # Natural language word-level
        "n": "noun",
        "v": "verb", 
        "adj": "adjective",
        "adv": "adverb",
        "det": "determiner",
        "prep": "preposition",
        "pron": "pronoun",
        "conj": "conjunction",
        "punct": "punctuation",
        
        # Natural language phrase-level
        "np": "noun_phrase",
        "vp": "verb_phrase",
        "adjp": "adjective_phrase",
        "advp": "adverb_phrase",
        "pp": "prepositional_phrase",
        
        # Natural language clause-level
        "mc": "main_clause",
        "sc": "subordinate_clause",
        "rc": "relative_clause",
        
        # Code word-level
        "kw": "keyword",
        "id": "identifier",
        "op": "operator",
        "lit": "literal",
        "delim": "delimiter",
        "type": "type_name",
        "comm": "comment",
        
        # Code phrase-level
        "expr": "expression",
        "fcall": "function_call",
        "assign": "assignment",
        "params": "parameter_list",
        "args": "argument_list",
        
        # Code clause-level
        "if": "if_statement",
        "loop": "loop_statement",
        "fdef": "function_definition",
        "cdef": "class_definition",
        "imp": "import_statement",
        "ret": "return_statement",
        
        # Mixed domain labels
        "url": "url",
        "email": "email_address",
        "num": "number",
        "date": "date_time",
        "ref": "reference",
        "quote": "quoted_text",
        "code": "code_block",
        "list": "list_item",
        "head": "heading",
        "para": "paragraph"
    }
    
    @classmethod
    def normalize_xbar_class(cls, xbar_class: str) -> str:
        """
        Normalize XBar class label using abbreviation mapping.
        
        Args:
            xbar_class: Input label (can be abbreviation or full name)
            
        Returns:
            Normalized full label name
        """
        if not xbar_class or not xbar_class.strip():
            return "unknown"
            
        normalized = xbar_class.strip().lower()
        
        # Check if it's an abbreviation
        if normalized in cls.ABBREVIATION_MAP:
            return cls.ABBREVIATION_MAP[normalized]
        
        # Replace spaces with underscores and lowercase
        normalized = normalized.replace(" ", "_")
        
        # Return normalized form
        return normalized
    
    @classmethod
    def get_labels_for_domain(cls, domain: DomainType) -> Dict[str, str]:
        """
        Get all applicable labels for a specific domain.
        
        Args:
            domain: Domain type (natural, code, or mixed)
            
        Returns:
            Dictionary mapping label names to descriptions
        """
        labels = {}
        
        if domain == DomainType.NATURAL:
            labels.update(cls.NATURAL_LABELS)
            
        elif domain == DomainType.CODE:
            labels.update(cls.CODE_LABELS)
            
        elif domain == DomainType.MIXED:
            # Mixed domain gets both natural and code labels plus mixed-specific ones
            labels.update(cls.NATURAL_LABELS)
            labels.update(cls.CODE_LABELS)
            labels.update(cls.MIXED_LABELS)
        
        return labels
    
    @classmethod
    def get_label_names(cls, domain: DomainType) -> List[str]:
        """
        Get list of label names for a domain.
        
        Args:
            domain: Domain type
            
        Returns:
            List of label names
        """
        return list(cls.get_labels_for_domain(domain).keys())
    
    @classmethod
    def validate_label(cls, label_name: str, domain: DomainType) -> bool:
        """
        Validate that a label is applicable for a domain.
        
        Args:
            label_name: Name of label to validate
            domain: Domain type
            
        Returns:
            True if label is valid for domain
        """
        valid_labels = cls.get_labels_for_domain(domain)
        return label_name in valid_labels
    
    @classmethod
    def normalize_label(cls, label: str) -> str:
        """
        Normalize X-bar class label to standard format using abbreviation mappings.
        
        Args:
            label: Raw X-bar class from LLM
            
        Returns:
            Normalized X-bar class
        """
        if not label:
            return "unknown"
            
        # Remove extra whitespace and convert to standard case
        normalized = label.strip()
        
        # Try exact match in abbreviation map (case insensitive)
        for key, value in cls.ABBREVIATION_MAP.items():
            if normalized.lower() == key.lower():
                return value
        
        # Return original if no mapping found
        return normalized
    
    @classmethod
    def get_hierarchical_level(cls, label: str) -> Optional[str]:
        """
        Determine hierarchical level (word_level, phrase_level, clause_level) for a label.
        
        Args:
            label: X-bar label to classify
            
        Returns:
            Hierarchical level string or None if unknown
        """
        if not label or not label.strip():
            return None
        
        # Normalize the label first
        normalized_label = label.lower().strip()
        
        # Word-level labels (terminals)
        word_level_patterns = {
            # Natural language word-level
            'noun', 'verb', 'adjective', 'adverb', 'determiner', 'preposition', 
            'pronoun', 'conjunction', 'punctuation',
            # Code word-level
            'keyword', 'identifier', 'operator', 'literal', 'delimiter', 'type_name', 'comment',
            # Additional patterns from pipeline output
            'proper_noun', 'proper noun', 'parenthesis', 'colon', 'prefix', 'numeral',
            # New labels from warnings
            'auxiliary', 'number', 'numerator', 'possessive pronoun', 'possessor', 'none'
        }
        
        # Phrase-level labels (intermediate projections)
        phrase_level_patterns = {
            # Natural language phrase-level
            'noun_phrase', 'verb_phrase', 'adjective_phrase', 'adverb_phrase', 'prepositional_phrase',
            # Code phrase-level
            'expression', 'function_call', 'assignment', 'parameter_list', 'argument_list',
            # Mixed domain phrase-level
            'inline_code', 'code_block',
            # Additional patterns from pipeline output
            'code_expression'
        }
        
        # Clause-level labels (maximal projections)
        clause_level_patterns = {
            # Natural language clause-level
            'main_clause', 'subordinate_clause', 'relative_clause',
            # Code clause-level
            'if_statement', 'loop_statement', 'function_definition', 'class_definition', 
            'import_statement', 'return_statement',
            # Mixed domain clause-level
            'documentation_comment',
            # Additional patterns from pipeline output
            'code_statement', 'code statement'
        }
        
        # Handle multi-label cases (e.g., "noun, punctuation")
        if ',' in normalized_label:
            # For multi-label, take the first valid label
            parts = [part.strip() for part in normalized_label.split(',')]
            for part in parts:
                level = cls.get_hierarchical_level(part)
                if level:
                    return level
            # If no valid parts found, default to word level for multi-labels
            return "word_level"
        
        # Check exact matches
        if normalized_label in word_level_patterns:
            return "word_level"
        elif normalized_label in phrase_level_patterns:
            return "phrase_level"
        elif normalized_label in clause_level_patterns:
            return "clause_level"
        
        # Check for pattern matches (substring matching for flexible labeling)
        # Word-level patterns
        if any(pattern in normalized_label for pattern in ['noun', 'verb', 'adj', 'adv', 'punct', 'paren', 'colon']):
            return "word_level"
        
        # Phrase-level patterns  
        if any(pattern in normalized_label for pattern in ['phrase', 'expression', 'call', 'assign', 'list']):
            return "phrase_level"
            
        # Clause-level patterns
        if any(pattern in normalized_label for pattern in ['clause', 'statement', 'definition', 'import', 'return']):
            return "clause_level"
        
        # Default fallback - if it contains "code" and no other indicators, assume phrase level
        if 'code' in normalized_label:
            return "phrase_level"
        
        # Unknown label
        return None
    
    @classmethod
    def get_label_mapping_suggestions(cls, invalid_label: str) -> Optional[str]:
        """
        Get suggestions for mapping invalid labels to valid ones.
        
        Args:
            invalid_label: Invalid label to map
            
        Returns:
            Valid label suggestion or None if no mapping found
        """
        if not invalid_label or not invalid_label.strip():
            return None
        
        normalized = invalid_label.strip().lower()
        
        # Direct mappings for common invalid labels
        mapping_rules = {
            # Proper nouns -> noun
            "proper noun": "noun",
            "proper_noun": "noun",
            
            # Code statement variants -> appropriate code labels
            "code_statement": "expression",  # Most code statements are expressions
            "code statement": "expression",
            "code_expression": "expression",
            
            # Punctuation variants
            "parenthesis": "punctuation",
            "colon": "punctuation",
            
            # Numbers
            "numeral": "literal",
            
            # Prefixes
            "prefix": "identifier",
            
            # Multi-label cases - take the first valid component
            "noun, punctuation": "noun",  # Prioritize content words over punctuation
            "noun,punctuation": "noun",  # Handle no-space variant
            
            # New labels from warnings
            "auxiliary": "verb",  # Auxiliary verbs are still verbs
            "auxiliary verb": "verb",
            "number": "literal",  # Numbers are literals
            "numeral": "literal",  # Already mapped but adding for clarity
            "numerator": "literal",  # Mathematical terms
            "possessive pronoun": "pronoun",
            "possessor": "noun",  # Possessive relationships
            "none": "literal",  # None/null values
            
            # Complex multi-label cases
            "adjective, noun, conjunction, parenthesis": "adjective",  # Take first meaningful label
        }
        
        # Check direct mappings first
        if normalized in mapping_rules:
            return mapping_rules[normalized]
        
        # Handle comma-separated labels (take first valid one)
        if ',' in normalized:
            parts = [part.strip() for part in normalized.split(',')]
            for part in parts:
                # Check if this part is a valid label
                all_labels = set()
                all_labels.update(cls.NATURAL_LABELS.keys())
                all_labels.update(cls.CODE_LABELS.keys())
                all_labels.update(cls.MIXED_LABELS.keys())
                
                if part in all_labels:
                    return part
                    
                # Try to map this part
                mapped = cls.get_label_mapping_suggestions(part)
                if mapped:
                    return mapped
        
        # Pattern-based mappings
        if 'noun' in normalized:
            return "noun"
        elif 'verb' in normalized:
            return "verb"
        elif 'code' in normalized and 'statement' in normalized:
            return "expression"
        elif 'code' in normalized and 'expression' in normalized:
            return "expression"
        elif any(punct in normalized for punct in ['punctuation', 'parenthesis', 'colon', 'bracket']):
            return "punctuation"
        elif any(id_pattern in normalized for id_pattern in ['identifier', 'name', 'prefix']):
            return "identifier"
        elif any(lit_pattern in normalized for lit_pattern in ['literal', 'number', 'numeral', 'string']):
            return "literal"
        
        return None
    
    @classmethod
    def is_valid_word_span(cls, text: str) -> bool:
        """
        Validate word-level spans based on content patterns.
        
        Args:
            text: Text content of the span
            
        Returns:
            True if the span is valid for word-level annotation
        """
        if not text or not text.strip():
            return False
            
        text = text.strip()
        
        # Rule 1: No spaces in word-level spans
        if ' ' in text:
            return False
        
        # Rule 2: Check for mixed character types (but allow pure types)
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        has_special = any(not c.isalnum() for c in text)
        
        # Pure numbers are valid
        if text.isdigit():
            return True
            
        # Pure letters are valid (words)
        if text.isalpha():
            return True
            
        # Single character spans are valid (numbers, letters, punctuation)
        if len(text) == 1:
            return True
            
        # Variable names with underscores are valid (letters + underscores only)
        if has_letters and not has_digits and '_' in text:
            # Check if it's only letters and underscores
            if all(c.isalpha() or c == '_' for c in text):
                return True
        
        # Rule 3: Invalid mixed patterns
        # Letters mixed with numbers (except valid identifiers)
        if has_letters and has_digits:
            # Allow valid programming identifiers (letters, digits, underscores)
            # but they must start with letter or underscore
            if text.replace('_', '').replace('-', '').isalnum():
                # Valid identifier pattern
                if text[0].isalpha() or text[0] == '_':
                    return True
            return False
            
        # Letters mixed with special characters (except underscores in identifiers)
        if has_letters and has_special:
            # Allow words ending with colon (like "words:")
            if text.endswith(':') and text[:-1].isalpha():
                return True
            # Allow words with periods (abbreviations like "Dr.", "etc.", "U.S.")
            if '.' in text:
                # Check if it's a valid abbreviation or word with periods
                parts = text.split('.')
                if all(part.isalpha() or part == '' for part in parts):
                    return True
            # Only allow underscores, hyphens, and periods in identifiers/words
            allowed_special = set(['_', '-', '.'])
            special_chars = set(c for c in text if not c.isalnum())
            if not special_chars.issubset(allowed_special):
                return False
                
        # Numbers mixed with special characters (except decimal points, negatives, percentages)
        if has_digits and has_special:
            # Allow decimal numbers and negative numbers (including multiple decimal places)
            if text.replace('.', '').replace('-', '').isdigit():
                # Valid number format (can have multiple decimal points for things like version numbers)
                return True
            # Allow percentages (number followed by %)
            if text.endswith('%') and text[:-1].replace('.', '').isdigit():
                return True
            # Allow single-character expressions like (t), |s|, [83]
            if len(text) <= 4:
                # Parenthetical expressions: (t), (83)
                if text.startswith('(') and text.endswith(')') and len(text) >= 3:
                    return True
                # Bracketed expressions: [83], [t]
                if text.startswith('[') and text.endswith(']') and len(text) >= 3:
                    return True
                # Pipe expressions: |s|, |83|
                if text.startswith('|') and text.endswith('|') and len(text) >= 3:
                    return True
            return False
        
        # Pure special characters are valid (punctuation)
        if has_special and not has_letters and not has_digits:
            return True
        
        return True  # Default to valid for other cases
    
    @classmethod
    def clean_and_validate_labels(cls, annotations: List[Dict]) -> tuple[List[Dict], Dict[str, int]]:
        """
        Clean annotations by removing or mapping invalid labels.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Cleaned list of annotations with valid labels
        """
        # Get all valid labels
        all_valid_labels = set()
        all_valid_labels.update(cls.NATURAL_LABELS.keys())
        all_valid_labels.update(cls.CODE_LABELS.keys())
        all_valid_labels.update(cls.MIXED_LABELS.keys())
        
        cleaned_annotations = []
        mapping_stats = {"mapped": 0, "removed": 0, "valid": 0, "invalid_word_spans": 0}
        
        for ann in annotations:
            label = ann.get('xbar_label', '').strip()
            text = ann.get('text', '').strip()
            
            # First check if this is a word-level span that should be filtered
            hierarchical_level = cls.get_hierarchical_level(label)
            if hierarchical_level == "word_level" and not cls.is_valid_word_span(text):
                mapping_stats["invalid_word_spans"] += 1
                continue  # Skip this annotation
            
            if label in all_valid_labels:
                # Label is valid, keep as-is
                cleaned_annotations.append(ann)
                mapping_stats["valid"] += 1
            else:
                # Try to map invalid label
                mapped_label = cls.get_label_mapping_suggestions(label)
                if mapped_label and mapped_label in all_valid_labels:
                    # Check word-level validation for mapped labels too
                    mapped_level = cls.get_hierarchical_level(mapped_label)
                    if mapped_level == "word_level" and not cls.is_valid_word_span(text):
                        mapping_stats["invalid_word_spans"] += 1
                        continue  # Skip this annotation
                        
                    # Update the annotation with mapped label
                    ann_copy = ann.copy()
                    ann_copy['xbar_label'] = mapped_label
                    ann_copy['original_label'] = label  # Keep track of original
                    cleaned_annotations.append(ann_copy)
                    mapping_stats["mapped"] += 1
                else:
                    # Cannot map, remove the annotation
                    mapping_stats["removed"] += 1
        
        return cleaned_annotations, mapping_stats
    

