"""
XBar label mapping for comprehensive span annotation.

Provides unified X-bar label definitions for all domains
(natural language, code, mixed) with descriptions for LLM agents.
"""

from typing import Dict, List, Optional
from enum import Enum


class DomainType(Enum):
    """Content domain types."""
    NATURAL = "natural"
    CODE = "code"  
    MIXED = "mixed"


class XBarLabelMap:
    """Unified X-bar label definitions for all domains."""
    
    # Natural language labels
    NATURAL_LABELS = {
        "noun": "Individual nouns including proper nouns, common nouns, and collective nouns",
        "verb": "Individual verbs including action verbs, linking verbs, and auxiliary verbs", 
        "adjective": "Individual adjectives including descriptive, comparative, and superlative forms",
        "adverb": "Individual adverbs modifying verbs, adjectives, or other adverbs",
        "determiner": "Determiners including articles (the, a, an), demonstratives (this, that), and quantifiers",
        "preposition": "Prepositions indicating relationships of time, place, or manner",
        "pronoun": "Pronouns including personal, possessive, demonstrative, and relative pronouns",
        "conjunction": "Coordinating and subordinating conjunctions",
        "punctuation": "Sentence-level punctuation marks with syntactic significance",
        "noun_phrase": "Complete noun phrases including determiners, modifiers, and head nouns",
        "verb_phrase": "Complete verb phrases including auxiliary verbs, main verbs, and complements",
        "adjective_phrase": "Adjective phrases with modifiers and complements",
        "adverb_phrase": "Adverb phrases with intensifiers and modifiers",
        "prepositional_phrase": "Prepositional phrases with their noun phrase objects",
        "main_clause": "Main/independent clauses that can stand alone as sentences",
        "subordinate_clause": "Dependent/subordinate clauses including relative and adverbial clauses",
        "relative_clause": "Relative clauses modifying noun phrases"
    }
    
    # Code labels
    CODE_LABELS = {
        "keyword": "Programming language keywords (if, for, class, def, return, etc.)",
        "identifier": "Variable names, function names, class names, and other user-defined identifiers",
        "operator": "All operators including arithmetic (+, -, *, /), logical (&&, ||), and comparison (==, !=)",
        "literal": "String literals, numeric literals, boolean literals, and null values",
        "delimiter": "Delimiters including parentheses, brackets, braces, semicolons, and commas",
        "type_name": "Built-in and user-defined type names",
        "comment": "Single-line and multi-line comments",
        "expression": "Mathematical, logical, and assignment expressions",
        "function_call": "Function calls with their argument lists",
        "assignment": "Variable assignment statements and expressions",
        "parameter_list": "Function parameter lists in definitions",
        "argument_list": "Function call argument lists",
        "if_statement": "Conditional statements including if, elif, and else branches",
        "loop_statement": "For loops, while loops, and other iteration constructs",
        "function_definition": "Complete function definitions including signatures and bodies",
        "class_definition": "Complete class definitions including inheritance and methods",
        "import_statement": "Import and include statements",
        "return_statement": "Return statements in functions"
    }
    
    # Mixed domain labels
    MIXED_LABELS = {
        "inline_code": "Inline code snippets within natural language text (e.g., `variable` in markdown)",
        "code_block": "Code blocks or examples within documentation or comments",
        "natural_instruction": "Natural language instructions or descriptions about code",
        "documentation_comment": "Structured documentation comments (docstrings, javadoc, etc.)",
        "api_reference": "References to APIs, functions, or classes within natural language",
        "error_message": "Error messages or exception text within code or logs"
    }
    
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
    

