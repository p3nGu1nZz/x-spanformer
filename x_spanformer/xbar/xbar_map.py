"""
XBar classifier mapping for comprehensive span annotation.

Provides comprehensive X-bar classifier definitions for all domains
(natural language, code, mixed) with detailed descriptions for LLM agents.
"""

from typing import Dict, List, Optional
from enum import Enum


class DomainType(Enum):
    """Content domain types."""
    NATURAL = "natural"
    CODE = "code"  
    MIXED = "mixed"


class XBarClassifierMap:
    """Comprehensive X-bar classifier definitions for all domains."""
    
    # Natural language classifiers
    NATURAL_WORD_LEVEL = {
        "noun": "Identify all nouns including proper nouns, common nouns, and collective nouns",
        "verb": "Identify all verbs including action verbs, linking verbs, and auxiliary verbs", 
        "adjective": "Identify all adjectives including descriptive, comparative, and superlative forms",
        "adverb": "Identify all adverbs modifying verbs, adjectives, or other adverbs",
        "determiner": "Identify determiners including articles (the, a, an), demonstratives (this, that), and quantifiers",
        "preposition": "Identify prepositions indicating relationships of time, place, or manner",
        "pronoun": "Identify all pronouns including personal, possessive, demonstrative, and relative pronouns",
        "conjunction": "Identify coordinating and subordinating conjunctions",
        "punctuation": "Identify sentence-level punctuation marks with syntactic significance"
    }
    
    NATURAL_PHRASE_LEVEL = {
        "noun_phrase": "Identify complete noun phrases including determiners, modifiers, and head nouns",
        "verb_phrase": "Identify complete verb phrases including auxiliary verbs, main verbs, and complements",
        "adjective_phrase": "Identify adjective phrases with modifiers and complements",
        "adverb_phrase": "Identify adverb phrases with intensifiers and modifiers",
        "prepositional_phrase": "Identify prepositional phrases with their noun phrase objects"
    }
    
    NATURAL_CLAUSE_LEVEL = {
        "main_clause": "Identify main/independent clauses that can stand alone as sentences",
        "subordinate_clause": "Identify dependent/subordinate clauses including relative and adverbial clauses",
        "relative_clause": "Identify relative clauses modifying noun phrases"
    }
    
    NATURAL_SENTENCE_LEVEL = {
        "simple_sentence": "Identify complete simple sentences with single main clauses",
        "compound_sentence": "Identify compound sentences with multiple coordinated main clauses",
        "complex_sentence": "Identify complex sentences with main clause and subordinate clauses"
    }
    
    # Code classifiers
    CODE_WORD_LEVEL = {
        "keyword": "Identify programming language keywords (if, for, class, def, return, etc.)",
        "identifier": "Identify variable names, function names, class names, and other user-defined identifiers",
        "operator": "Identify all operators including arithmetic (+, -, *, /), logical (&&, ||), and comparison (==, !=)",
        "literal": "Identify string literals, numeric literals, boolean literals, and null values",
        "delimiter": "Identify delimiters including parentheses, brackets, braces, semicolons, and commas",
        "type_name": "Identify built-in and user-defined type names",
        "comment": "Identify single-line and multi-line comments"
    }
    
    CODE_PHRASE_LEVEL = {
        "expression": "Identify mathematical, logical, and assignment expressions",
        "function_call": "Identify function calls with their argument lists",
        "assignment": "Identify variable assignment statements and expressions",
        "parameter_list": "Identify function parameter lists in definitions",
        "argument_list": "Identify function call argument lists"
    }
    
    CODE_STATEMENT_LEVEL = {
        "if_statement": "Identify conditional statements including if, elif, and else branches",
        "loop_statement": "Identify for loops, while loops, and other iteration constructs",
        "function_definition": "Identify complete function definitions including signatures and bodies",
        "class_definition": "Identify complete class definitions including inheritance and methods",
        "import_statement": "Identify import and include statements",
        "return_statement": "Identify return statements in functions"
    }
    
    # Mixed domain classifiers
    MIXED_CONTENT = {
        "inline_code": "Identify inline code snippets within natural language text (e.g., `variable` in markdown)",
        "code_block": "Identify code blocks or examples within documentation or comments",
        "natural_instruction": "Identify natural language instructions or descriptions about code",
        "documentation_comment": "Identify structured documentation comments (docstrings, javadoc, etc.)",
        "api_reference": "Identify references to APIs, functions, or classes within natural language",
        "error_message": "Identify error messages or exception text within code or logs"
    }
    
    # X-bar theory roles (universal across domains)
    XBAR_ROLES = {
        "head": "Identify the head element that determines the category of the phrase",
        "specifier": "Identify specifiers that appear at the left edge of phrases (determiners, subjects)",
        "modifier": "Identify modifiers that provide additional information (adjectives, adverbs)",
        "complement": "Identify complements required by heads to complete their meaning",
        "adjunct": "Identify adjuncts that provide optional additional information"
    }
    
    @classmethod
    def get_classifiers_for_domain(cls, domain: DomainType) -> Dict[str, str]:
        """
        Get all applicable classifiers for a specific domain.
        
        Args:
            domain: Domain type (natural, code, or mixed)
            
        Returns:
            Dictionary mapping classifier names to descriptions
        """
        classifiers = {}
        
        if domain == DomainType.NATURAL:
            classifiers.update(cls.NATURAL_WORD_LEVEL)
            classifiers.update(cls.NATURAL_PHRASE_LEVEL)
            classifiers.update(cls.NATURAL_CLAUSE_LEVEL)
            classifiers.update(cls.NATURAL_SENTENCE_LEVEL)
            
        elif domain == DomainType.CODE:
            classifiers.update(cls.CODE_WORD_LEVEL)
            classifiers.update(cls.CODE_PHRASE_LEVEL)
            classifiers.update(cls.CODE_STATEMENT_LEVEL)
            
        elif domain == DomainType.MIXED:
            # Mixed domain gets both natural and code classifiers plus mixed-specific ones
            classifiers.update(cls.NATURAL_WORD_LEVEL)
            classifiers.update(cls.NATURAL_PHRASE_LEVEL) 
            classifiers.update(cls.CODE_WORD_LEVEL)
            classifiers.update(cls.CODE_PHRASE_LEVEL)
            classifiers.update(cls.MIXED_CONTENT)
        
        # All domains get X-bar roles
        classifiers.update(cls.XBAR_ROLES)
        
        return classifiers
    
    @classmethod
    def build_system_prompt(cls, domain: DomainType) -> str:
        """
        Build comprehensive system prompt for domain-specific annotation.
        
        Args:
            domain: Domain type for the system prompt
            
        Returns:
            Complete system prompt with all applicable classifiers
        """
        classifiers = cls.get_classifiers_for_domain(domain)
        
        base_prompt = """You are an expert linguistic annotator specializing in X-bar theory span identification.
Your task is to identify ALL applicable linguistic spans in the given text using the comprehensive classifier set below.

IMPORTANT GUIDELINES:
1. Use position-based indexing (character positions in the original text)
2. Use 0-based indexing with char_end INCLUSIVE (last character of span)  
3. Provide confidence scores between 0.0 and 1.0
4. Ensure spans don't inappropriately overlap unless hierarchically related
5. Include hierarchical levels: word, phrase, clause, sentence
6. Return comprehensive JSON with ALL identified spans

OUTPUT FORMAT:
Return a JSON array where each span object has:
{
  "text": "actual span text",
  "start": start_character_position,
  "end": end_character_position_inclusive,
  "label": "classifier_name",
  "confidence": confidence_score
}

"""
        
        # Add domain-specific classifier definitions
        if domain == DomainType.NATURAL:
            classifier_section = """
NATURAL LANGUAGE CLASSIFIERS:

Word Level:
"""
            for name, desc in cls.NATURAL_WORD_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
            
            classifier_section += "\nPhrase Level:\n"
            for name, desc in cls.NATURAL_PHRASE_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
                
            classifier_section += "\nClause Level:\n"
            for name, desc in cls.NATURAL_CLAUSE_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
                
            classifier_section += "\nSentence Level:\n"
            for name, desc in cls.NATURAL_SENTENCE_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
        
        elif domain == DomainType.CODE:
            classifier_section = """
CODE CLASSIFIERS:

Word Level:
"""
            for name, desc in cls.CODE_WORD_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
            
            classifier_section += "\nPhrase Level:\n"
            for name, desc in cls.CODE_PHRASE_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
                
            classifier_section += "\nStatement Level:\n"
            for name, desc in cls.CODE_STATEMENT_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
        
        elif domain == DomainType.MIXED:
            classifier_section = """
MIXED DOMAIN CLASSIFIERS:

Natural Language Elements:
"""
            for name, desc in cls.NATURAL_WORD_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
            
            classifier_section += "\nCode Elements:\n"
            for name, desc in cls.CODE_WORD_LEVEL.items():
                classifier_section += f"- {name}: {desc}\n"
                
            classifier_section += "\nMixed Content:\n"
            for name, desc in cls.MIXED_CONTENT.items():
                classifier_section += f"- {name}: {desc}\n"
        
        # Add X-bar roles for all domains
        classifier_section += "\nX-bar Theory Roles:\n"
        for name, desc in cls.XBAR_ROLES.items():
            classifier_section += f"- {name}: {desc}\n"
        
        return base_prompt + classifier_section
    
    @classmethod
    def get_classifier_names(cls, domain: DomainType) -> List[str]:
        """
        Get list of classifier names for a domain.
        
        Args:
            domain: Domain type
            
        Returns:
            List of classifier names
        """
        return list(cls.get_classifiers_for_domain(domain).keys())
    
    @classmethod
    def validate_classifier(cls, classifier_name: str, domain: DomainType) -> bool:
        """
        Validate that a classifier is applicable for a domain.
        
        Args:
            classifier_name: Name of classifier to validate
            domain: Domain type
            
        Returns:
            True if classifier is valid for domain
        """
        valid_classifiers = cls.get_classifiers_for_domain(domain)
        return classifier_name in valid_classifiers
