from typing import Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


# Code domain labels based on pretraining_schema.md
CodeLabel = Literal[
    "keyword", "identifier", "operator", "delimiter", "literal", "type", "specifier",
    "space", "newline", "comment", "preprocessor", "call", "block", "control"
]

CodeRole = Literal[
    "variable name", "assignment", "loop body", "function body", "function", "macro invocation",
    "indent", "separator", "line separator", "documentation", "directive", "conditional",
    "iteration", "declaration", "expression", "statement", "parameter", "argument",
    "return value", "class name", "method name", "field name", "constant", "variable declaration",
    "loop type", "primitive", "loop variable", "assignment operator", "numeric value", 
    "statement terminator", "float", "constant name", "string literal", "boolean literal"
]

# Natural language labels based on pretraining_schema.md  
NaturalLabel = Literal[
    "noun", "verb", "adjective", "adverb", "preposition", "conjunction", "determiner",
    "pronoun", "interjection", "punctuation", "space", "newline", "number", "symbol"
]

NaturalRole = Literal[
    "subject", "object", "complement", "agent", "theme", "predicate", "action", "state",
    "modifier", "attributive", "predicative", "manner", "time", "place", "degree",
    "adverbial modifier", "relation", "complement introducer", "coordination", "subordination",
    "definite", "indefinite", "quantifier", "demonstrative", "possessive", "personal",
    "relative", "interrogative", "reflexive", "exclamation", "greeting", "separator",
    "line break", "cardinal", "ordinal", "special character", "formatting", "noun specifier",
    "instruction", "tense anchor", "auxiliary", "attribute", "negation", "temporal",
    "article", "terminator", "clause boundary", "soft break", "phrase gap", 
    "paragraph divider", "token"
]

# Hybrid/mixed domain roles
HybridRole = Literal[
    "code_delimiter", "natural_language_context", "embedded_syntax", "transition_marker",
    "inline_code", "documentation_context", "instruction", "example", "reference",
    "inline code open", "inline code close"
]

# X-bar structural roles
XBarRole = Literal[
    "specifier", "complement", "adjunct", "head", "modifier", "determiner", "nucleus"
]


class XPSpan(BaseModel):
    span: Tuple[int, int] = Field(..., description="Start and end token indices for the X-bar phrase")
    category: Literal["X⁰", "X′", "XP"] = Field(..., description="Constituent category (head, intermediate, phrase)")
    role: Optional[XBarRole] = Field(None, description="Constituent role relative to the head")
    label: Optional[str] = Field(None, description="Optional syntactic label (e.g., 'NP', 'VP', 'PP')")
    text: Optional[str] = Field(None, description="Human-readable span content (optional, for debugging)")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "span": [2, 5],
                "category": "XP",
                "role": "specifier",
                "label": "NP",
                "text": "the quick fox"
            }
        }
    )