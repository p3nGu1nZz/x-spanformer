from pathlib import Path
import logging
from typing import Optional

from rich.console import Console
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from x_spanformer.xbar.xbar_map import XBarClassifierMap, DomainType

c = Console()
logger = logging.getLogger(__name__)

env = Environment(
	loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
	autoescape=False,
	trim_blocks=True,
	lstrip_blocks=True,
)

def render_prompt(template_name: str, **kwargs) -> str:
    try:
        tmpl = env.get_template(f"{template_name}.j2")
    except TemplateNotFound:
        # Fallback to direct string template if file not found
        tmpl = env.from_string(template_name)
    c.print(f"[dim]Rendering template: {template_name}.j2[/dim]")
    return tmpl.render(**kwargs)

def get_system_prompt(template_name: str = "judge_system", **kwargs) -> str:
    c.print(f"[dim]Using system prompt: {template_name}.j2[/dim]")
    return render_prompt(template_name, **kwargs)

def render_span_annotator_system_prompt(domain_type: str = "natural", **kwargs) -> str:
    """
    Render system prompt for span annotator with domain-specific classifiers.
    
    Args:
        domain_type: Domain type (natural, code, mixed)
        **kwargs: Additional template variables
        
    Returns:
        Rendered system prompt
    """
    try:
        # Convert string to DomainType enum
        if isinstance(domain_type, str):
            domain_enum = DomainType(domain_type.lower())
        else:
            domain_enum = domain_type
    except ValueError:
        logger.warning(f"Invalid domain type: {domain_type}, defaulting to natural")
        domain_enum = DomainType.NATURAL
    
    # Organize classifiers by category for template
    template_vars = {
        "domain_type": domain_enum.value,
        "xbar_roles": XBarClassifierMap.XBAR_ROLES,
        **kwargs
    }
    
    if domain_enum == DomainType.NATURAL:
        template_vars.update({
            "levels": ["word", "phrase", "clause", "sentence"],
            "word_level": XBarClassifierMap.NATURAL_WORD_LEVEL,
            "phrase_level": XBarClassifierMap.NATURAL_PHRASE_LEVEL,
            "clause_level": XBarClassifierMap.NATURAL_CLAUSE_LEVEL,
            "sentence_level": XBarClassifierMap.NATURAL_SENTENCE_LEVEL,
        })
    elif domain_enum == DomainType.CODE:
        template_vars.update({
            "levels": ["word", "phrase", "statement"],
            "word_level": XBarClassifierMap.CODE_WORD_LEVEL,
            "phrase_level": XBarClassifierMap.CODE_PHRASE_LEVEL,
            "statement_level": XBarClassifierMap.CODE_STATEMENT_LEVEL,
        })
    elif domain_enum == DomainType.MIXED:
        template_vars.update({
            "levels": ["word", "phrase", "mixed content"],
            "natural_elements": XBarClassifierMap.NATURAL_WORD_LEVEL,
            "code_elements": XBarClassifierMap.CODE_WORD_LEVEL,
            "mixed_content": XBarClassifierMap.MIXED_CONTENT,
        })
    
    return render_prompt("span_annotator_system", **template_vars)

def render_span_annotation_request(
    text: str,
    domain_type: str = "natural",
    turn_number: int = 1,
    max_turns: int = 8,
    focus_area: Optional[str] = None,
    **kwargs
) -> str:
    """
    Render user request for span annotation.
    
    Args:
        text: Text to analyze
        domain_type: Domain type (natural, code, mixed)
        turn_number: Current turn number
        max_turns: Maximum number of turns
        focus_area: Optional focus area for follow-up turns
        **kwargs: Additional template variables
        
    Returns:
        Rendered user request
    """
    try:
        # Convert string to DomainType enum
        if isinstance(domain_type, str):
            domain_enum = DomainType(domain_type.lower())
        else:
            domain_enum = domain_type
    except ValueError:
        logger.warning(f"Invalid domain type: {domain_type}, defaulting to natural")
        domain_enum = DomainType.NATURAL
    
    # Get expected classifiers for this domain
    expected_classifiers = XBarClassifierMap.get_classifier_names(domain_enum)
    
    # Define hierarchical levels based on domain
    if domain_enum == DomainType.NATURAL:
        levels = ["word", "phrase", "clause", "sentence"]
    elif domain_enum == DomainType.CODE:
        levels = ["word", "phrase", "statement"]
    else:  # MIXED
        levels = ["word", "phrase", "mixed content"]
    
    template_vars = {
        "text": text,
        "domain_type": domain_enum.value,
        "turn_number": turn_number,
        "max_turns": max_turns,
        "focus_area": focus_area,
        "expected_classifiers": expected_classifiers,
        "levels": levels,
        **kwargs
    }
    
    return render_prompt("span_annotation_request", **template_vars)