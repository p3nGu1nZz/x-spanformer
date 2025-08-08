from pathlib import Path
import logging
from typing import Optional, List

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from x_spanformer.xbar.xbar_map import XBarLabelMap, DomainType

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
    logger.debug(f"Rendering template: {template_name}.j2")
    return tmpl.render(**kwargs)

def get_system_prompt(template_name: str = "judge_system", **kwargs) -> str:
    logger.debug(f"Using system prompt: {template_name}.j2")
    return render_prompt(template_name, **kwargs)