from pathlib import Path

from rich.console import Console
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

c = Console()

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