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
    c.print(f"[bold green]📄 Rendering:[/] [white]{template_name}.j2[/] with keys: {', '.join(kwargs.keys()) or '—'}")
    return tmpl.render(**kwargs)

def get_system_prompt(template_name: str = "selfcrit_system", **kwargs) -> str:
    c.print(f"[bold cyan]🧠 Using system prompt:[/] [white]{template_name}.j2[/]")
    return render_prompt(template_name, **kwargs)