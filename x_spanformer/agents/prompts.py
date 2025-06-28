from pathlib import Path

from rich.console import Console
from jinja2 import Environment, FileSystemLoader

c = Console()

env = Environment(
	loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
	autoescape=False,
	trim_blocks=True,
	lstrip_blocks=True,
)

def render_prompt(name: str, **kwargs) -> str:
	tmpl = env.get_template(f"{name}.j2")
	c.print(f"[bold green]📄 Rendering:[/] [white]{name}.j2[/] with keys: {', '.join(kwargs.keys()) or '—'}")
	return tmpl.render(**kwargs)

def get_system_prompt(name: str = "selfcrit_system", **kwargs) -> str:
	c.print(f"[bold cyan]🧠 Using system prompt:[/] [white]{name}.j2[/]")
	return render_prompt(name, **kwargs)