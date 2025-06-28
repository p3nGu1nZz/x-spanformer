from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

c = Console()

def load_selfcrit_config(name: str = "selfcrit.yaml") -> dict:
	p = Path(__file__).parent / "config" / name
	c.rule(f"[bold cyan]Loading SelfCrit Agent Config")

	if not p.exists():
		c.print(f"[red]❌ Missing config file:[/] {p}")
		raise FileNotFoundError(f"Missing selfcrit config: {p}")

	c.print(f"[green]✔ Found:[/] [cyan]{p}[/cyan] — loading...")

	with p.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	c.print(f"[green]✔ Successfully parsed:[/] [white]{name}[/white]")

	tbl = Table(show_header=True, header_style="bold magenta")
	tbl.add_column("Field", style="dim")
	tbl.add_column("Value")

	tbl.add_row("Agent", cfg.get("agent_name", "—"))
	tbl.add_row("Model", cfg["model"]["name"])
	tbl.add_row("Passes", str(cfg["evaluation"]["passes"]))
	tbl.add_row("Retries", str(cfg["evaluation"]["max_retries"]))
	tbl.add_row("Max Turns", str(cfg["dialogue"]["max_turns"]))
	tbl.add_row("Regex Filters", str(len(cfg.get("regex_filters", []))))
	tbl.add_row("Templates", ", ".join(cfg["templates"].keys()))

	c.print(tbl)
	return cfg