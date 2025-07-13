from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

c = Console()

def load_judge_config(name: str = "judge.yaml", quiet: bool = False) -> dict:
	"""
	Load the agent configuration file and render a summary table.
	
	Args:
		config_name: The name of the config file to load.
		quiet: If True, suppresses printing the config table.
	"""
	p = Path(__file__).parent / "config" / name
	c.rule(f"[bold cyan]Loading Judge Agent Config")

	if not p.exists():
		c.print(f"[red]❌ Missing config file:[/] {p}")
		raise FileNotFoundError(f"Missing judge config: {p}")

	c.print(f"[green]✔ Found:[/] [cyan]{p}[/cyan] — loading...")

	with p.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	c.print(f"[green]✔ Successfully parsed:[/] [white]{name}[/white]")

	if not quiet:
		tbl = Table(show_header=True, header_style="bold magenta")
		tbl.add_column("Field", style="dim")
		tbl.add_column("Value")

		tbl.add_row("Agent Type", cfg.get("agent_type", "—"))
		tbl.add_row("Model", cfg["model"]["name"])
		tbl.add_row("Judge Count", str(cfg["judge"]["judges"]))
		tbl.add_row("Judge Model", cfg["judge"]["model_name"])
		tbl.add_row("Judge Temperature", str(cfg["judge"]["temperature"]))
		tbl.add_row("Judge Threshold", str(cfg["judge"]["threshold"]))
		tbl.add_row("Judge Max Retries", str(cfg["judge"]["max_retries"]))
		tbl.add_row("Max Turns", str(cfg["dialogue"]["max_turns"]))
		tbl.add_row("Regex Filters", str(len(cfg.get("regex_filters", []))))
		tbl.add_row("Templates", ", ".join(cfg["templates"].keys()))

		c.print(tbl)
	
	return cfg