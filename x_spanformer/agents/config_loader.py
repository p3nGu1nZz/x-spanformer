from pathlib import Path
from functools import lru_cache

import yaml
from rich.console import Console
from rich.table import Table

c = Console()

# Global cache for config to avoid loading multiple times
_cached_config = None
_config_loaded = False

@lru_cache(maxsize=1)
def load_selfcrit_config(name: str = "selfcrit.yaml", quiet: bool = False) -> dict:
	"""
	Load the agent configuration file and render a summary table.
	Uses caching to avoid loading the same config multiple times.
	
	Args:
		config_name: The name of the config file to load.
		quiet: If True, suppresses printing the config table.
	"""
	global _config_loaded
	
	p = Path(__file__).parent / "config" / name
	
	# Only show loading message on first load
	if not _config_loaded:
		c.rule(f"[bold cyan]Loading SelfCrit Agent Config")
		c.print(f"[green]✔ Found:[/] [cyan]{p}[/cyan] — loading...")
	
	if not p.exists():
		c.print(f"[red]❌ Missing config file:[/] {p}")
		raise FileNotFoundError(f"Missing selfcrit config: {p}")

	with p.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	# Only show success message and table on first load
	if not _config_loaded:
		c.print(f"[green]✔ Successfully parsed:[/] [white]{name}[/white]")
		_config_loaded = True

		if not quiet:
			tbl = Table(show_header=True, header_style="bold magenta")
			tbl.add_column("Field", style="dim")
			tbl.add_column("Value")

			tbl.add_row("Agent Type", cfg.get("agent_type", "—"))
			tbl.add_row("Model", cfg["model"]["name"])
			tbl.add_row("Judge Passes", str(cfg["judge"]["passes"]))
			tbl.add_row("Judge Model", cfg["judge"]["model_name"])
			tbl.add_row("Judge Temperature", str(cfg["judge"]["temperature"]))
			tbl.add_row("Judge Threshold", str(cfg["judge"]["threshold"]))
			tbl.add_row("Judge Discard Threshold", str(cfg["judge"]["discard_threshold"]))
			tbl.add_row("Critique Threshold", str(cfg["critique"]["threshold"]))
			tbl.add_row("Critique Discard Threshold", str(cfg["critique"]["discard_threshold"]))
			tbl.add_row("Max Retries", str(cfg["critique"]["max_retries"]))
			tbl.add_row("Max Turns", str(cfg["dialogue"]["max_turns"]))
			tbl.add_row("Improver Model", cfg["improver"]["model_name"])
			tbl.add_row("Regex Filters", str(len(cfg.get("regex_filters", []))))
			tbl.add_row("Templates", ", ".join(cfg["templates"].keys()))

			c.print(tbl)
	
	return cfg