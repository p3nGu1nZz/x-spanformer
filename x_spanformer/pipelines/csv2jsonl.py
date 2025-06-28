import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import langid
from rich.console import Console

from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.selfcrit import judge_segment
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.pretrain_record import PretrainRecord

c = Console()

def echo_config(cfg: dict):
    c.print("[bold cyan]═══ SelfCrit Configuration ═══[/bold cyan]")
    c.print(f"[white]Model:[/white] [green]{cfg['model']['name']}[/green] @ T={cfg['model']['temperature']}")
    c.print(f"[white]Voting:[/white] [yellow]{cfg['evaluation']['passes']}[/yellow] passes | Retry: [yellow]{cfg['evaluation']['max_retries']}[/yellow]")
    c.print(f"[white]Regex filters:[/white] [blue]{len(cfg.get('regex_filters', []))}[/blue]")
    c.print(f"[white]Templates:[/white] [cyan]{', '.join(cfg['templates'].keys())}[/cyan]")
    c.print()

def manifest(p: Path):
    stem = p.stem
    m = p.parent / stem / f"{stem}.json"
    if m.exists():
        with m.open("r", encoding="utf-8") as f: 
            d = json.load(f)
        return d.get("csv") or p.name, "pdf2seg (manifest v1)"
    return p.name, "unknown"

def rows(p: Path, col: str, w: int, cfg: dict, save_interval: int = 10, output_path: Optional[Path] = None, base_name: str = "dataset") -> list[PretrainRecord]:
    with p.open("r", encoding="utf-8") as x: 
        data = list(csv.DictReader(x))
    if not data: 
        c.print(f"[red]⚠ No rows in {p.name}[/red]")
        return []
    if col not in data[0]: 
        c.print(f"[red]⚠ Missing '{col}' in {p.name}[/red]")
        return []

    src, tool = manifest(p)
    spans = [r.get(col, "").strip() for r in data if r.get(col, "").strip()]
    if not spans: 
        c.print(f"[red]⚠ No usable '{col}' values in {p.name}[/red]")
        return []

    c.print(f"[green]Processing {len(spans)} text segments from {p.name}[/green]")
    if save_interval > 0 and output_path:
        c.print(f"[cyan]Incremental saving enabled: every {save_interval} segments to {output_path}[/cyan]")
    c.print()

    def save_incremental(records: list[PretrainRecord]):
        """Save records incrementally to prevent data loss"""
        if not output_path or not records:
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_file = output_path / f"{base_name}.jsonl"
        
        with dataset_file.open("w", encoding="utf-8") as writer:
            for record in records:
                writer.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        
        c.print(f"[blue]💾 Saved {len(records)} records to {dataset_file.name}[/blue]")

    async def process():
        sem = asyncio.Semaphore(w)
        stats = Counter()
        recs, reasons = [], []

        async def score(idx: int, t: str):
            try:
                async with sem:
                    c.print(f"[bold blue]━━━ Processing segment {idx + 1}/{len(spans)} ━━━[/bold blue]")
                    c.print(f"[dim]Text ({len(t)} chars):[/dim] {t}")
                    c.print()
                    
                    r = await judge_segment(t)
                    
                    c.print(f"[bold green]Response:[/bold green]")
                    c.print(f"[white]{json.dumps(r, indent=2)}[/white]")
                    c.print()
                    
                    return idx, r
            except Exception as e:
                c.print(f"[red]Error processing segment {idx + 1}:[/red] {str(e)}")
                c.print(f"[dim]Text was:[/dim] {t[:100]}...")
                c.print()
                return idx, {"score": 0.5, "status": "revise", "reason": "processing error"}

        for i, text in enumerate(spans):
            try:
                idx, r = await score(i, text)
                tag = r["status"]
                reasons.append(r["reason"])
                stats[tag] += 1
                lang = langid.classify(text)[0]
                recs.append(PretrainRecord(raw=text, meta=RecordMeta(
                    source_file=src, doc_language=lang, extracted_by=tool,
                    confidence=r.get("score"), tags=[tag] if tag != "keep" else [], notes=r.get("reason")
                )))
                
                # Incremental save check
                if save_interval > 0 and output_path and (i + 1) % save_interval == 0:
                    save_incremental(recs)
                    
            except asyncio.CancelledError:
                c.print("[yellow]Processing cancelled.[/yellow]")
                # Save what we have so far
                if save_interval > 0 and output_path and recs:
                    save_incremental(recs)
                break
            except Exception as e:
                c.print(f"[red]Error in main loop for segment {i + 1}:[/red] {str(e)}")

        # Final incremental save if there are remaining records
        if save_interval > 0 and output_path and recs and len(recs) % save_interval != 0:
            save_incremental(recs)

        return recs, stats, reasons

    try:
        recs, stats, reasons = asyncio.run(process())
        show_summary(p.name, stats, reasons)
        return recs
    except KeyboardInterrupt:
        c.print("\n[bold red]Interrupted by user. Exiting.[/bold red]")
        return []

def show_summary(name: str, stats: Counter, reasons: list[str]):
    c.print(f"[bold cyan]═══ Summary for {name} ═══[/bold cyan]")
    for k in ("keep", "revise", "discard"):
        count = stats.get(k, 0)
        color = "green" if k == "keep" else "yellow" if k == "revise" else "red"
        c.print(f"[{color}]{k.capitalize()}:[/{color}] {count}")
    
    top = Counter(reasons).most_common(5)
    if top:
        c.print("\n[blue]Top reasons:[/blue]")
        for r, n in top:
            c.print(f"[cyan]•[/cyan] [white]{r}[/white] [dim]({n}x)[/dim]")
    c.print()

def run(i: Path, o: Path, f: str, pretty: bool, n: str, w: int, save_interval: int = 10):
    cfg = load_selfcrit_config()
    echo_config(cfg)

    csvs = [i] if i.is_file() else sorted(i.glob("*.csv"))
    if not csvs:
        c.print(f"[red]⚠ No CSVs found in {i}[/red]")
        return

    if not i.is_file():
        c.print(f"[bold magenta]Found {len(csvs)} CSV files:[/bold magenta]")
        for x in csvs:
            c.print(f"[cyan]• {x.name}[/cyan]")
    else:
        c.print(f"[bold magenta]Processing single CSV file: {i.name}[/bold magenta]")
    c.print()

    base = n.strip().removesuffix(".json").removesuffix(".jsonl") or "dataset"
    
    allr = []
    for src in csvs:
        c.print(f"[white]→ Processing CSV: [cyan]{src.name}[/cyan][/white]")
        r = rows(src, f, w, cfg, save_interval, o, base)
        if r:
            allr.extend(r)
            c.print(f"[green]✔ Successfully processed: {src.name}[/green]")
        else:
            c.print(f"[red]⚠ No records extracted from: {src.name}[/red]")
        c.print()

    if not allr:
        c.print(f"[red]⚠ No valid records found across all CSVs[/red]")
        return

    total = len(allr)
    kept = sum(1 for r in allr if not r.meta.tags)
    ratio = round(kept / total * 100, 2)
    
    c.print("[bold green]═══ Final Summary ═══[/bold green]")
    c.print(f"[white]Total records:[/white] {total}")
    c.print(f"[green]Kept:[/green] {kept} [cyan]({ratio}%)[/cyan]")
    c.print(f"[yellow]Discarded:[/yellow] {total - kept}")
    c.print()

    o.mkdir(parents=True, exist_ok=True)
    j1, j2 = o / f"{base}.jsonl", o / f"{base}.json"

    with j1.open("w", encoding="utf-8") as writer:
        for x in allr:
            writer.write(json.dumps(x.model_dump(), ensure_ascii=False) + "\n")
    c.print(f"[green]✔ Wrote {total} entries → {j1.name}[/green]")

    if pretty:
        with j2.open("w", encoding="utf-8") as writer:
            json.dump([x.model_dump() for x in allr], writer, ensure_ascii=False, indent=2)
        c.print(f"[cyan]• Pretty JSON → {j2.name}[/cyan]")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-f", "--field", type=str, default="text")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("-n", "--name", type=str, default="dataset")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=10, help="Save dataset incrementally after every N segments (0 to disable)")
    a = p.parse_args()
    run(a.input, a.output, a.field, a.pretty, a.name, a.workers, a.save_interval)