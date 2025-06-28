import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import langid
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.selfcrit import judge_segment
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.pretrain_record import PretrainRecord

c = Console()

def echo_config(cfg: dict):
    c.rule("[bold cyan]Using SelfCrit Configuration")
    c.print(f"[white]• Model:[/] {cfg['model']['name']} @ T={cfg['model']['temperature']}")
    c.print(f"[white]• Voting:[/] {cfg['evaluation']['passes']} passes | Retry: {cfg['evaluation']['max_retries']}")
    c.print(f"[white]• Regex filters:[/] {len(cfg.get('regex_filters', []))}")
    c.print(f"[white]• Templates:[/] {', '.join(cfg['templates'].keys())}")

def manifest(p: Path):
    stem = p.stem
    m = p.parent / stem / f"{stem}.json"
    if m.exists():
        with m.open("r", encoding="utf-8") as f: d = json.load(f)
        return d.get("csv") or p.name, "pdf2seg (manifest v1)"
    return p.name, "unknown"

def rows(p: Path, col: str, w: int, cfg: dict) -> list[PretrainRecord]:
    with p.open("r", encoding="utf-8") as x: data = list(csv.DictReader(x))
    if not data: c.print(f"[red]⚠ No rows in {p.name}"); return []
    if col not in data[0]: c.print(f"[red]⚠ Missing '{col}' in {p.name}"); return []

    src, tool = manifest(p)
    spans = [r.get(col, "").strip() for r in data if r.get(col, "").strip()]
    if not spans: c.print(f"[red]⚠ No usable '{col}' values in {p.name}"); return []

    async def process():
        sem = asyncio.Semaphore(w)
        stats = Counter()
        recs, reasons = [], []

        async def score(idx: int, t: str):
            try:
                async with sem:
                    if cfg["logging"].get("log_queries", False):
                        c.log(f"[grey50]Judging text:[/] {t[:120]}…")
                    r = await judge_segment(t)
                    if cfg["logging"].get("log_responses", False):
                        c.log(f"[grey50]Judge response:[/] {r}")
                    return idx, r
            except Exception as e:
                c.log(f"[red]Judge error:[/] {str(e)[:100]}… for text: {t[:60]}…")
                return idx, {"score": 0.5, "status": "revise", "reason": "selfcrit error"}

        with Progress(SpinnerColumn(), TextColumn("[bold cyan]→"), BarColumn(bar_width=None),
                      transient=True, console=c) as pb:
            task_id = pb.add_task(f"[white]Evaluating {p.name}", total=len(spans))
            
            tasks = [score(i, t) for i, t in enumerate(spans)]

            for future in asyncio.as_completed(tasks):
                try:
                    idx, r = await future
                    text = spans[idx]
                    tag = r["status"]
                    reasons.append(r["reason"])
                    stats[tag] += 1
                    lang = langid.classify(text)[0]
                    recs.append(PretrainRecord(raw=text, meta=RecordMeta(
                        source_file=src, doc_language=lang, extracted_by=tool,
                        confidence=r.get("score"), tags=[tag] if tag != "keep" else [], notes=r.get("reason")
                    )))
                    if cfg["logging"].get("track_consensus", True) and r["reason"] == "unparseable":
                        c.log(f"[yellow]⚠ Unparseable output:[/] {text[:72]}…")
                except asyncio.CancelledError:
                    c.print("[yellow]Processing cancelled.[/]")
                    break
                finally:
                    pb.update(task_id, advance=1)

        return recs, stats, reasons

    try:
        recs, stats, reasons = asyncio.run(process())
        show_summary(p.name, stats, reasons)
        return recs
    except KeyboardInterrupt:
        c.print("\n[bold red]Interrupted by user. Exiting.[/]")
        return []

def show_summary(name: str, stats: Counter, reasons: list[str]):
    tbl = Table(title=f"📊 SelfCrit Summary – {name}", expand=True)
    tbl.add_column("Status", justify="center")
    tbl.add_column("Count", justify="right")
    for k in ("keep", "revise", "discard"):
        tbl.add_row(k, str(stats.get(k, 0)))
    c.print(tbl)

    top = Counter(reasons).most_common(5)
    if top:
        c.print("[blue]🔍 Top reasons returned:")
        for r, n in top:
            c.print(f"[cyan]•[/] [white]{r}[/] — [dim]{n}x")

def run(i: Path, o: Path, f: str, pretty: bool, n: str, w: int):
    cfg = load_selfcrit_config()
    echo_config(cfg)

    csvs = [i] if i.is_file() else sorted(i.glob("*.csv"))
    if not csvs:
        c.print(f"[red]⚠ No CSVs found in {i}[/]"); return

    if not i.is_file():
        s = "\n".join(f"[cyan]•[/] {x.name}" for x in csvs)
        c.print(Panel.fit(f"[bold magenta]📁 Found {len(csvs)} CSV files\n{s}", border_style="bright_magenta"))
    else:
        c.print(f"[bold magenta]📁 Processing single CSV file: [cyan]{i.name}[/]")

    allr = []
    for src in csvs:
        c.print(f"[white]→ Processing CSV: [cyan]{src.name}[/]")
        r = rows(src, f, w, cfg)
        if r:
            allr.extend(r)
            c.print(f"[green]✔ Successfully processed: [cyan]{src.name}[/]")
        else:
            c.print(f"[red]⚠ No records extracted from: [cyan]{src.name}[/]")

    if not allr:
        c.print(f"[red]⚠ No valid records found across all CSVs[/]")
        return

    total = len(allr)
    kept = sum(1 for r in allr if not r.meta.tags)
    ratio = round(kept / total * 100, 2)
    c.rule("[bold green]Final Summary")
    c.print(f"[white]Total records:[/] {total}")
    c.print(f"[green]Kept:[/] {kept} ([cyan]{ratio}%[/])")
    c.print(f"[yellow]Discarded:[/] {total - kept}")

    base = n.strip().removesuffix(".json").removesuffix(".jsonl") or "dataset"
    o.mkdir(parents=True, exist_ok=True)
    j1, j2 = o / f"{base}.jsonl", o / f"{base}.json"

    with j1.open("w", encoding="utf-8") as writer:
        for x in allr:
            writer.write(json.dumps(x.model_dump(), ensure_ascii=False) + "\n")
    c.print(f"[green]✔ Wrote[/] [white]{total}[/] entries → [cyan]{j1.name}[/]")

    if pretty:
        with j2.open("w", encoding="utf-8") as writer:
            json.dump([x.model_dump() for x in allr], writer, ensure_ascii=False, indent=2)
        c.print(f"[bold cyan]•[/] Pretty JSON → {j2.name}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-f", "--field", type=str, default="text")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("-n", "--name", type=str, default="dataset")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    run(a.input, a.output, a.field, a.pretty, a.name, a.workers)