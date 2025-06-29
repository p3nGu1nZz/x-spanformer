import argparse
import asyncio
import csv
import hashlib
import json
import sys
import pandas as pd
import langid
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rich.console import Console

from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.session import JudgeSession, ImproveSession
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.pretrain_record import PretrainRecord

c = Console()

def hash_name(p: Path) -> str:
    """Generate a hash for a given path's name."""
    return hashlib.sha256(p.name.encode()).hexdigest()[:8]

def run_pdf2seg(pdf_file: Path, output_dir: Path, force_regenerate: bool = False) -> Optional[Path]:
    """Run pdf2seg on a PDF file to generate CSV output."""
    
    expected_csv_name = f"{hash_name(pdf_file)}.csv"
    csv_file = output_dir / expected_csv_name

    # If CSV already exists and we're not forcing regeneration, return it
    if not force_regenerate and csv_file.exists() and csv_file.stat().st_size > 0:
        c.print(f"[green]✔ Using existing CSV: {csv_file.name}[/green]")
        return csv_file

    try:
        import pdf2seg
        c.print(f"[yellow]Running pdf2seg on {pdf_file.name}...[/yellow]")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Only clean up existing files for this PDF if we're regenerating
        if force_regenerate:
            for existing_file in output_dir.glob(f"{hash_name(pdf_file)}.*"):
                try:
                    existing_file.unlink()
                except OSError:
                    pass
        
        nlp = pdf2seg.load("en_core_web_sm")
        segments = pdf2seg.extract(str(pdf_file), str(output_dir), nlp)

        # Look for any CSV files that were created after running pdf2seg
        created_csvs = list(output_dir.glob("*.csv"))
        
        if segments is not None:
            pdf2seg.save_csv(segments, str(csv_file), str(pdf_file), {})
            c.print(f"[green]✔ Generated CSV: {csv_file.name}[/green]")
        elif created_csvs:
            # If segments is None but CSVs exist, find the most recent one
            csv_file = max(created_csvs, key=lambda f: f.stat().st_mtime)
            # Rename it to our expected name if it's different
            if csv_file.name != expected_csv_name:
                new_csv_file = output_dir / expected_csv_name
                csv_file.rename(new_csv_file)
                csv_file = new_csv_file
            c.print(f"[green]✔ Found and renamed CSV: {csv_file.name}[/green]")
        else:
            c.print(f"[yellow]No segments extracted, creating minimal CSV for {pdf_file.name}[/yellow]")
            csv_file.write_text("text\n\"No extractable content\"")
        
        if not csv_file.exists():
            c.print(f"[yellow]Creating fallback CSV for {pdf_file.name}[/yellow]")
            csv_file.write_text("text\n\"sample text\"")
        
    except ImportError:
        c.print(f"[red]pdf2seg package not found. Please ensure pdf2seg is installed.[/red]")
        return None
    except Exception as e:
        error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
        c.print(f"[red]⚠ Error processing {pdf_file.name}: {error_msg}[/red]")
        # Check if a CSV was still created despite the error
        if csv_file.exists():
            c.print(f"[yellow]⚠ Error was thrown, but CSV file was found. Proceeding...[/yellow]")
            return csv_file
        return None
    finally:
        # Clean up temporary PNG files
        for png_file in output_dir.glob("*.png"):
            try:
                png_file.unlink()
            except OSError as e:
                c.print(f"[yellow]⚠ Could not remove temp file {png_file}: {e}[/yellow]")

    if csv_file.exists():
        c.print(f"[green]✔ CSV ready: {csv_file.name}[/green]")
        return csv_file
    else:
        c.print(f"[red]⚠ Expected CSV file not found: {csv_file.name}[/red]")
        return None

def manifest(p: Path):
    stem = p.stem
    m = p.parent / f"{stem}.json"
    if m.exists():
        with m.open("r", encoding="utf-8") as f: 
            d = json.load(f)
        return d.get("csv") or p.name, "pdf2seg (manifest v1)"
    return p.name, "unknown"

def process_all_csvs(csv_files: list[Path], col: str, w: int, cfg: dict, save_interval: int = 10, output_path: Optional[Path] = None, base_name: str = "dataset", pdf_mapping: Optional[dict[str, str]] = None) -> list[PretrainRecord]:
    if not csv_files:
        c.print(f"[red]⚠ No CSV files provided[/red]")
        return []
    
    if pdf_mapping is None:
        pdf_mapping = {}
    
    all_dfs = []
    source_files = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if col in df.columns:
                original_pdf_name = pdf_mapping.get(csv_file.name, csv_file.name)
                df['source_file'] = original_pdf_name
                all_dfs.append(df)
                source_files.append(original_pdf_name)
                c.print(f"[green]✔ Loaded {len(df)} rows from {csv_file.name} (original: {original_pdf_name})[/green]")
            else:
                c.print(f"[red]⚠ Missing '{col}' column in {csv_file.name}[/red]")
        except Exception as e:
            c.print(f"[red]⚠ Error reading {csv_file.name}: {e}[/red]")
    
    if not all_dfs:
        c.print(f"[red]⚠ No valid CSV files found[/red]")
        return []
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    spans = combined_df[col].dropna().astype(str).str.strip().tolist()
    source_mapping = combined_df['source_file'].tolist()
    
    if not spans:
        c.print(f"[red]⚠ No usable '{col}' values found across all CSV files[/red]")
        return []

    c.print(f"[green]Processing {len(spans)} text segments from {len(source_files)} CSV files[/green]")
    if save_interval > 0 and output_path:
        c.print(f"[cyan]Incremental saving enabled: every {save_interval} segments to {output_path}[/cyan]")
    c.print()

    def save_incremental(records: list[PretrainRecord], force: bool = False):
        """Save records incrementally to prevent data loss"""
        if not output_path or not records:
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_file = output_path / f"{base_name}.jsonl"
        
        # For the first save, check if we should start fresh or append
        mode = "a" if dataset_file.exists() else "w"
        
        with dataset_file.open(mode, encoding="utf-8") as writer:
            for record in records:
                writer.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        
        c.print(f"[blue]💾 Saved {len(records)} records to {dataset_file.name}[/blue]")

    async def process():
        sem = asyncio.Semaphore(w)
        stats = Counter()
        all_processed_recs = []
        recs, reasons = [], []
        processed_count = 0

        # Initialize sessions once, sharing the same config
        agent_config = load_selfcrit_config()
        judge_session = JudgeSession(config=agent_config, quiet=True)
        improve_session = ImproveSession(config=agent_config, quiet=True)

        async def score_and_improve(idx: int, t: str, source_file: str):
            try:
                async with sem:
                    c.print(f"[bold blue]━━━ Processing segment {idx + 1}/{len(spans)} ━━━[/bold blue]")
                    c.print(f"[dim]Source: {source_file} | Text ({len(t)} chars):[/dim] {t}")
                    c.print()
                    
                    # Use JudgeSession to evaluate the text
                    r = await judge_session.evaluate(t)
                    
                    c.print(f"[bold green]Initial Response:[/bold green]")
                    c.print(f"[white]{json.dumps(r, indent=2)}[/white]")
                    c.print()
                    
                    improved_text = None
                    content_type = None
                    final_text = t
                    improvement_iterations = 0
                    max_improvements = 3
                    
                    current_text = t
                    current_score = r.get("score", 0)
                    
                    while (current_score < 0.7 and 
                           r.get("status") in ["revise", "discard"] and 
                           improvement_iterations < max_improvements):
                        
                        c.print(f"[yellow]💡 Attempting improvement #{improvement_iterations + 1}/{max_improvements}...[/yellow]")
                        temp_improved, temp_type = await improve_session.improve(current_text)
                        
                        if temp_improved and temp_improved != current_text:
                            c.print(f"[yellow] Evaluating improved text...[/yellow]")
                            r_improved = await judge_session.evaluate(temp_improved)
                            improved_score = r_improved.get("score", 0)
                            
                            c.print(f"[bold green]Improvement #{improvement_iterations + 1} Response:[/bold green]")
                            c.print(f"[white]{json.dumps(r_improved, indent=2)}[/white]")
                            
                            if improved_score > current_score:
                                current_text = temp_improved
                                current_score = improved_score
                                improved_text = temp_improved
                                content_type = temp_type
                                r = r_improved
                                final_text = temp_improved
                                improvement_iterations += 1
                                
                                c.print(f"[green]✔ Improvement #{improvement_iterations} successful! New score: {improved_score:.2f}[/green]")
                                
                                if improved_score >= 0.7 and r_improved.get("status") == "keep":
                                    c.print(f"[green]🎉 Score threshold reached, stopping improvements[/green]")
                                    break
                            else:
                                c.print(f"[yellow]⚠ Improvement #{improvement_iterations + 1} didn't help (score: {improved_score:.2f} vs {current_score:.2f})[/yellow]")
                                break
                        else:
                            c.print(f"[yellow]⚠ No improvement generated on attempt #{improvement_iterations + 1}[/yellow]")
                            break
                    
                    if improved_text and improved_text != t:
                        c.print(f"[green]📝 Final result: Using improved text (after {improvement_iterations} iterations)[/green]")
                    else:
                        improved_text = None
                        c.print(f"[blue]📝 Final result: Using original text[/blue]")
                    
                    return idx, r, improved_text, content_type, final_text, improvement_iterations, t
            except Exception as e:
                # Escape potential Rich markup in error messages
                error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
                c.print(f"[red]Error processing segment {idx + 1}:[/red] {error_msg}")
                # Escape potential Rich markup in text preview
                safe_text = t[:100].replace('[', '\\[').replace(']', '\\]')
                c.print(f"[dim]Text was:[/dim] {safe_text}...")
                c.print()
                return idx, {"score": 0.5, "status": "revise", "reason": "processing error"}, None, None, t, 0, t

        tasks = [score_and_improve(i, text, source_mapping[i]) for i, text in enumerate(spans)]
        
        try:
            for f in asyncio.as_completed(tasks):
                try:
                    idx, r, improved_text, content_type, final_text, improvement_iterations, original_text = await f
                    tag = r["status"]
                    reasons.append(r["reason"])
                    stats[tag] += 1
                    
                    src_file = source_mapping[idx]
                    
                    improvement_note = f"Improvement iterations: {improvement_iterations}" if improvement_iterations > 0 else ""
                    combined_notes = f"{r.get('reason', '')} | {improvement_note}".strip(" |") if improvement_note else r.get('reason', '')
                    
                    record = PretrainRecord(
                        raw=final_text,
                        improved=improved_text if improved_text and improved_text != original_text else None,
                        type=content_type,
                        meta=RecordMeta(
                            source_file=src_file, 
                            doc_language=langid.classify(final_text)[0], 
                            extracted_by="pdf2seg",
                            confidence=r.get("score"), 
                            tags=[tag] if tag != "keep" else [], 
                            notes=combined_notes
                        )
                    )
                    recs.append(record)
                    all_processed_recs.append(record)
                    processed_count += 1
                    
                    if save_interval > 0 and output_path and processed_count % save_interval == 0:
                        save_incremental(recs)
                        recs.clear()
                        
                except Exception as e:
                    # Escape potential Rich markup in error messages
                    error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
                    c.print(f"[red]Error processing a segment result:[/red] {error_msg}")

        except asyncio.CancelledError:
            c.print("[yellow]Processing cancelled.[/yellow]")
        finally:
            if save_interval > 0 and output_path and recs:
                save_incremental(recs, force=True)
                recs.clear()

        return all_processed_recs, stats, reasons

    try:
        all_recs, stats, reasons = asyncio.run(process())
        show_summary("Combined CSV files", stats, reasons)
        return all_recs
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

def run(i: Path, o: Path, f: str, pretty: bool, n: str, w: int, save_interval: int = 10, force_regenerate: bool = False):
    c.print("[bold cyan]═══ X-Spanformer PDF2JSONL Pipeline ═══[/bold cyan]")
    c.print("[green]✔ Initializing agents and processing pipeline")
    
    # Load config once and share it
    agent_config = load_selfcrit_config()
    
    # Initialize sessions with the shared config
    judge = JudgeSession(config=agent_config, quiet=True)
    improver = ImproveSession(config=agent_config, quiet=True)

    if not i.exists() or not any(i.iterdir()):
        c.print(f"[red]⚠ No PDFs found in {i}[/red]")
        return

    if not i.is_file():
        pdfs = list(i.glob("*.pdf"))
        c.print(f"[bold magenta]Found {len(pdfs)} PDF files:[/bold magenta]")
        for x in pdfs:
            c.print(f"[cyan]• {x.name}[/cyan]")
    else:
        c.print(f"[bold magenta]Processing single PDF file: {i.name}[/bold magenta]")
    c.print()

    base = n.strip().removesuffix(".json").removesuffix(".jsonl") or "dataset"
    
    csv_dir = o / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    if force_regenerate:
        c.print(f"[red]🔄 Force regenerate mode: Clearing all existing CSV files[/red]")
        for existing_csv in csv_dir.glob("*.csv"):
            try:
                existing_csv.unlink()
                c.print(f"[dim]  Removed: {existing_csv.name}[/dim]")
            except OSError:
                pass
        c.print()
    
    # Check for existing CSV files that can be reused
    existing_csvs = list(csv_dir.glob("*.csv"))
    existing_csv_map = {}
    
    if existing_csvs:
        c.print(f"[blue]📁 Found {len(existing_csvs)} existing CSV files in csv directory[/blue]")
        c.print(f"[cyan]⚡ Resume mode: Will reuse existing CSVs where possible[/cyan]")
        for csv_file in existing_csvs:
            c.print(f"[dim]  Available: {csv_file.name}[/dim]")
            existing_csv_map[csv_file.name] = csv_file
        c.print()
    else:
        c.print(f"[cyan]🆕 Fresh start: No existing CSV files found[/cyan]")
        c.print()
    
    csvs = []
    pdf_mapping = {}
    
    c.print(f"[bold blue]━━━ Processing {len(pdfs)} PDF files ━━━[/bold blue]")
    for pdf in pdfs:
        expected_csv_name = f"{hash_name(pdf)}.csv"
        expected_csv_path = csv_dir / expected_csv_name
        
        c.print(f"[cyan]→ Processing: {pdf.name}[/cyan]")
        
        # Check if CSV already exists for this PDF
        if not force_regenerate and expected_csv_path.exists() and expected_csv_path.stat().st_size > 0:
            c.print(f"[green]✔ Found existing CSV: {expected_csv_name}[/green]")
            csvs.append(expected_csv_path)
            pdf_mapping[expected_csv_name] = pdf.name
            c.print(f"[green]✔ Reusing {expected_csv_name} → {pdf.name}[/green]")
        else:
            c.print(f"[yellow]⚙ Generating CSV for {pdf.name}...[/yellow]")
            csv_file = run_pdf2seg(pdf, csv_dir, force_regenerate=force_regenerate)
            if csv_file and csv_file.exists():
                csvs.append(csv_file)
                pdf_mapping[csv_file.name] = pdf.name
                c.print(f"[green]✔ Generated {csv_file.name} → {pdf.name}[/green]")
            else:
                c.print(f"[red]✗ Failed to generate CSV for {pdf.name}[/red]")
    
    if not csvs:
        c.print(f"[red]⚠ No CSV files generated from PDFs[/red]")
        return
    
    c.print(f"[white]→ Processing {len(csvs)} CSV files together[/white]")
    
    # Check if dataset file already exists
    dataset_file = o / f"{base}.jsonl"
    if dataset_file.exists():
        c.print(f"[yellow]⚠ Existing dataset file found: {dataset_file.name}[/yellow]")
        c.print(f"[yellow]  Incremental saves will append to existing file[/yellow]")
    else:
        c.print(f"[cyan]📝 Will create new dataset file: {dataset_file.name}[/cyan]")
    
    # Log the mapping for verification
    c.print("[dim]PDF → CSV mapping:[/dim]")
    for csv_name, pdf_name in pdf_mapping.items():
        c.print(f"[dim]  {csv_name} ← {pdf_name}[/dim]")
    c.print()
    allr = process_all_csvs(csvs, f, w, {}, save_interval, o, base, pdf_mapping)
    
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
    p.add_argument("--force-regenerate", action="store_true", help="Force regeneration of all CSV files, ignoring existing ones")
    a = p.parse_args()
    run(a.input, a.output, a.field, a.pretty, a.name, a.workers, a.save_interval, a.force_regenerate)
