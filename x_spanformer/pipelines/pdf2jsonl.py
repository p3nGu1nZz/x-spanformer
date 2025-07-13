import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import langid
from collections import Counter
from pathlib import Path
from typing import Optional
import re
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.agent_utils import (
    console,
    display_summary_panel,
    display_telemetry_panel,
)
from x_spanformer.agents.selfcrit import process_segment_cycle

from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.pretrain_record import PretrainRecord


def hash_name(p: Path) -> str:
    """Generate a hash for a given path's name."""
    return hashlib.sha256(p.name.encode()).hexdigest()[:8]


def run_pdf2seg(pdf_file: Path, output_dir: Path, force_regenerate: bool = False) -> Optional[Path]:
    """Run pdf2seg on a PDF file to generate CSV output."""

    expected_csv_name = f"{hash_name(pdf_file)}.csv"
    csv_file = output_dir / expected_csv_name

    # If CSV already exists and we're not forcing regeneration, return it
    if not force_regenerate and csv_file.exists() and csv_file.stat().st_size > 0:
        console.print(f"[green]✔ Using existing CSV: {csv_file.name}[/green]")
        return csv_file

    try:
        import pdf2seg
        console.print(f"[yellow]Running pdf2seg on {pdf_file.name}...[/yellow]")

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
            console.print(f"[green]✔ Generated CSV: {csv_file.name}[/green]")
        elif created_csvs:
            # If segments is None but CSVs exist, find the most recent one
            csv_file = max(created_csvs, key=lambda f: f.stat().st_mtime)
            # Rename it to our expected name if it's different
            if csv_file.name != expected_csv_name:
                new_csv_file = output_dir / expected_csv_name
                csv_file.rename(new_csv_file)
                csv_file = new_csv_file
            console.print(f"[green]✔ Found and renamed CSV: {csv_file.name}[/green]")
        else:
            console.print(f"[yellow]No segments extracted, creating minimal CSV for {pdf_file.name}[/yellow]")
            csv_file.write_text("text\n\"No extractable content\"")

        if not csv_file.exists():
            console.print(f"[yellow]Creating fallback CSV for {pdf_file.name}[/yellow]")
            csv_file.write_text("text\n\"sample text\"")

    except ImportError:
        console.print(f"[red]pdf2seg package not found. Please ensure pdf2seg is installed.[/red]")
        return None
    except Exception as e:
        error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
        console.print(f"[red]⚠ Error processing {pdf_file.name}: {error_msg}[/red]")
        # Check if a CSV was still created despite the error
        if csv_file.exists():
            console.print(f"[yellow]⚠ Error was thrown, but CSV file was found. Proceeding...[/yellow]")
            return csv_file
        return None
    finally:
        # Clean up temporary PNG files
        for png_file in output_dir.glob("*.png"):
            try:
                png_file.unlink()
            except OSError as e:
                console.print(f"[yellow]⚠ Could not remove temp file {png_file}: {e}[/yellow]")

    if csv_file.exists():
        # Add the original PDF name to the JSON metadata file
        add_pdf_name_to_json_metadata(csv_file, pdf_file)
        
        console.print(f"[green]✔ CSV ready: {csv_file.name}[/green]")
        return csv_file
    else:
        console.print(f"[red]⚠ Expected CSV file not found: {csv_file.name}[/red]")
        return None


def manifest(p: Path):
    stem = p.stem
    m = p.parent / f"{stem}.json"
    if m.exists():
        with m.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return d.get("csv") or p.name, "pdf2seg (manifest v1)"
    return p.name, "unknown"


def process_all_csvs(csv_files: list[Path], col: str, w: int, cfg: dict, save_interval: int = 1, output_path: Optional[Path] = None, base_name: str = "dataset", pdf_mapping: Optional[dict[str, str]] = None, existing_records: Optional[list[PretrainRecord]] = None) -> list[PretrainRecord]:
    if not csv_files:
        console.print(f"[red]⚠ No CSV files provided[/red]")
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
                console.print(f"[green]✔ Loaded {len(df)} rows from {csv_file.name} (original: {original_pdf_name})[/green]")
            else:
                console.print(f"[red]⚠ Missing '{col}' column in {csv_file.name}[/red]")
        except Exception as e:
            console.print(f"[red]⚠ Error reading {csv_file.name}: {e}[/red]")

    if not all_dfs:
        console.print(f"[red]⚠ No valid CSV files found[/red]")
        return []

    combined_df = pd.concat(all_dfs, ignore_index=True)
    spans = combined_df[col].dropna().astype(str).str.strip().tolist()
    source_mapping = combined_df['source_file'].tolist()

    if not spans:
        console.print(f"[red]⚠ No usable '{col}' values found across all CSV files[/red]")
        return []

    console.print(f"[green]Processing {len(spans)} text segments from {len(source_files)} CSV files[/green]")
    if save_interval > 0:
        console.print(f"[cyan]💾 Incremental saving enabled: saving after every {save_interval} record(s) to {output_path}[/cyan]")
    else:
        console.print("[yellow]💾 Incremental saving disabled. Results will be processed in memory.[/yellow]")
    console.print()

    # Initialize timing metrics
    start_time = time.time()
    # Note: We can't predict how many valid records we'll have, so estimated_total_saves will be updated after text splitting
    estimated_total_saves = 1  # Placeholder, will be updated after text splitting
    records_saved_this_session = 0
    records_discarded_this_session = 0
    total_segment_count = len(spans)  # Initialize with original count, will be updated after splitting

    def save_chunk(records: list[PretrainRecord]):
        """Save a chunk of records incrementally to prevent data loss"""
        nonlocal records_saved_this_session

        if not output_path or not records:
            return

        output_path.mkdir(parents=True, exist_ok=True)
        dataset_file = output_path / f"{base_name}.jsonl"
        mode = "a"  # Always append

        with dataset_file.open(mode, encoding="utf-8") as writer:
            for record in records:
                writer.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

        num_saved = len(records)
        records_saved_this_session += num_saved

        console.print(f"[blue]💾 Saved {num_saved} record(s) to {dataset_file.name} (total this session: {records_saved_this_session})[/blue]")

    def save_discard_chunk(records: list[PretrainRecord]):
        """Save discarded records to discard.jsonl file"""
        nonlocal records_discarded_this_session

        if not output_path or not records:
            return

        output_path.mkdir(parents=True, exist_ok=True)
        discard_file = output_path / "discard.jsonl"
        mode = "a"  # Always append

        with discard_file.open(mode, encoding="utf-8") as writer:
            for record in records:
                writer.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

        num_discarded = len(records)
        records_discarded_this_session += num_discarded

        console.print(f"[red]🗑️ Saved {num_discarded} discarded record(s) to {discard_file.name} (total this session: {records_discarded_this_session})[/red]")

    async def process():
        sem = asyncio.Semaphore(w)
        stats = Counter()
        all_processed_recs = []
        records_to_save = []
        recs, reasons = [], []
        # Initialize counters, including already-processed records if resuming
        processed_count_total = len(existing_records) if existing_records else 0
        # Count existing valid records (status "keep") toward valid count
        valid_records_count = sum(1 for r in existing_records if r.meta.status == "keep") if existing_records else 0

        # Initialize once, sharing the same config  
        max_raw_length = cfg.get("processor", {}).get("max_raw_length", 512)
        discard_threshold = cfg.get("critique", {}).get("discard_threshold", 0.25)
        improvement_threshold = cfg.get("critique", {}).get("threshold", 0.8)
        # Pre-process spans to split long texts
        discard_threshold = cfg.get("critique", {}).get("discard_threshold", 0.25)
        improvement_threshold = cfg.get("critique", {}).get("threshold", 0.8)
        
        async def score_and_improve(idx: int, t: str, source_file: str):
            try:
                console.print(f"[bold blue]━━━ Processing segment {idx + 1} ━━━[/bold blue]")
                console.print(f"[dim]Source: {source_file} | Text ({len(t)} chars):[/dim] {t}")
                console.print()

                # Use the SelfCritAgent system for complete processing
                result = await process_segment_cycle(t, max_cycles=6)
                
                # Extract final results
                final_text = result.get("final_text", t)
                improvement_iterations = result.get("cycles_completed", 0)
                
                # Determine if improvement was made
                improved_text = final_text if final_text != t else None
                
                # Get content type from judge result (Natural, Code, or Mixed)
                content_type = result.get("type", "Natural")  # Use judge's classification
                
                # Create detailed conversation log that includes full processing history
                detailed_conversation_log = result.get("conversation_history", [])
                
                # If no conversation history returned, create basic log entry
                if not detailed_conversation_log:
                    detailed_conversation_log = [{
                        "step": "selfcrit_cycle",
                        "input_text": t,
                        "final_text": final_text,
                        "cycles_completed": improvement_iterations,
                        "model": cfg.get("model", {}).get("name", "unknown"),
                        "result": result,
                        "judge_evaluations": result.get("judge_history", []),
                        "improvement_history": result.get("improvement_history", []),
                        "consensus_details": result.get("consensus_details", {}),
                        "threshold_checks": {
                            "improvement_threshold": cfg.get("critique", {}).get("threshold", 0.8),
                            "discard_threshold": cfg.get("critique", {}).get("discard_threshold", 0.25),
                            "final_score": result.get("score", 0),
                            "triggered_improvement": improvement_iterations > 0
                        }
                    }]

                # Save AI processing log for this segment
                if output_path:
                    save_ai_processing_log(output_path, source_file, str(idx), t, improved_text, content_type, [result], improvement_iterations, detailed_conversation_log, result)

                return idx, result, improved_text, content_type, final_text, improvement_iterations, t
                
            except Exception as e:
                # Escape potential Rich markup in error messages
                error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
                console.print(f"[red]Error processing segment {idx + 1}:[/red] {error_msg}")
                # Escape potential Rich markup in text preview
                safe_text = t[:100].replace('[', '\\[').replace(']', '\\]')
                console.print(f"[dim]Text was:[/dim] {safe_text}...")
                console.print()
                return idx, {"score": 0.5, "status": "revise", "reason": "processing error"}, None, None, t, 0, t

        # Process files sequentially to maintain order
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if col in df.columns:
                    original_pdf_name = pdf_mapping.get(csv_file.name, csv_file.name)
                    console.print(f"[cyan]� Processing {csv_file.name} (original: {original_pdf_name})[/cyan]")
                    
                    file_spans = df[col].dropna().astype(str).str.strip().tolist()
                    
                    # Pre-process spans to split long texts for this file
                    expanded_file_spans = []
                    
                    for text in file_spans:
                        if len(text) > max_raw_length:
                            text_chunks = split_long_text(text, max_raw_length)
                        else:
                            text_chunks = [text]
                        
                        for chunk in text_chunks:
                            expanded_file_spans.append(chunk)
                    
                    console.print(f"[dim]  → {len(file_spans)} segments → {len(expanded_file_spans)} after splitting[/dim]")
                    
                    # Skip already processed segments
                    if existing_records:
                        processed_raws = {rec.raw for rec in existing_records}
                        unprocessed_file_spans = [text for text in expanded_file_spans if text not in processed_raws]
                        
                        if len(unprocessed_file_spans) < len(expanded_file_spans):
                            skipped_count = len(expanded_file_spans) - len(unprocessed_file_spans)
                            console.print(f"[cyan]🔄 Skipped {skipped_count} already processed segments from {csv_file.name}[/cyan]")
                            expanded_file_spans = unprocessed_file_spans
                    
                    console.print(f"[green]✔ Ready to process {len(expanded_file_spans)} segments from {csv_file.name}[/green]")
                    console.print()

                    # Process this file's segments with controlled concurrency
                    semaphore = asyncio.Semaphore(w)  # Use worker count for concurrency within each file
                    
                    async def process_segment_with_semaphore(local_idx, text):
                        async with semaphore:
                            global_idx = processed_count_total + len(all_processed_recs) + local_idx
                            result = await score_and_improve(global_idx, text, original_pdf_name)
                            return local_idx, result
                    
                    # Create tasks for concurrent processing within this file
                    tasks = [process_segment_with_semaphore(local_idx, text) 
                            for local_idx, text in enumerate(expanded_file_spans)]
                    
                    # Process segments concurrently but maintain results order
                    segment_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results in order to maintain deterministic output
                    for local_idx, result_tuple in enumerate(segment_results):
                        try:
                            if isinstance(result_tuple, Exception):
                                console.print(f"[red]Error processing segment {local_idx + 1}: {result_tuple}[/red]")
                                continue
                            
                            if result_tuple is None:
                                continue
                            
                            # Ensure we can unpack the outer tuple (local_idx, score_and_improve_result)
                            if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                                console.print(f"[red]Unexpected result format for segment {local_idx + 1}[/red]")
                                continue
                                
                            result_local_idx, result = result_tuple
                            # result is the return value from score_and_improve which should be a tuple
                            if result:
                                idx, r, improved_text, content_type, final_text, improvement_iterations, original_text = result
                                
                                # Handle discarded records - save them to discard.jsonl but not main dataset
                                if r.get("score", 0) < discard_threshold or r.get("status") == "discard":
                                    console.print(f"[red]🗑️ Segment discarded (score: {r.get('score', 0):.3f})[/red]")
                                    
                                    # Still create a record for the discarded segment
                                    tag = "discard"
                                    improvement_note = f"Improvement iterations: {improvement_iterations}" if improvement_iterations > 0 else ""
                                    combined_notes = f"{r.get('reason', '')} | {improvement_note}".strip(" |") if improvement_note else r.get('reason', '')

                                    discard_record = PretrainRecord(
                                        raw=original_text,
                                        improved=improved_text if improved_text and improved_text != original_text else None,
                                        type=content_type,
                                        meta=RecordMeta(
                                            source_file=original_pdf_name,
                                            doc_language=langid.classify(final_text)[0],
                                            extracted_by="pdf2seg",
                                            confidence=r.get("score"),
                                            status=tag,
                                            tags=[tag],
                                            notes=combined_notes
                                        )
                                    )
                                    
                                    # Save discarded record immediately
                                    save_discard_chunk([discard_record])
                                    console.print()
                                    continue
                                
                                tag = r["status"]
                                stats[tag] += 1
                                reasons.append(r["reason"])

                                improvement_note = f"Improvement iterations: {improvement_iterations}" if improvement_iterations > 0 else ""
                                combined_notes = f"{r.get('reason', '')} | {improvement_note}".strip(" |") if improvement_note else r.get('reason', '')

                                record = PretrainRecord(
                                    raw=original_text,
                                    improved=improved_text if improved_text and improved_text != original_text else None,
                                    type=content_type,
                                    meta=RecordMeta(
                                        source_file=original_pdf_name,
                                        doc_language=langid.classify(final_text)[0],
                                        extracted_by="pdf2seg",
                                        confidence=r.get("score"),
                                        status=tag,
                                        tags=[tag] if tag != "keep" else [],
                                        notes=combined_notes
                                    )
                                )
                                
                                all_processed_recs.append(record)

                                # Only save records with status "keep" to the main dataset
                                if tag == "keep":
                                    if save_interval > 0:
                                        # For immediate saving (save_interval=1), save each record right away
                                        # For batch saving (save_interval>1), accumulate and save when batch is full
                                        if save_interval == 1:
                                            save_chunk([record])
                                        else:
                                            records_to_save.append(record)
                                            if len(records_to_save) >= save_interval:
                                                save_chunk(records_to_save)
                                                records_to_save.clear()
                                    else:
                                        # save_interval = 0 means no incremental saving, accumulate all
                                        records_to_save.append(record)

                                    # Count valid records (status "keep") toward valid count
                                    valid_records_count += 1
                                elif tag == "revise":
                                    # Save revised records to a separate file for analysis
                                    save_discard_chunk([record])
                                    console.print(f"[yellow]📝 Revised record saved to discard.jsonl (score: {r.get('score', 0):.3f})[/yellow]")
                                    
                        except Exception as e:
                            error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
                            console.print(f"[red]Error processing segment {local_idx + 1}:[/red] {error_msg}")
                            console.print()
                    
                    console.print(f"[green]✔ Completed processing {csv_file.name}[/green]")
                    console.print()
                else:
                    console.print(f"[red]⚠ Missing '{col}' column in {csv_file.name}[/red]")
            except Exception as e:
                console.print(f"[red]⚠ Error reading {csv_file.name}: {e}[/red]")

        # Save any remaining records that weren't saved yet
        if records_to_save:
            save_chunk(records_to_save)
            records_to_save.clear()

        return all_processed_recs, stats, reasons

    try:
        all_recs, stats, reasons = asyncio.run(process())
        display_summary_panel("Combined CSV files", stats, reasons)
        return all_recs
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting.[/bold red]")
        return []


def run(i: Path, o: Path, f: str, pretty: bool, n: str, w: int, save_interval: int = 1, force: bool = False):
    console.print("[bold cyan]═══ X-Spanformer PDF2JSONL Pipeline ═══[/bold cyan]")
    console.print("[green]✔ Initializing agents and processing pipeline[/green]")

    if save_interval < 0:
        console.print(f"[red]Error: --save-interval cannot be negative. Use 0 to disable incremental saving.[/red]")
        return

    if not i.exists():
        console.print(f"[red]Error: Input path does not exist: {i}[/red]")
        return

    if not i.is_file() and not any(i.iterdir()):
        console.print(f"[yellow]Warning: Input directory is empty: {i}[/yellow]")
        # Allow continuing, as there might be existing CSVs to process
    
    # Discover input files (PDFs or CSVs)
    pdfs = []
    csv_direct_mode = False
    direct_csv_file = None
    
    if i.is_dir():
        pdfs = sorted(list(i.glob("*.pdf")))
        if not pdfs:
            console.print(f"[yellow]No PDF files found in {i}[/yellow]")
        else:
            console.print(f"[blue]📋 Found {len(pdfs)} PDF files in input directory[/blue]")
            
            # Count total pages across all PDFs
            total_pages = 0
            unknown_count = 0
            console.print(f"[cyan]📄 Counting pages in {len(pdfs)} PDF files...[/cyan]")
            
            for idx, pdf in enumerate(pdfs, 1):
                page_count = count_pdf_pages(pdf)
                if page_count > 0:
                    total_pages += page_count
                    console.print(f"[dim]  {idx:2d}. {pdf.name} ({page_count} pages)[/dim]")
                else:
                    unknown_count += 1
                    console.print(f"[dim]  {idx:2d}. {pdf.name}[/dim]")
            
            if total_pages > 0:
                if unknown_count > 0:
                    console.print(f"[bold blue]📊 Total workload: {total_pages}+ pages across {len(pdfs)} PDF files ({unknown_count} PDFs with unknown page count)[/bold blue]")
                else:
                    console.print(f"[bold blue]📊 Total workload: {total_pages} pages across {len(pdfs)} PDF files[/bold blue]")
            else:
                console.print(f"[yellow]⚠ Could not determine page counts (PyPDF2 may not be installed)[/yellow]")
    elif i.is_file() and i.suffix.lower() == ".pdf":
        pdfs = [i]
        page_count = count_pdf_pages(i)
        if page_count > 0:
            console.print(f"[blue]📋 Processing single PDF file: {i.name} ({page_count} pages)[/blue]")
        else:
            console.print(f"[blue]📋 Processing single PDF file: {i.name}[/blue]")
    elif i.is_file() and i.suffix.lower() == ".csv":
        # Direct CSV processing mode
        csv_direct_mode = True
        direct_csv_file = i
        console.print(f"[blue]📋 Processing single CSV file directly: {i.name}[/blue]")
        console.print(f"[cyan]🚀 CSV Direct Mode: Skipping PDF2SEG conversion[/cyan]")
    else:
        console.print(f"[red]Error: Input must be a PDF file, CSV file, or a directory containing PDF files.[/red]")
        return

    console.print()

    base = n.strip().removesuffix(".json").removesuffix(".jsonl") or "dataset"

    csv_dir = o / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    if force:
        console.print(f"[red]🔄 Force regenerate mode: Clearing all existing CSV files[/red]")
        for existing_csv in csv_dir.glob("*.csv"):
            try:
                existing_csv.unlink()
                console.print(f"[dim]  Removed: {existing_csv.name}[/dim]")
            except OSError:
                pass
        console.print()

    # Check for existing CSV files that can be reused
    existing_csvs = list(csv_dir.glob("*.csv"))
    existing_csv_map = {}

    if existing_csvs:
        console.print(f"[blue]📁 Found {len(existing_csvs)} existing CSV files in csv directory[/blue]")
        console.print(f"[cyan]⚡ Resume mode: Will reuse existing CSVs where possible[/cyan]")
        for csv_file in existing_csvs:
            console.print(f"[dim]  Available: {csv_file.name}[/dim]")
            existing_csv_map[csv_file.name] = csv_file
        console.print()
    else:
        console.print(f"[cyan]🆕 Fresh start: No existing CSV files found[/cyan]")
        console.print()

    csvs = []
    pdf_mapping = {}

    # Handle direct CSV mode (when input is a single CSV file)
    if csv_direct_mode and direct_csv_file:
        console.print(f"[bold blue]━━━ Direct CSV Processing Mode ━━━[/bold blue]")
        console.print(f"[cyan]📁 Using provided CSV file: {direct_csv_file.name}[/cyan]")
        
        # Check if the CSV file exists and has content
        if not direct_csv_file.exists():
            console.print(f"[red]Error: CSV file does not exist: {direct_csv_file}[/red]")
            return
        
        file_size = direct_csv_file.stat().st_size
        if file_size == 0:
            console.print(f"[red]Error: CSV file is empty: {direct_csv_file}[/red]")
            return
        
        console.print(f"[green]✔ CSV file found with {file_size} bytes[/green]")
        
        # Check if the specified field exists in the CSV
        try:
            import pandas as pd
            df_check = pd.read_csv(direct_csv_file)
            if f not in df_check.columns:
                console.print(f"[red]Error: Field '{f}' not found in CSV. Available columns: {list(df_check.columns)}[/red]")
                return
            console.print(f"[green]✔ Found field '{f}' with {len(df_check)} rows in CSV[/green]")
        except Exception as e:
            console.print(f"[red]Error reading CSV file: {e}[/red]")
            return
        
        csvs = [direct_csv_file]
        # For direct CSV mode, use the CSV filename as the source
        pdf_mapping[direct_csv_file.name] = direct_csv_file.name.replace('.csv', '_direct.csv')
        console.print(f"[cyan]🚀 Ready to process {len(df_check)} text segments from CSV[/cyan]")
        console.print()
    else:
        # Standard PDF processing mode
        console.print(f"[bold blue]━━━ Converting PDFs to CSV format ━━━[/bold blue]")
        new_csvs_needed = 0
        reusable_csvs = 0
        
        # First pass: check what needs to be generated
        for pdf in pdfs:
            expected_csv_name = f"{hash_name(pdf)}.csv"
            expected_csv_path = csv_dir / expected_csv_name
            if not force and expected_csv_path.exists() and expected_csv_path.stat().st_size > 0:
                reusable_csvs += 1
            else:
                new_csvs_needed += 1
        
        if reusable_csvs > 0:
            console.print(f"[green]✔ Can reuse {reusable_csvs} existing CSV files[/green]")
        if new_csvs_needed > 0:
            console.print(f"[yellow]⚙ Need to generate {new_csvs_needed} new CSV files[/yellow]")
        console.print()

        # Second pass: actually process the PDFs
        for idx, pdf in enumerate(pdfs, 1):
            expected_csv_name = f"{hash_name(pdf)}.csv"
            expected_csv_path = csv_dir / expected_csv_name

            console.print(f"[cyan]→ Processing {idx}/{len(pdfs)}: {pdf.name}[/cyan]")

            # Check if CSV already exists for this PDF
            if not force and expected_csv_path.exists() and expected_csv_path.stat().st_size > 0:
                console.print(f"[green]✔ Found existing CSV: {expected_csv_name}[/green]")
                csvs.append(expected_csv_path)
                pdf_mapping[expected_csv_name] = pdf.name
                console.print(f"[green]✔ Reusing {expected_csv_name} → {pdf.name}[/green]")
            else:
                console.print(f"[yellow]⚙ Generating CSV for {pdf.name}...[/yellow]")
                csv_file = run_pdf2seg(pdf, csv_dir, force_regenerate=force)
                if csv_file and csv_file.exists():
                    csvs.append(csv_file)
                    pdf_mapping[csv_file.name] = pdf.name
                    console.print(f"[green]✔ Generated {csv_file.name} → {pdf.name}[/green]")
                else:
                    console.print(f"[red]✗ Failed to generate CSV for {pdf.name}[/red]")

        if not csvs:
            console.print(f"[red]⚠ No CSV files generated from PDFs[/red]")
            return

    console.print()
    console.print(f"[bold blue]━━━ Processing text segments to training data ━━━[/bold blue]")
    console.print(f"[white]📊 Processing {len(csvs)} CSV files for text extraction and AI analysis[/white]")

    # Log the mapping for verification
    if not csv_direct_mode:
        console.print("[dim]PDF → CSV mapping:[/dim]")
        for csv_name, pdf_name in pdf_mapping.items():
            console.print(f"[dim]  {csv_name} ← {pdf_name}[/dim]")
    elif direct_csv_file:
        console.print(f"[dim]Direct CSV mode: {direct_csv_file.name}[/dim]")
    
    # Check if dataset file already exists
    dataset_file = o / f"{base}.jsonl"
    existing_records = []
    if dataset_file.exists():
        console.print(f"[yellow]⚠ Existing dataset file found: {dataset_file.name}[/yellow]")
        try:
            with dataset_file.open("r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if line.strip():
                        existing_records.append(PretrainRecord.model_validate_json(line))
            if existing_records:
                console.print(f"[cyan]✔ Loaded {len(existing_records)} records. Will skip processing for these.[/cyan]")
        except (json.JSONDecodeError, ValidationError) as e:
            console.print(f"[red]⚠ Could not parse existing dataset file: {e}. Starting fresh.[/red]")
            existing_records = []
    else:
        console.print(f"[cyan]📝 Will create new dataset file: {dataset_file.name}[/cyan]")

    console.print()
    # Load the agent config for processing
    agent_config = load_selfcrit_config()
    allr = process_all_csvs(csvs, f, w, agent_config, save_interval, o, base, pdf_mapping, existing_records=existing_records)

    if not allr:
        console.print(f"[red]⚠ No valid records found across all CSVs[/red]")
        return

    total = len(allr)
    # Filter for records with status "keep"
    kept = sum(1 for r in allr if r.meta.status == "keep")
    ratio = round(kept / total * 100, 2) if total > 0 else 0

    console.print("[bold green]═══ Final Summary ═══[/bold green]")
    console.print(f"[white]Total records processed this session:[/white] {total}")
    console.print(f"[green]Kept:[/green] {kept} [cyan]({ratio}%)[/cyan]")
    console.print(f"[yellow]Revised/Discarded:[/yellow] {total - kept}")
    console.print()

    # The final save is now handled by the incremental saver.
    # We can optionally write a pretty-printed JSON for inspection.
    if pretty:
        j2 = o / f"{base}.json"
        with j2.open("w", encoding="utf-8") as writer:
            json.dump([x.model_dump() for x in allr], writer, ensure_ascii=False, indent=2)
        console.print(f"[cyan]• Pretty JSON → {j2.name}[/cyan]")


def split_long_text(text: str, max_length: int = 512) -> list[str]:
    """
    Split text that exceeds max_length into smaller chunks, ensuring no chunk is longer than max_length.
    It prioritizes splitting at sentence boundaries, then word boundaries, and finally at the character level
    for very long, unbroken strings of text.
    """
    if len(text) <= max_length:
        return [text]

    # Use a simple regex for sentence splitting to avoid heavy dependencies like spaCy
    # This regex looks for sentence-ending punctuation followed by a space or the end of the string.
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    
    chunks = []
    
    for sent in sentences:
        if not sent.strip():
            continue
            
        if len(sent) <= max_length:
            chunks.append(sent)
        else:
            # The sentence itself is too long, so we need to split it further.
            # First, try splitting by words.
            words = sent.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_length:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
            
            if current_chunk:
                chunks.append(current_chunk)

    # Final check: If any chunk is still too long (e.g., a very long word or token),
    # we must split it at the character level.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            for i in range(0, len(chunk), max_length):
                final_chunks.append(chunk[i:i+max_length])
        else:
            final_chunks.append(chunk)
            
    # If, after all that, we have no chunks, it means the original text was probably just whitespace.
    # Return an empty list in that case.
    if not final_chunks and not text.strip():
        return []
        
    return final_chunks


def count_pdf_pages(pdf_path: Path) -> int:
    """Count the number of pages in a PDF file using PyPDF2."""
    try:
        import pypdf
        with pdf_path.open('rb') as file:
            reader = pypdf.PdfReader(file)
            return len(reader.pages)
    except ImportError:
        # PyPDF2 not available, return unknown count
        return -1
    except Exception as e:
        # If there's any error reading the PDF, return unknown count silently
        return -1


def main():
    parser = argparse.ArgumentParser(description="X-Spanformer PDF2JSONL Pipeline")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input: PDF file, CSV file, or directory containing PDF files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory for results")
    parser.add_argument("-f", "--field", type=str, default="text", help="Field name in CSV to process")
    parser.add_argument("-p", "--pretty", action="store_true", help="Output pretty JSON file")
    parser.add_argument("-n", "--name", type=str, default="dataset", help="Base name for output files")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--save-interval", type=int, default=1, help="Save progress every N records. Default is 1 (save every record). Use 0 to disable incremental saving.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all cached data")
    args = parser.parse_args()

    run(args.input, args.output, args.field, args.pretty, args.name, args.workers, args.save_interval, args.force)


def save_ai_processing_log(output_dir: Path, source_file: str, segment_id: str, 
                          original_text: str, improved_text: Optional[str], content_type: Optional[str],
                          judge_responses: list, improvement_iterations: int, detailed_conversation_log: Optional[list] = None, result_data: Optional[dict] = None):
    """Save detailed AI processing logs for a segment as structured JSON with full conversation context."""
    # Create jsonl directory structure for debug logs
    hash_str = hash_name(Path(source_file))
    jsonl_dir = output_dir / "jsonl" / hash_str
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with padded segment ID (8 digits)
    padded_segment_id = str(segment_id).zfill(8)
    log_filename = f"{hash_str}_{padded_segment_id}.json"
    log_path = jsonl_dir / log_filename
    
    # Build structured log data with enhanced conversation details
    log_data = {
        "metadata": {
            "segment_id": segment_id,
            "source_file": source_file,
            "hash": hash_str,
            "timestamp": datetime.now().isoformat(),
            "improvement_iterations": improvement_iterations,
            "version": "2.0"  # Version to track log format changes
        },
        "processing": {
            "original_text": original_text,
            "improved_text": improved_text if improved_text and improved_text != original_text else None,
            "content_type": content_type,
            "judge_responses": judge_responses,
            "final_status": judge_responses[-1].get("status") if judge_responses else None,
            "final_score": judge_responses[-1].get("score") if judge_responses else None,
            "final_reason": judge_responses[-1].get("reason") if judge_responses else None
        },
        "conversation_log": detailed_conversation_log or [],
        "debugging_info": {
            "total_judge_calls": result_data.get("total_judge_calls", len(judge_responses)) if result_data else len(judge_responses),
            "conversation_steps": len(detailed_conversation_log) if detailed_conversation_log else 0,
            "processing_flow": [step.get("step", "unknown") for step in (detailed_conversation_log or [])],
            "models_used": result_data.get("models_used", ["unknown"]) if result_data else ["unknown"],
            "score_progression": result_data.get("score_progression", [resp.get("score") for resp in judge_responses if resp.get("score") is not None]) if result_data else [resp.get("score") for resp in judge_responses if resp.get("score") is not None],
            "processing_time_seconds": result_data.get("processing_time_seconds", 0.0) if result_data else 0.0
        }
    }
    
    # Write to JSON file (pretty formatted for debugging)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def add_pdf_name_to_json_metadata(csv_file: Path, pdf_file: Path):
    """Add the original PDF name to the JSON metadata file immediately after creation."""
    try:
        # Find the corresponding JSON file
        hash_str = hash_name(pdf_file)
        json_dir = csv_file.parent / hash_str
        json_file = json_dir / f"{hash_str}.json"
        
        if json_file.exists():
            # Read existing JSON data
            with json_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Add original PDF filename if not already present
            if "original_pdf" not in metadata:
                metadata["original_pdf"] = pdf_file.name
                
                # Write back updated JSON
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                console.print(f"[green]✔ Added PDF name to {json_file.name}: {pdf_file.name}[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not update JSON metadata: {e}[/yellow]")


if __name__ == "__main__":
    main()