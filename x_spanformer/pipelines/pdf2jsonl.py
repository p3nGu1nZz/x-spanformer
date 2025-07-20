import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Optional
import re
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from x_spanformer.agents.config_loader import load_judge_config
from x_spanformer.agents.ollama_client import check_ollama_connection
from x_spanformer.agents.rich_utils import (
    console,
    display_summary_panel,
    display_telemetry_panel,
    display_judgment_result,
)
from x_spanformer.agents.session.judge_session import JudgeSession
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.pipelines.shared.csv_processor import process_all_csvs


def hash_name(p: Path) -> str:
    """Generate a hash for a given path's name."""
    return hashlib.sha256(p.name.encode()).hexdigest()[:8]


def save_ai_processing_log(output_dir: Path, source_file: str, segment_id: str, 
                          original_text: str, judge_responses: list, consensus_result: dict):
    """Save detailed AI processing logs for a segment."""
    # Create jsonl directory structure with hash-based subdirectories
    hash_str = hash_name(Path(source_file))
    jsonl_dir = output_dir / "jsonl" / hash_str
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    
    # Create conversation log filename
    log_filename = f"{hash_str}_{segment_id}.json"
    log_path = jsonl_dir / log_filename
    
    # Build comprehensive conversation log as JSON
    conversation_log = {
        "metadata": {
            "segment_id": segment_id,
            "source_file": source_file,
            "source_hash": hash_str,
            "timestamp": datetime.now().isoformat()
        },
        "original_text": original_text,
        "content_type": consensus_result.get("type", "natural"),
        "judge_responses": judge_responses,
        "processing_summary": {
            "total_judge_calls": len(judge_responses),
            "final_status": consensus_result.get("status", "discard"),
            "final_score": consensus_result.get("score", 0.0)
        }
    }
    
    # Write to JSON file for easy loading into other systems
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(conversation_log, f, indent=2, ensure_ascii=False)


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
    
    # Check Ollama connection before starting any processing
    console.print("[cyan]🔍 Testing Ollama connection...[/cyan]")
    agent_config = load_judge_config()
    model = agent_config.get("judge", {}).get("model_name", "llama3.2:1b")
    
    try:
        if not asyncio.run(check_ollama_connection(model)):
            console.print(f"[red]❌ Ollama connection failed! Please ensure Ollama is running and model '{model}' is available.[/red]")
            console.print(f"[yellow]💡 Start Ollama with: ollama serve[/yellow]")
            console.print(f"[yellow]💡 Load model with: ollama run {model}[/yellow]")
            return
        else:
            console.print(f"[green]✅ Ollama is running and accessible (model: {model})[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error testing Ollama connection: {str(e)}[/red]")
        console.print(f"[red]🚫 Exiting due to connection test failure. Please check Ollama is running.[/red]")
        return
    
    console.print()
    
    # First, discover all PDF files
    pdfs = []
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
                console.print(f"[yellow]⚠ Could not determine page counts (pypdf may not be installed)[/yellow]")
    elif i.is_file() and i.suffix.lower() == ".pdf":
        pdfs = [i]
        page_count = count_pdf_pages(i)
        if page_count > 0:
            console.print(f"[blue]📋 Processing single PDF file: {i.name} ({page_count} pages)[/blue]")
        else:
            console.print(f"[blue]📋 Processing single PDF file: {i.name}[/blue]")
    else:
        console.print(f"[red]Error: Input must be a PDF file or a directory containing PDF files.[/red]")
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
    existing_csvs = sorted(list(csv_dir.glob("*.csv")))  # Sort for deterministic ordering
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

    # Create temporary directory for batch processing
    temp_dir = o / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set page threshold for batching (configurable)
    LARGE_PDF_THRESHOLD = 400  # Pages
    PAGES_PER_BATCH = 200     # Pages per batch
    
    # Second pass: actually process the PDFs with batching support
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
            # Check if this is a large PDF that needs batching
            page_count = count_pdf_pages(pdf)
            
            if page_count > LARGE_PDF_THRESHOLD:
                console.print(f"[yellow]📄 Large PDF detected ({page_count} pages), using batch processing...[/yellow]")
                
                # Split PDF into batches
                batch_files = split_large_pdf(pdf, temp_dir, PAGES_PER_BATCH)
                
                if len(batch_files) > 1:
                    # Process each batch
                    batch_results = []
                    temp_csvs = []
                    
                    for batch_idx, batch_pdf in enumerate(batch_files, 1):
                        console.print(f"[dim]  Processing batch {batch_idx}/{len(batch_files)}: {batch_pdf.name}[/dim]")
                        
                        # Create temporary CSV directory for batches
                        batch_csv_dir = temp_dir / "batch_csvs"
                        batch_csv_dir.mkdir(parents=True, exist_ok=True)
                        
                        batch_csv = run_pdf2seg(batch_pdf, batch_csv_dir, force_regenerate=True)
                        
                        if batch_csv and batch_csv.exists():
                            batch_results.append((batch_pdf, batch_csv))
                            temp_csvs.append(batch_csv)
                            console.print(f"[dim]    ✔ Batch {batch_idx} complete[/dim]")
                        else:
                            console.print(f"[yellow]    ⚠ Batch {batch_idx} failed[/yellow]")
                    
                    # Aggregate batch results
                    if batch_results:
                        final_csv = aggregate_batch_results(pdf, batch_results, csv_dir)
                        if final_csv and final_csv.exists():
                            csvs.append(final_csv)
                            pdf_mapping[final_csv.name] = pdf.name
                            console.print(f"[green]✔ Large PDF processed via batching: {final_csv.name} → {pdf.name}[/green]")
                        else:
                            console.print(f"[red]✗ Failed to aggregate batch results for {pdf.name}[/red]")
                    else:
                        console.print(f"[red]✗ No successful batches for {pdf.name}[/red]")
                    
                    # Clean up batch files and temporary CSVs
                    cleanup_batch_files(batch_files)
                    for temp_csv in temp_csvs:
                        try:
                            if temp_csv.exists():
                                temp_csv.unlink()
                        except OSError:
                            pass
                else:
                    # Only one batch (or split failed), process normally
                    console.print(f"[yellow]⚙ Generating CSV for {pdf.name} (single batch)...[/yellow]")
                    csv_file = run_pdf2seg(pdf, csv_dir, force_regenerate=force)
                    if csv_file and csv_file.exists():
                        csvs.append(csv_file)
                        pdf_mapping[csv_file.name] = pdf.name
                        console.print(f"[green]✔ Generated {csv_file.name} → {pdf.name}[/green]")
                    else:
                        console.print(f"[red]✗ Failed to generate CSV for {pdf.name}[/red]")
            else:
                # Normal-sized PDF, process directly
                console.print(f"[yellow]⚙ Generating CSV for {pdf.name}...[/yellow]")
                csv_file = run_pdf2seg(pdf, csv_dir, force_regenerate=force)
                if csv_file and csv_file.exists():
                    csvs.append(csv_file)
                    pdf_mapping[csv_file.name] = pdf.name
                    console.print(f"[green]✔ Generated {csv_file.name} → {pdf.name}[/green]")
                else:
                    console.print(f"[red]✗ Failed to generate CSV for {pdf.name}[/red]")
    
    # Clean up temporary batch directory
    try:
        if temp_dir.exists():
            # Remove any remaining files
            for temp_file in temp_dir.rglob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
            # Remove directories (from deepest to shallowest)
            for temp_subdir in sorted(temp_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if temp_subdir.is_dir():
                    temp_subdir.rmdir()
            # Remove the temp directory itself
            temp_dir.rmdir()
            console.print(f"[dim]Cleaned up temporary batch processing directory[/dim]")
    except OSError:
        # Cleanup failed, but continue processing
        pass

    if not csvs:
        console.print(f"[red]⚠ No CSV files generated from PDFs[/red]")
        return

    console.print()
    console.print(f"[bold blue]━━━ Processing text segments to training data ━━━[/bold blue]")
    console.print(f"[white]📊 Processing {len(csvs)} CSV files for text extraction and AI analysis[/white]")

    # Log the mapping for verification
    console.print("[dim]PDF → CSV mapping:[/dim]")
    for csv_name, pdf_name in pdf_mapping.items():
        console.print(f"[dim]  {csv_name} ← {pdf_name}[/dim]")
    
    # Check if dataset and discard files already exist in jsonl directory
    jsonl_dir = o / "jsonl"
    dataset_file = jsonl_dir / f"{base}.jsonl"
    discard_file = jsonl_dir / "discard.jsonl"
    existing_records = []
    
    # Clear existing files if force mode is enabled
    if force:
        if dataset_file.exists():
            console.print(f"[red]🔄 Force mode: Removing existing dataset file: {dataset_file.name}[/red]")
            try:
                dataset_file.unlink()
                console.print(f"[dim]  Removed: {dataset_file.name}[/dim]")
            except OSError:
                pass
        if discard_file.exists():
            console.print(f"[red]🔄 Force mode: Removing existing discard file: {discard_file.name}[/red]")
            try:
                discard_file.unlink()
                console.print(f"[dim]  Removed: {discard_file.name}[/dim]")
            except OSError:
                pass
    
    # Load existing records from both dataset.jsonl and discard.jsonl
    keep_count = 0
    discard_count = 0
    
    if dataset_file.exists():
        console.print(f"[yellow]⚠ Existing dataset file found: {dataset_file.name}[/yellow]")
        try:
            with dataset_file.open("r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if line.strip():
                        existing_records.append(PretrainRecord.model_validate_json(line))
                        keep_count += 1
            console.print(f"[cyan]✔ Loaded {keep_count} keep records from {dataset_file.name}[/cyan]")
        except (json.JSONDecodeError, ValidationError) as e:
            console.print(f"[red]⚠ Could not parse existing dataset file: {e}. Starting fresh.[/red]")
            existing_records = []
            keep_count = 0
    else:
        console.print(f"[cyan]📝 Will create new dataset file: {dataset_file.name}[/cyan]")
    
    if discard_file.exists():
        console.print(f"[yellow]⚠ Existing discard file found: {discard_file.name}[/yellow]")
        try:
            with discard_file.open("r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if line.strip():
                        existing_records.append(PretrainRecord.model_validate_json(line))
                        discard_count += 1
            console.print(f"[cyan]✔ Loaded {discard_count} discard records from {discard_file.name}[/cyan]")
        except (json.JSONDecodeError, ValidationError) as e:
            console.print(f"[red]⚠ Could not parse existing discard file: {e}. Ignoring discards.[/red]")
    else:
        console.print(f"[cyan]📝 Will create new discard file: {discard_file.name}[/cyan]")
    
    if existing_records:
        total_existing = len(existing_records)
        console.print(f"[cyan]� Resume mode: Found {total_existing} total existing records ({keep_count} keeps + {discard_count} discards). Will skip processing for these.[/cyan]")
    else:
        console.print(f"[cyan]🆕 Fresh start: No existing records found[/cyan]")

    console.print()
    allr = process_all_csvs(csvs, f, w, {}, save_interval, o, base, pdf_mapping, existing_records=existing_records)

    if not allr:
        console.print(f"[red]⚠ No valid records found across all CSVs[/red]")
        return

    total = len(allr)
    # Filter for records with status "keep"
    kept = sum(1 for r in allr if r.meta.status == "keep")
    ratio = round(kept / total * 100, 2) if total > 0 else 0
    
    # Check if these are existing records or newly processed ones
    existing_count = len(existing_records) if existing_records else 0
    new_records_count = total - existing_count
    
    console.print("[bold green]═══ Final Summary ═══[/bold green]")
    if new_records_count > 0:
        console.print(f"[white]Total records processed this session:[/white] {new_records_count}")
        console.print(f"[dim]Total records in dataset (including existing):[/dim] {total}")
    else:
        console.print(f"[cyan]✔ All processing complete - no new records processed this session[/cyan]")
        console.print(f"[white]Total records in dataset:[/white] {total}")
    
    console.print(f"[green]Kept:[/green] {kept} [cyan]({ratio}%)[/cyan]")
    console.print(f"[yellow]Discarded:[/yellow] {total - kept}")
    console.print()

    # The final save is now handled by the incremental saver.
    # We can optionally write a pretty-printed JSON for inspection.
    if pretty:
        jsonl_dir = o / "jsonl"
        jsonl_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        j2 = jsonl_dir / f"{base}.json"
        with j2.open("w", encoding="utf-8") as writer:
            json.dump([x.model_dump() for x in allr], writer, ensure_ascii=False, indent=2)
        console.print(f"[cyan]• Pretty JSON → {j2.name}[/cyan]")


def concatenate_small_segments(spans: list[str], source_mapping: list[str], 
                             min_length: int = 64, max_length: int = 512) -> tuple[list[str], list[str]]:
    """
    Iteratively concatenate small segments within the same document until reaching acceptable length.
    Respects max_length boundary - never exceeds it during concatenation.
    
    Args:
        spans: List of text segments  
        source_mapping: List of source files corresponding to each segment
        min_length: Minimum segment length to keep standalone (default: 64)
        max_length: Maximum length after concatenation (default: 512)
    
    Returns:
        Tuple of (concatenated_spans, updated_source_mapping)
    """
    if not spans:
        return spans, source_mapping
    
    concatenated_spans = []
    concatenated_sources = []
    concatenated_count = 0
    
    i = 0
    while i < len(spans):
        current_text = spans[i].strip()
        current_source = source_mapping[i]
        
        # If segment is too small, try iterative concatenation
        if len(current_text) < min_length and i < len(spans) - 1:
            combined_text = current_text
            segments_used = 1
            j = i + 1
            
            # Iteratively add subsequent segments from same document
            while j < len(spans) and source_mapping[j] == current_source:
                
                next_segment = spans[j].strip()
                
                # Don't concatenate with segments that are already long enough by themselves
                if len(next_segment) >= min_length:
                    break
                
                potential_combined = combined_text + " " + next_segment
                
                # RESPECT MAX LENGTH - don't exceed during concatenation
                if len(potential_combined) <= max_length:
                    combined_text = potential_combined
                    segments_used += 1
                    j += 1
                    
                    # Stop if we've reached a good length
                    if len(combined_text) >= min_length:
                        break
                else:
                    # Would exceed max_length, stop concatenation
                    break
            
            concatenated_spans.append(combined_text)
            concatenated_sources.append(current_source)
            
            if segments_used > 1:
                concatenated_count += segments_used - 1  # Count extra segments merged
            
            i = j  # Skip the segments we just concatenated
        else:
            # Segment is long enough or is last segment - keep as-is
            concatenated_spans.append(current_text)
            concatenated_sources.append(current_source)
            i += 1
    
    if concatenated_count > 0:
        console.print(f"[cyan]🔗 Concatenated {concatenated_count} small segments (min: {min_length} chars, max: {max_length} chars)[/cyan]")
    
    return concatenated_spans, concatenated_sources


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
    """Count the number of pages in a PDF file using pypdf."""
    try:
        import pypdf
        with pdf_path.open('rb') as file:
            reader = pypdf.PdfReader(file)
            return len(reader.pages)
    except ImportError:
        # pypdf not available, return unknown count
        return -1
    except Exception as e:
        # If there's any error reading the PDF, return unknown count silently
        return -1


def split_large_pdf(pdf_path: Path, temp_dir: Path, pages_per_batch: int = 200) -> list[Path]:
    """
    Split a large PDF into smaller batches for processing.
    Returns list of temporary PDF file paths.
    """
    try:
        import pypdf
        
        with pdf_path.open('rb') as file:
            reader = pypdf.PdfReader(file)
            total_pages = len(reader.pages)
            
            if total_pages <= pages_per_batch:
                # No need to split, return original
                return [pdf_path]
            
            console.print(f"[cyan]📄 Splitting {pdf_path.name} ({total_pages} pages) into batches of {pages_per_batch} pages...[/cyan]")
            
            # Create temporary directory for batches
            pdf_stem = pdf_path.stem
            batch_dir = temp_dir / f"{pdf_stem}_batches"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            batch_files = []
            num_batches = (total_pages + pages_per_batch - 1) // pages_per_batch
            
            for batch_idx in range(num_batches):
                start_page = batch_idx * pages_per_batch
                end_page = min(start_page + pages_per_batch, total_pages)
                
                # Create batch filename with suffix
                batch_filename = f"{pdf_stem}_batch_{batch_idx + 1:02d}.pdf"
                batch_path = batch_dir / batch_filename
                
                # Create new PDF with pages for this batch
                writer = pypdf.PdfWriter()
                for page_idx in range(start_page, end_page):
                    writer.add_page(reader.pages[page_idx])
                
                with batch_path.open('wb') as output_file:
                    writer.write(output_file)
                
                batch_files.append(batch_path)
                console.print(f"[dim]  Created batch {batch_idx + 1}/{num_batches}: {batch_filename} (pages {start_page + 1}-{end_page})[/dim]")
            
            console.print(f"[green]✔ Split into {len(batch_files)} batches[/green]")
            return batch_files
            
    except ImportError:
        console.print(f"[yellow]⚠ pypdf not available, cannot split large PDFs. Processing as single file.[/yellow]")
        return [pdf_path]
    except Exception as e:
        console.print(f"[yellow]⚠ Error splitting PDF {pdf_path.name}: {e}. Processing as single file.[/yellow]")
        return [pdf_path]


def aggregate_batch_results(original_pdf: Path, batch_results: list[tuple[Path, Path]], final_csv_dir: Path) -> Optional[Path]:
    """
    Aggregate results from multiple PDF batches back into a single result structure.
    batch_results: list of (batch_pdf_path, csv_result_path) tuples
    Returns the path to the aggregated CSV file.
    """
    if not batch_results:
        return None
    
    # If only one batch, just move/copy the result
    if len(batch_results) == 1:
        _, csv_path = batch_results[0]
        if csv_path:
            original_hash = hash_name(original_pdf)
            final_csv_name = f"{original_hash}.csv"
            final_csv_path = final_csv_dir / final_csv_name
            
            # Copy the single batch result to final location
            try:
                import shutil
                shutil.copy2(csv_path, final_csv_path)
                console.print(f"[green]✔ Single batch result ready: {final_csv_name}[/green]")
                return final_csv_path
            except Exception as e:
                console.print(f"[yellow]⚠ Error copying single batch result: {e}[/yellow]")
                return csv_path
        return None
    
    console.print(f"[cyan]🔗 Aggregating results from {len(batch_results)} batches for {original_pdf.name}...[/cyan]")
    
    # Aggregate multiple batch results
    successful_batches = [(batch_pdf, csv_path) for batch_pdf, csv_path in batch_results if csv_path and csv_path.exists()]
    
    if not successful_batches:
        console.print(f"[red]⚠ No successful batches to aggregate for {original_pdf.name}[/red]")
        return None
    
    try:
        # Read all CSV files and combine them
        all_rows = []
        headers = None
        
        for batch_pdf, csv_path in successful_batches:
            try:
                with csv_path.open('r', encoding='utf-8', newline='') as f:
                    reader = csv.reader(f)
                    batch_headers = next(reader, None)
                    
                    if headers is None:
                        headers = batch_headers
                    elif headers != batch_headers:
                        console.print(f"[yellow]⚠ Header mismatch in batch {batch_pdf.name}, using first batch headers[/yellow]")
                    
                    # Read all rows from this batch
                    batch_rows = list(reader)
                    all_rows.extend(batch_rows)
                    console.print(f"[dim]  Added {len(batch_rows)} rows from {batch_pdf.name}[/dim]")
                    
            except Exception as e:
                console.print(f"[yellow]⚠ Error reading batch {csv_path}: {e}[/yellow]")
                continue
        
        if not all_rows:
            console.print(f"[red]⚠ No data to aggregate for {original_pdf.name}[/red]")
            return None
        
        # Write aggregated CSV
        original_hash = hash_name(original_pdf)
        final_csv_name = f"{original_hash}.csv"
        final_csv_path = final_csv_dir / final_csv_name
        
        with final_csv_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(all_rows)
        
        console.print(f"[green]✔ Aggregated {len(all_rows)} total rows into: {final_csv_name}[/green]")
        
        # Update JSON metadata to reflect original PDF name
        add_pdf_name_to_json_metadata(final_csv_path, original_pdf)
        
        return final_csv_path
        
    except Exception as e:
        console.print(f"[red]⚠ Error aggregating batch results for {original_pdf.name}: {e}[/red]")
        return None


def cleanup_batch_files(batch_files: list[Path]):
    """Clean up temporary batch files and directories."""
    if not batch_files:
        return
    
    try:
        # Get unique batch directories
        batch_dirs = set()
        for batch_file in batch_files:
            if batch_file.exists():
                batch_dirs.add(batch_file.parent)
                batch_file.unlink()
        
        # Remove empty batch directories
        for batch_dir in batch_dirs:
            try:
                if batch_dir.exists() and batch_dir.is_dir():
                    # Remove any remaining files
                    for file in batch_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                    # Remove the directory
                    batch_dir.rmdir()
                    console.print(f"[dim]Cleaned up batch directory: {batch_dir.name}[/dim]")
            except OSError:
                # Directory not empty or other issue, leave it
                pass
                
    except Exception as e:
        console.print(f"[yellow]⚠ Error during batch cleanup: {e}[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="X-Spanformer PDF2JSONL Pipeline")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input directory of PDF files or a single PDF file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory for results")
    parser.add_argument("-f", "--field", type=str, default="text", help="Field name in CSV to process")
    parser.add_argument("-p", "--pretty", action="store_true", help="Output pretty JSON file")
    parser.add_argument("-n", "--name", type=str, default="dataset", help="Base name for output files")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--save-interval", type=int, default=1, help="Save progress every N records. Default is 1 (save every record). Use 0 to disable incremental saving.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all cached data")
    args = parser.parse_args()

    run(args.input, args.output, args.field, args.pretty, args.name, args.workers, args.save_interval, args.force)


if __name__ == "__main__":
    main()