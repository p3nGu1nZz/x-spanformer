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


def process_all_csvs(csv_files: list[Path], col: str, w: int, cfg: dict, save_interval: int = 1, output_path: Optional[Path] = None, base_name: str = "dataset", pdf_mapping: Optional[dict[str, str]] = None, existing_records: Optional[list[PretrainRecord]] = None) -> list[PretrainRecord]:
    if not csv_files:
        console.print(f"[red]⚠ No CSV files provided[/red]")
        return []

    if pdf_mapping is None:
        pdf_mapping = {}

    # Ensure deterministic processing order
    csv_files = sorted(csv_files)
    
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
    total_segment_count = len(spans)  # Initialize with original count, will be updated after splitting

    def save_chunk(records: list[PretrainRecord]):
        """Save a chunk of records incrementally to prevent data loss"""
        nonlocal records_saved_this_session

        if not output_path or not records:
            return

        # Separate records by status
        keep_records = [record for record in records if record.meta.status == "keep"]
        discard_records = [record for record in records if record.meta.status == "discard"]
        
        # Create jsonl directory for dataset files
        jsonl_dir = output_path / "jsonl"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        # Save "keep" records to main dataset file
        if keep_records:
            dataset_file = jsonl_dir / f"{base_name}.jsonl"
            with dataset_file.open("a", encoding="utf-8") as writer:
                for record in keep_records:
                    writer.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            saved_count += len(keep_records)
            console.print(f"[blue]💾 Saved {len(keep_records)} keep record(s) to {dataset_file.name}[/blue]")
        
        # Save "discard" records to discard file
        if discard_records:
            discard_file = jsonl_dir / "discard.jsonl"
            with discard_file.open("a", encoding="utf-8") as writer:
                for record in discard_records:
                    writer.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            console.print(f"[blue]💾 Saved {len(discard_records)} discard record(s) to {discard_file.name}[/blue]")
        
        records_saved_this_session += saved_count
        total_saved = len(keep_records) + len(discard_records)
        console.print(f"[blue]💾 Total saved this chunk: {total_saved} records (session total: {records_saved_this_session} keeps)[/blue]")

    async def process():
        stats = Counter()
        all_processed_recs = []
        records_to_save = []
        recs, reasons = [], []
        # Initialize counters, including already-processed records if resuming
        processed_count_total = len(existing_records) if existing_records else 0
        # Count existing valid records (status "keep") toward valid count
        valid_records_count = sum(1 for r in existing_records if r.meta.status == "keep") if existing_records else 0

        # Initialize sessions once, sharing the same config
        agent_config = load_judge_config()

        # Pre-process spans to split long texts
        max_raw_length = agent_config.get("processor", {}).get("max_raw_length", 512)
        expanded_spans = []
        expanded_source_mapping = []

        for i, (text, source_file) in enumerate(zip(spans, source_mapping)):
            if len(text) > max_raw_length:
                text_chunks = split_long_text(text, max_raw_length)
            else:
                text_chunks = [text]
            
            for chunk in text_chunks:
                expanded_spans.append(chunk)
                expanded_source_mapping.append(source_file)

        # Save the total number of segments before filtering (for progress tracking)
        total_segments_including_processed = len(expanded_spans)

        if existing_records:
            processed_raws = {rec.raw for rec in existing_records}
            keep_raws = {rec.raw for rec in existing_records if rec.meta.status == "keep"}
            discard_raws = {rec.raw for rec in existing_records if rec.meta.status == "discard"}
            
            if processed_raws:
                unprocessed_spans = []
                unprocessed_source_mapping = []
                skipped_keeps = 0
                skipped_discards = 0
                
                for text, source in zip(expanded_spans, expanded_source_mapping):
                    if text not in processed_raws:
                        unprocessed_spans.append(text)
                        unprocessed_source_mapping.append(source)
                    else:
                        # Count what type of record we're skipping
                        if text in keep_raws:
                            skipped_keeps += 1
                        elif text in discard_raws:
                            skipped_discards += 1

                if len(unprocessed_spans) < len(expanded_spans):
                    skipped_count = len(expanded_spans) - len(unprocessed_spans)
                    console.print(f"[cyan]🔄 Resuming: Skipped {skipped_count} already processed segments ({skipped_keeps} keeps + {skipped_discards} discards). Total processed so far: {processed_count_total}[/cyan]")
                    expanded_spans = unprocessed_spans
                    expanded_source_mapping = unprocessed_source_mapping
                else:
                    console.print("[yellow]No matching segments found in existing records. Processing all from scratch.[/yellow]")

        if not expanded_spans:
            console.print("[green]✔ All segments have already been processed. Nothing to do.[/green]")
            return []

        if len(expanded_spans) != len(spans):
            console.print(f"[cyan]📐 Text splitting: {len(spans)} → {len(expanded_spans)} segments to process (max length: {max_raw_length} chars)[/cyan]")

        # Update references to use total segments including already processed
        nonlocal total_segment_count
        total_segment_count = total_segments_including_processed  # Use total including already processed
        nonlocal estimated_total_saves
        estimated_total_saves = total_segment_count

        # Get judge configuration
        num_judges = agent_config.get("judge", {}).get("judges", 5)
        threshold = agent_config.get("judge", {}).get("threshold", 0.69)
        max_retries = agent_config.get("judge", {}).get("max_retries", 3)
        
        console.print(f"[bold magenta]🚀 Using {w} concurrent workers with {num_judges} judges each (sequential per worker)[/bold magenta]")
        console.print(f"[yellow]💡 Sequential judge processing reduces Ollama queue pressure for better stability![/yellow]")
        
        # Initialize sessions once, sharing the same config
        from x_spanformer.agents.session.judge_session import JudgeSession
        sessions = [JudgeSession(config=agent_config, quiet=True) for _ in range(w)]
        
        # Use asyncio.Semaphore for concurrency control
        semaphore = asyncio.Semaphore(w)
        consecutive_retry_errors = 0  # Track consecutive RetryError failures
        
        async def judge_segment(idx, text, source_file):
            nonlocal consecutive_retry_errors
            async with semaphore:
                try:
                    session = sessions[idx % len(sessions)]
                    judge_responses = []
                    
                    # Process judges sequentially - let tenacity handle retries
                    for judge_num in range(num_judges):
                        try:
                            console.print(f"[dim]⚖️ Judging text (len={len(text)}): {text[:80]}...[/dim]")
                            judge_result = await asyncio.wait_for(
                                session.evaluate(text), 
                                timeout=30.0
                            )
                            judge_responses.append(judge_result)
                            # Reset consecutive error counter on any successful evaluation
                            consecutive_retry_errors = 0
                            
                        except Exception as e:
                            error_str = str(e)
                            
                            # Check for RetryError from tenacity exhaustion
                            if "RetryError" in error_str and "ConnectionError" in error_str:
                                consecutive_retry_errors += 1
                                console.print(f"[red]❌ RetryError for segment {idx} judge {judge_num + 1} (consecutive: {consecutive_retry_errors}): {error_str}[/red]")
                                
                                # Exit after 3 consecutive RetryErrors across any judges
                                if consecutive_retry_errors >= 3:
                                    console.print(f"[bold red]🚫 CRITICAL: {consecutive_retry_errors} consecutive RetryErrors - Ollama connection unstable![/bold red]")
                                    raise Exception(f"CRITICAL: {consecutive_retry_errors} consecutive RetryErrors - Ollama connection unstable")
                                
                                # RetryError means tenacity has exhausted all retries for this judge
                                # This judge failed completely, so we continue to next judge
                                console.print(f"[red]❌ Judge {judge_num + 1} failed after all retries - continuing to next judge[/red]")
                                continue
                            else:
                                # For other errors (timeouts, etc.), increment counter and exit immediately
                                consecutive_retry_errors += 1
                                console.print(f"[red]❌ Critical error in judge {judge_num + 1} (consecutive: {consecutive_retry_errors}): {str(e)}[/red]")
                                raise Exception(f"CRITICAL: Judge evaluation error for segment {idx}: {str(e)}")
                    
                    # Check if we got any successful judge responses
                    if not judge_responses:
                        console.print(f"[bold red]🚫 CRITICAL: All judges failed for segment {idx} - system failure detected![/bold red]")
                        raise Exception(f"CRITICAL: No successful judge responses for segment {idx} - all judges failed")
                    
                    # Calculate consensus from available responses
                    scores = [r.get("score", 0) for r in judge_responses]
                    consensus_score = sum(scores) / len(scores)
                    
                    content_types = [r.get("type", "natural") for r in judge_responses]
                    consensus_type = max(set(content_types), key=content_types.count)
                    
                    final_status = "keep" if consensus_score >= threshold else "discard"
                    
                    all_reasons = [r.get("reason", "") for r in judge_responses]
                    combined_reason = " / ".join(all_reasons)
                    
                    return idx, {
                        "score": consensus_score,
                        "status": final_status,
                        "type": consensus_type,
                        "reason": combined_reason
                    }, {
                        "text": text,
                        "source_file": source_file,
                        "judge_responses": judge_responses
                    }
                    
                except Exception as e:
                    error_str = str(e)
                    # All critical errors should bubble up and stop processing
                    console.print(f"[bold red]🚫 CRITICAL FAILURE: {error_str}[/bold red]")
                    raise e
        
        # Group segments by source file to process documents sequentially
        segments_by_source = {}
        for i, (text, source_file) in enumerate(zip(expanded_spans, expanded_source_mapping)):
            if source_file not in segments_by_source:
                segments_by_source[source_file] = []
            segments_by_source[source_file].append((i, text))
        
        console.print(f"[cyan]📋 Processing {len(segments_by_source)} documents sequentially, with concurrent judge evaluation within each document[/cyan]")
        
        # Debug: Show first few segments to verify resume logic
        if len(expanded_spans) < 20:
            console.print(f"[dim]Debug: First few segments to process:[/dim]")
            for i, text in enumerate(expanded_spans[:5]):
                console.print(f"[dim]  {i+1}: {text[:60]}...[/dim]")
        
        # Process each document sequentially
        overall_completed_count = len(existing_records) if existing_records else 0  # Start from already processed count
        session_start_count = overall_completed_count  # Remember where we started this session
        overall_keep_count = valid_records_count  # Start from already kept records count
        overall_discard_count = len(existing_records) - valid_records_count if existing_records else 0  # Start from already discarded records count
        results = []  # Track all results for final return
        
        for doc_idx, (source_file, segments_list) in enumerate(segments_by_source.items(), 1):
            console.print(f"[bold yellow]📄 Processing Document {doc_idx}/{len(segments_by_source)}: {source_file}[/bold yellow]")
            console.print(f"[dim]   {len(segments_list)} segments in this document[/dim]")
            
            # Create tasks for this document only
            doc_tasks = [
                judge_segment(original_idx, text, source_file) 
                for original_idx, text in segments_list
            ]
            
            # Process this document's segments concurrently
            doc_results = []
            doc_completed_count = 0
            
            for task in asyncio.as_completed(doc_tasks):
                idx, r, log_data = await task
                doc_results.append((idx, r, log_data))
                doc_completed_count += 1
                overall_completed_count += 1
                
                # Process this result immediately for incremental saving
                tag = r["status"]
                reasons.append(r["reason"])
                stats[tag] += 1

                # Handle AI processing log 
                if log_data and output_path:
                    save_ai_processing_log(
                        output_path,
                        log_data["source_file"],
                        str(idx),
                        log_data["text"],
                        log_data["judge_responses"],
                        r
                    )

                src_file = expanded_source_mapping[idx]
                original_text = expanded_spans[idx]

                record = PretrainRecord(
                    raw=original_text,
                    type=r.get("type", "natural"),
                    meta=RecordMeta(
                        source_file=src_file,
                        doc_language="en",
                        extracted_by="pdf2seg",
                        confidence=r.get("score"),
                        status=tag,
                        tags=[tag] if tag != "keep" else [],
                        notes=r.get('reason', '')
                    )
                )
                
                all_processed_recs.append(record)
                processed_count_total += 1

                # Incremental saving - save immediately after each record
                if save_interval > 0:
                    records_to_save.append(record)
                    if len(records_to_save) >= save_interval:
                        save_chunk(records_to_save)
                        records_to_save.clear()

                # Only count valid records (status "keep") toward valid count
                if tag == "keep":
                    valid_records_count += 1
                    overall_keep_count += 1  # Track overall kept records
                elif tag == "discard":
                    overall_discard_count += 1  # Track overall discarded records
                
                # Display individual judgment result for every segment
                if log_data:
                    display_judgment_result(
                        idx=overall_completed_count - 1,  # Use sequential progress counter (0-based)
                        text=log_data["text"], 
                        status=r["status"],
                        score=r["score"],
                        reason=r["reason"],
                        content_type=r.get("type", "natural"),
                        total_count=total_segment_count  # Use total including already processed
                    )
                
                # Show progress summary every 10 completed or at end of document
                if doc_completed_count % 10 == 0 or doc_completed_count == len(doc_tasks):
                    keep_count = sum(1 for _, result, _ in doc_results if result["status"] == "keep")
                    discard_count = doc_completed_count - keep_count
                    console.print(f"[bright_blue]📊 Document Progress: {doc_completed_count}/{len(doc_tasks)} | [green]✅ Keep: {keep_count}[/green] | [red]❌ Discard: {discard_count}[/red] | Overall: {overall_completed_count}/{total_segment_count}[/bright_blue]")
                    console.print()
                    
                    # Display telemetry panel after progress summary
                    if overall_completed_count > 0:  # Only show if we have processed some records
                        # Calculate estimated total saves based on current keep rate
                        current_keep_rate = overall_keep_count / overall_completed_count if overall_completed_count > 0 else 0.5
                        estimated_total_keeps = int(total_segment_count * current_keep_rate)
                        
                        display_telemetry_panel(
                            processed_count=overall_completed_count,
                            total_count=total_segment_count,  # Use total including already processed
                            start_time=start_time,
                            save_count=overall_keep_count,  # Total saved records (all keeps)
                            estimated_total_saves=estimated_total_keeps,  # Estimated based on current keep rate
                            records_saved_this_session=records_saved_this_session,
                            keep_count=overall_keep_count,
                            session_start_count=session_start_count  # Pass the session start count
                        )
            
            # Sort this document's results by original index to maintain order within document
            doc_results.sort(key=lambda x: x[0])
            results.extend(doc_results)
            
            console.print(f"[green]✅ Completed Document {doc_idx}/{len(segments_by_source)}: {source_file}[/green]")
            console.print()
        
        # Save any remaining records in the buffer
        if save_interval > 0 and records_to_save:
            console.print(f"[blue]Finalizing... saving {len(records_to_save)} remaining records.[/blue]")
            save_chunk(records_to_save)
            records_to_save.clear()

        return all_processed_recs, stats, reasons

    try:
        result = asyncio.run(process())
        if result is None:
            console.print("[yellow]⚠ Processing returned no results[/yellow]")
            return []
        all_recs, stats, reasons = result
        display_summary_panel("Combined CSV files", stats, reasons)
        return all_recs
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting.[/bold red]")
        return []
    except Exception as e:
        error_str = str(e)
        if ("consecutive RetryErrors" in error_str or 
            ("failed after" in error_str and "retries" in error_str) or
            "Segment evaluation timeout" in error_str or
            "Hard processing error" in error_str or
            "all judges failed" in error_str):
            console.print(f"\n[bold red]❌ Critical system failure: {error_str}[/bold red]")
            console.print(f"[red]🚫 Exiting program immediately. Please check system status and try again.[/red]")
            sys.exit(1)  # Exit immediately on critical system failures
        else:
            console.print(f"\n[bold red]❌ Processing failed: {error_str}[/bold red]")
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

    console.print("[bold green]═══ Final Summary ═══[/bold green]")
    console.print(f"[white]Total records processed this session:[/white] {total}")
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