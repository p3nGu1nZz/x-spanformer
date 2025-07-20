"""
Shared CSV processing logic for judge evaluation.
Extracted from pdf2jsonl.py for reuse across pipelines.
"""
import asyncio
import json
import sys
import time
import re
from pathlib import Path
from typing import Optional, List
from collections import Counter
import pandas as pd
from pydantic import ValidationError

from x_spanformer.agents.config_loader import load_judge_config
from x_spanformer.agents.rich_utils import (
    console,
    display_summary_panel,
    display_telemetry_panel,
    display_judgment_result,
)
from x_spanformer.agents.session.judge_session import JudgeSession
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.pretrain_record import PretrainRecord

def process_all_csvs(csv_files: list[Path], col: str, w: int, cfg: dict, 
                    save_interval: int = 1, output_path: Optional[Path] = None, 
                    base_name: str = "dataset", pdf_mapping: Optional[dict[str, str]] = None, 
                    existing_records: Optional[list[PretrainRecord]] = None) -> list[PretrainRecord]:
    """
    Process CSV files through judge evaluation pipeline.
    
    This is the core logic extracted from pdf2jsonl.py process_all_csvs function
    for reuse in other pipelines like repo2jsonl.py.
    """
    if not csv_files:
        console.print(f"[red]⚠ No CSV files provided[/red]")
        return []

    if pdf_mapping is None:
        pdf_mapping = {}

    # Ensure deterministic processing order
    csv_files = sorted(csv_files)
    
    all_dfs = []
    source_files = []

    # Load and combine CSV files
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if col in df.columns:
                original_source_name = pdf_mapping.get(csv_file.name, csv_file.name)
                df['source_file'] = original_source_name
                all_dfs.append(df)
                source_files.append(original_source_name)
                console.print(f"[green]✔ Loaded {len(df)} rows from {csv_file.name} (source: {original_source_name})[/green]")
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
        console.print(f"[cyan]💾 Incremental saving enabled: saving after every {save_interval} record(s)[/cyan]")
    else:
        console.print("[yellow]💾 Incremental saving disabled. Results will be processed in memory.[/yellow]")
    console.print()

    # Initialize timing and tracking
    start_time = time.time()
    records_saved_this_session = 0
    total_segment_count = len(spans)

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
            console.print(f"[blue]💾 Saved {len(discard_records)} discard record(s) to discard.jsonl[/blue]")
        
        records_saved_this_session += saved_count
        total_saved = len(keep_records) + len(discard_records)
        console.print(f"[blue]💾 Total saved this chunk: {total_saved} records (session total: {records_saved_this_session} keeps)[/blue]")

    async def process():
        stats = Counter()
        all_processed_recs = []
        records_to_save = []
        reasons = []
        
        # Initialize counters
        processed_count_total = len(existing_records) if existing_records else 0
        valid_records_count = sum(1 for r in existing_records if r.meta.status == "keep") if existing_records else 0

        # Load agent configuration
        agent_config = load_judge_config()

        # Pre-process spans for text length management
        max_raw_length = agent_config.get("processor", {}).get("max_raw_length", 512)
        min_raw_length = agent_config.get("processor", {}).get("min_raw_length", 64)
        
        expanded_spans = []
        expanded_source_mapping = []

        for text, source_file in zip(spans, source_mapping):
            if len(text) > max_raw_length:
                text_chunks = split_long_text(text, max_raw_length)
            else:
                text_chunks = [text]
            
            for chunk in text_chunks:
                expanded_spans.append(chunk)
                expanded_source_mapping.append(source_file)

        # Handle existing records (resume functionality)
        if existing_records:
            processed_raws = {rec.raw for rec in existing_records}
            if processed_raws:
                unprocessed_spans = []
                unprocessed_source_mapping = []
                skipped_count = 0
                
                for text, source in zip(expanded_spans, expanded_source_mapping):
                    if text not in processed_raws:
                        unprocessed_spans.append(text)
                        unprocessed_source_mapping.append(source)
                    else:
                        skipped_count += 1

                if skipped_count > 0:
                    console.print(f"[cyan]🔄 Resume: Skipped {skipped_count} already processed segments[/cyan]")
                    expanded_spans = unprocessed_spans
                    expanded_source_mapping = unprocessed_source_mapping

        if not expanded_spans:
            console.print("[green]✔ All segments have already been processed.[/green]")
            return existing_records if existing_records else [], Counter(), []

        # Update total count after processing
        nonlocal total_segment_count
        total_segment_count = len(expanded_spans) + processed_count_total

        # Initialize judge sessions
        sessions = [JudgeSession(config=agent_config, quiet=True) for _ in range(w)]
        
        # Judge configuration
        num_judges = agent_config.get("judge", {}).get("judges", 5)
        threshold = agent_config.get("judge", {}).get("threshold", 0.69)
        
        console.print(f"[bold magenta]🚀 Using {w} concurrent workers with {num_judges} judges each[/bold magenta]")

        # Process segments
        semaphore = asyncio.Semaphore(w)
        
        async def judge_segment(idx, text, source_file):
            async with semaphore:
                session = sessions[idx % len(sessions)]
                judge_responses = []
                
                # Process judges sequentially for stability
                for judge_num in range(num_judges):
                    try:
                        judge_result = await asyncio.wait_for(
                            session.evaluate(text), 
                            timeout=30.0
                        )
                        judge_responses.append(judge_result)
                    except Exception as e:
                        console.print(f"[red]❌ Judge {judge_num + 1} failed for segment {idx}: {e}[/red]")
                        continue
                
                if not judge_responses:
                    return idx, {
                        "score": 0.0,
                        "status": "discard", 
                        "type": "natural",
                        "reason": "all judges failed"
                    }, {"text": text, "source_file": source_file, "judge_responses": []}
                
                # Calculate consensus
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

        # Create and run tasks
        tasks = [
            judge_segment(idx, text, source_file) 
            for idx, (text, source_file) in enumerate(zip(expanded_spans, expanded_source_mapping))
        ]

        # Process results
        completed_count = 0
        session_start_count = processed_count_total
        overall_keep_count = valid_records_count

        for task in asyncio.as_completed(tasks):
            idx, r, log_data = await task
            completed_count += 1
            processed_count_total += 1

            # Process result
            tag = r["status"]
            reasons.append(r["reason"])
            stats[tag] += 1

            # Create record
            src_file = expanded_source_mapping[idx]
            original_text = expanded_spans[idx]

            record = PretrainRecord(
                raw=original_text,
                type=r.get("type", "natural"),
                meta=RecordMeta(
                    source_file=src_file,
                    doc_language="en",
                    extracted_by="shared_processor",  # Will be overridden by calling pipeline
                    confidence=r.get("score"),
                    status=tag,
                    tags=[tag] if tag != "keep" else [],
                    notes=r.get('reason', '')
                )
            )
            
            all_processed_recs.append(record)

            # Incremental saving
            if save_interval > 0:
                records_to_save.append(record)
                if len(records_to_save) >= save_interval:
                    save_chunk(records_to_save)
                    records_to_save.clear()

            # Update counters
            if tag == "keep":
                overall_keep_count += 1

            # Display progress
            display_judgment_result(
                idx=completed_count - 1,
                text=log_data["text"], 
                status=r["status"],
                score=r["score"],
                reason=r["reason"],
                content_type=r.get("type", "natural"),
                total_count=len(expanded_spans)
            )

            # Show progress every 10 items
            if completed_count % 10 == 0 or completed_count == len(expanded_spans):
                display_telemetry_panel(
                    processed_count=processed_count_total,
                    total_count=total_segment_count,
                    start_time=start_time,
                    save_count=overall_keep_count,
                    estimated_total_saves=int(total_segment_count * 0.5),  # Rough estimate
                    records_saved_this_session=records_saved_this_session,
                    keep_count=overall_keep_count,
                    session_start_count=session_start_count
                )

        # Save any remaining records
        if save_interval > 0 and records_to_save:
            save_chunk(records_to_save)

        return all_processed_recs, stats, reasons

    # Run the async processing
    try:
        result = asyncio.run(process())
        if result is None:
            return []
        
        all_recs, stats, reasons = result
        display_summary_panel("CSV Processing", stats, reasons)
        return all_recs
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting.[/bold red]")
        return []
    except Exception as e:
        console.print(f"\n[bold red]❌ Processing failed: {e}[/bold red]")
        return []

def split_long_text(text: str, max_length: int = 512) -> list[str]:
    """Split text that exceeds max_length into smaller chunks."""
    if len(text) <= max_length:
        return [text]

    # Split on newlines first for code
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # If any chunk is still too long, split by characters
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            for i in range(0, len(chunk), max_length):
                final_chunks.append(chunk[i:i+max_length])
        else:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else [""]
