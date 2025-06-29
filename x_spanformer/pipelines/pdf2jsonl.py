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

try:
    import spacy
except ImportError:
    raise ImportError("spaCy is required for text splitting. Please install spaCy: pip install spacy")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.agent_utils import (
    console,
    display_summary_panel,
    display_telemetry_panel,
)
from x_spanformer.agents.session import JudgeSession, ImproveSession
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


def process_all_csvs(csv_files: list[Path], col: str, w: int, cfg: dict, save_interval: int = 10, output_path: Optional[Path] = None, base_name: str = "dataset", pdf_mapping: Optional[dict[str, str]] = None) -> list[PretrainRecord]:
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
    if save_interval > 0 and output_path:
        console.print(f"[cyan]Incremental saving enabled: every {save_interval} valid records (status='keep') to {output_path}[/cyan]")
    console.print()

    # Initialize timing metrics
    start_time = time.time()
    # Note: We can't predict how many valid records we'll have, so estimated_total_saves will be updated after text splitting
    estimated_total_saves = 1  # Placeholder, will be updated after text splitting
    records_saved_this_session = 0
    total_segment_count = len(spans)  # Initialize with original count, will be updated after splitting

    def save_incremental(records: list[PretrainRecord], valid_count: int, total_processed: int, force: bool = False):
        """Save records incrementally to prevent data loss"""
        nonlocal records_saved_this_session

        if not output_path or not records:
            return

        output_path.mkdir(parents=True, exist_ok=True)
        dataset_file = output_path / f"{base_name}.jsonl"

        # For the first save, check if we should start fresh or append
        mode = "a" if dataset_file.exists() else "w"

        with dataset_file.open(mode, encoding="utf-8") as writer:
            for record in records:
                writer.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

        records_saved_this_session += len(records)
        save_count = (valid_count + save_interval - 1) // save_interval if save_interval > 0 else 1

        console.print(f"[blue]💾 Saved {len(records)} records to {dataset_file.name} (triggered by {valid_count} valid records)[/blue]")

        # Display telemetry panel
        display_telemetry_panel(
            processed_count=total_processed,
            total_count=total_segment_count,
            start_time=start_time,
            save_count=save_count,
            estimated_total_saves=estimated_total_saves,
            records_saved_this_session=records_saved_this_session
        )

    async def process():
        sem = asyncio.Semaphore(w)
        stats = Counter()
        all_processed_recs = []
        recs, reasons = [], []
        processed_count = 0
        valid_records_count = 0  # Count only records with status "keep"

        # Initialize sessions once, sharing the same config
        agent_config = load_selfcrit_config()
        judge_session = JudgeSession(config=agent_config, quiet=True)
        improve_session = ImproveSession(config=agent_config, quiet=True)

        # Pre-process spans to split long texts
        max_raw_length = agent_config.get("processor", {}).get("max_raw_length", 512)
        expanded_spans = []
        expanded_source_mapping = []

        for i, (text, source_file) in enumerate(zip(spans, source_mapping)):
            text_chunks = split_long_text(text, max_raw_length)
            for chunk in text_chunks:
                expanded_spans.append(chunk)
                expanded_source_mapping.append(source_file)

        if len(expanded_spans) != len(spans):
            console.print(f"[cyan]📐 Text splitting: {len(spans)} → {len(expanded_spans)} segments (max length: {max_raw_length} chars)[/cyan]")

        # Update references to use expanded spans
        nonlocal total_segment_count
        total_segment_count = len(expanded_spans)
        nonlocal estimated_total_saves
        estimated_total_saves = max(1, (total_segment_count // 2 + save_interval - 1) // save_interval) if save_interval > 0 else 1  # Rough estimate assuming ~50% valid

        async def score_and_improve(idx: int, t: str, source_file: str):
            try:
                async with sem:
                    console.print(f"[bold blue]━━━ Processing segment {idx + 1}/{total_segment_count} ━━━[/bold blue]")
                    console.print(f"[dim]Source: {source_file} | Text ({len(t)} chars):[/dim] {t}")
                    console.print()

                    # Use JudgeSession to evaluate the text
                    r = await judge_session.evaluate(t)

                    console.print(f"[bold green]Initial Response:[/bold green]")
                    console.print(f"[white]{json.dumps(r, indent=2)}[/white]")
                    console.print()

                    # Check discard threshold - if score is below threshold, immediately discard
                    discard_threshold = agent_config.get("critique", {}).get("discard_threshold", 0.25)
                    current_score = r.get("score", 0)
                    
                    if current_score < discard_threshold:
                        console.print(f"[red]🗑️ Score {current_score:.3f} below discard threshold {discard_threshold:.3f}, marking as discard[/red]")
                        r["status"] = "discard"
                        r["reason"] = f"score {current_score:.3f} below discard threshold {discard_threshold:.3f}"
                        return idx, r, None, None, t, 0, t

                    improved_text = None
                    content_type = None
                    final_text = t
                    improvement_iterations = 0
                    max_improvements = 6

                    current_text = t
                    best_improved_text = None
                    best_content_type = None
                    best_score = current_score
                    best_result = r

                    # Only attempt improvement for segments with "revise" status
                    while (current_score < 0.7 and
                           r.get("status") == "revise" and
                           improvement_iterations < max_improvements):

                        console.print(f"[yellow]💡 Attempting improvement #{improvement_iterations + 1}/{max_improvements}...[/yellow]")
                        # Pass the judge's reason to help guide the improvement
                        judge_reason = r.get("reason", "")
                        temp_improved, temp_type = await improve_session.improve(current_text, reason=judge_reason)
                        improvement_iterations += 1

                        if temp_improved and temp_improved != current_text:
                            console.print(f"[yellow] Evaluating improved text...[/yellow]")
                            r_improved = await judge_session.evaluate(temp_improved)
                            improved_score = r_improved.get("score", 0)

                            console.print(f"[bold green]Improvement #{improvement_iterations} Response:[/bold green]")
                            console.print(f"[white]{json.dumps(r_improved, indent=2)}[/white]")

                            if improved_score > best_score:
                                # This is the best improvement so far
                                best_improved_text = temp_improved
                                best_content_type = temp_type
                                best_score = improved_score
                                best_result = r_improved

                                console.print(f"[green]✔ Improvement #{improvement_iterations} successful! New best score: {improved_score:.2f}[/green]")

                                # Update current for next iteration
                                current_text = temp_improved
                                current_score = improved_score
                                r = r_improved

                                if improved_score >= 0.7 and r_improved.get("status") == "keep":
                                    console.print(f"[green]🎉 Score threshold reached, stopping improvements[/green]")
                                    break
                            else:
                                console.print(f"[yellow]⚠ Improvement #{improvement_iterations} didn't help (score: {improved_score:.2f} vs {best_score:.2f})[/yellow]")
                        else:
                            console.print(f"[yellow]⚠ No improvement generated on attempt #{improvement_iterations}[/yellow]")

                    # Use the best improvement found, if any
                    if best_improved_text and best_improved_text != t:
                        improved_text = best_improved_text
                        content_type = best_content_type
                        final_text = best_improved_text
                        r = best_result

                    if improved_text and improved_text != t:
                        console.print(f"[green]📝 Final result: Using improved text (after {improvement_iterations} attempts)[/green]")
                    else:
                        improved_text = None
                        console.print(f"[blue]📝 Final result: Using original text (tried {improvement_iterations} improvements)[/blue]")

                    return idx, r, improved_text, content_type, final_text, improvement_iterations, t
            except Exception as e:
                # Escape potential Rich markup in error messages
                error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
                console.print(f"[red]Error processing segment {idx + 1}:[/red] {error_msg}")
                # Escape potential Rich markup in text preview
                safe_text = t[:100].replace('[', '\\[').replace(']', '\\]')
                console.print(f"[dim]Text was:[/dim] {safe_text}...")
                console.print()
                return idx, {"score": 0.5, "status": "revise", "reason": "processing error"}, None, None, t, 0, t

        tasks = [score_and_improve(i, text, expanded_source_mapping[i]) for i, text in enumerate(expanded_spans)]

        try:
            for f in asyncio.as_completed(tasks):
                try:
                    idx, r, improved_text, content_type, final_text, improvement_iterations, original_text = await f
                    tag = r["status"]
                    reasons.append(r["reason"])
                    stats[tag] += 1

                    src_file = expanded_source_mapping[idx]

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

                    # Only count valid records (status "keep") toward save interval
                    if tag == "keep":
                        valid_records_count += 1

                    if save_interval > 0 and output_path and valid_records_count % save_interval == 0 and len(recs) > 0:
                        save_incremental(recs, valid_records_count, processed_count)
                        recs.clear()

                except Exception as e:
                    # Escape potential Rich markup in error messages
                    error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
                    console.print(f"[red]Error processing a segment result:[/red] {error_msg}")

        except asyncio.CancelledError:
            console.print("[yellow]Processing cancelled.[/yellow]")
        finally:
            if save_interval > 0 and output_path and recs:
                save_incremental(recs, valid_records_count, processed_count, force=True)
                recs.clear()

        return all_processed_recs, stats, reasons

    try:
        all_recs, stats, reasons = asyncio.run(process())
        display_summary_panel("Combined CSV files", stats, reasons)
        return all_recs
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting.[/bold red]")
        return []


def run(i: Path, o: Path, f: str, pretty: bool, n: str, w: int, save_interval: int = 10, force: bool = False):
    console.print("[bold cyan]═══ X-Spanformer PDF2JSONL Pipeline ═══[/bold cyan]")
    console.print("[green]✔ Initializing agents and processing pipeline[/green]")

    if not i.exists() or not any(i.iterdir()):
        console.print(f"[red]⚠ No PDFs found in {i}[/red]")
        return

    if not i.is_file():
        pdfs = list(i.glob("*.pdf"))
        console.print(f"[bold magenta]Found {len(pdfs)} PDF files:[/bold magenta]")
        for x in pdfs:
            console.print(f"[cyan]• {x.name}[/cyan]")
    else:
        console.print(f"[bold magenta]Processing single PDF file: {i.name}[/bold magenta]")
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

    console.print(f"[bold blue]━━━ Processing {len(pdfs)} PDF files ━━━[/bold blue]")
    for pdf in pdfs:
        expected_csv_name = f"{hash_name(pdf)}.csv"
        expected_csv_path = csv_dir / expected_csv_name

        console.print(f"[cyan]→ Processing: {pdf.name}[/cyan]")

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

    console.print(f"[white]→ Processing {len(csvs)} CSV files together[/white]")

    # Check if dataset file already exists
    dataset_file = o / f"{base}.jsonl"
    if dataset_file.exists():
        console.print(f"[yellow]⚠ Existing dataset file found: {dataset_file.name}[/yellow]")
        console.print(f"[yellow]  Incremental saves will append to existing file[/yellow]")
    else:
        console.print(f"[cyan]📝 Will create new dataset file: {dataset_file.name}[/cyan]")

    # Log the mapping for verification
    console.print("[dim]PDF → CSV mapping:[/dim]")
    for csv_name, pdf_name in pdf_mapping.items():
        console.print(f"[dim]  {csv_name} ← {pdf_name}[/dim]")
    console.print()
    allr = process_all_csvs(csvs, f, w, {}, save_interval, o, base, pdf_mapping)

    if not allr:
        console.print(f"[red]⚠ No valid records found across all CSVs[/red]")
        return

    total = len(allr)
    kept = sum(1 for r in allr if not r.meta.tags)
    ratio = round(kept / total * 100, 2)

    console.print("[bold green]═══ Final Summary ═══[/bold green]")
    console.print(f"[white]Total records:[/white] {total}")
    console.print(f"[green]Kept:[/green] {kept} [cyan]({ratio}%)[/cyan]")
    console.print(f"[yellow]Discarded:[/yellow] {total - kept}")
    console.print()

    o.mkdir(parents=True, exist_ok=True)
    j1, j2 = o / f"{base}.jsonl", o / f"{base}.json"

    with j1.open("w", encoding="utf-8") as writer:
        for x in allr:
            writer.write(json.dumps(x.model_dump(), ensure_ascii=False) + "\n")
    console.print(f"[green]✔ Wrote {total} entries → {j1.name}[/green]")

    if pretty:
        with j2.open("w", encoding="utf-8") as writer:
            json.dump([x.model_dump() for x in allr], writer, ensure_ascii=False, indent=2)
        console.print(f"[cyan]• Pretty JSON → {j2.name}[/cyan]")


def split_long_text(text: str, max_length: int = 512) -> list[str]:
    """
    Split long text into smaller chunks using spaCy sentence boundaries.

    Args:
        text: The text to potentially split
        max_length: Maximum length per chunk

    Returns:
        List of text chunks (single item if no splitting needed)
    """
    if len(text) <= max_length:
        return [text]

    try:
        # Load spaCy model (try small models first)
        nlp = None
        for model_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
            try:
                nlp = spacy.load(model_name)
                break
            except OSError:
                continue

        if nlp is None:
            raise OSError("No spaCy English model found. Please install one: python -m spacy download en_core_web_sm")

        doc = nlp(text)
        chunks = []
        current_chunk = ""

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # If adding this sentence would exceed max_length, save current chunk
            if current_chunk and len(current_chunk) + len(sent_text) + 1 > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = sent_text
            else:
                if current_chunk:
                    current_chunk += " " + sent_text
                else:
                    current_chunk = sent_text

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if len(chunks) > 1:
            console.print(f"[cyan]✂️ Split long text ({len(text)} chars) into {len(chunks)} chunks[/cyan]")

        return chunks if chunks else [text]

    except Exception as e:
        raise RuntimeError(f"Error in spaCy text splitting: {e}. Ensure spaCy and an English model are properly installed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-f", "--field", type=str, default="text")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("-n", "--name", type=str, default="dataset")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=10, help="Save dataset incrementally after every N segments (0 to disable)")
    p.add_argument("--force", action="store_true", help="Force regeneration of all CSV files, ignoring existing ones")
    a = p.parse_args()
    run(a.input, a.output, a.field, a.pretty, a.name, a.workers, a.save_interval, a.force)
