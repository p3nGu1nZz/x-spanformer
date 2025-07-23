#!/usr/bin/env python3
"""
repo2jsonl.py

Export a GitHub repository to local directory and extract code files for training data generation.
Uses git clone --depth 1 to get source files only, then removes .git to avoid conflicts.

This pipeline:
1. Exports a public GitHub repository via shallow git clone
2. Removes .git directory to avoid conflicts with parent repository
3. Extracts relevant code files from exported source
4. Processes files through CSV extraction
5. Evaluates content using Judge agents
6. Outputs structured JSONL records using PretrainRecord schema
"""
import argparse
import asyncio
import json
import sys
import yaml
from pathlib import Path
from typing import Optional, List, Dict

# Add the parent directory to the path to import schema modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.agents.config_loader import load_judge_config
from x_spanformer.agents.ollama_client import check_ollama_connection
from x_spanformer.agents.rich_utils import console
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.vocab.vocab_logging import setup_vocab_logging, get_vocab_logger
from x_spanformer.pipelines.shared.csv_processor import process_all_csvs
from x_spanformer.pipelines.shared.repo_exporter import GitRepoExporter, CodeFileExtractor

# Module-level logger that gets configured in main()
logger = None

def load_pipeline_config() -> Dict:
    """Load pipeline configuration from repo2jsonl.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "pipelines" / "repo2jsonl.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="repo2jsonl",
        description="Export GitHub repository and extract code files for training data generation"
    )
    parser.add_argument(
        "-u", "--url", 
        type=str, required=True,
        help="GitHub repository URL (e.g., https://github.com/user/repo)"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path, required=True,
        help="Input directory to export repository files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, required=True,
        help="Output directory for CSV and JSONL results"
    )
    parser.add_argument(
        "-n", "--name",
        type=str, default="repo_dataset",
        help="Base name for output files (default: repo_dataset)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int, default=None,
        help="Number of concurrent workers for judge evaluation (default: from config)"
    )
    parser.add_argument(
        "--save-interval",
        type=int, default=None,
        help="Save progress every N records (default: from config)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-export of repository and regeneration of all files"
    )
    parser.add_argument(
        "-p", "--pretty",
        action="store_true",
        help="Output pretty JSON file for inspection"
    )
    parser.add_argument(
        "--branch",
        type=str, default=None,
        help="Specific branch to clone (default: repository default branch)"
    )
    return parser.parse_args()

def run(url: str, input_dir: Path, output_dir: Path, name: str, workers: Optional[int] = None, 
        save_interval: Optional[int] = None, force: bool = False, pretty: bool = False,
        branch: Optional[str] = None):
    """Main pipeline execution function."""
    
    # Load pipeline configuration
    config = load_pipeline_config()
    
    # Get configuration values with CLI override capability
    processing_config = config.get("processing", {})
    if workers is None:
        workers = processing_config.get("default_workers", 2)
    if save_interval is None:
        save_interval = processing_config.get("default_save_interval", 5)
    
    # Ensure we have valid integer values
    assert isinstance(workers, int), f"Workers must be an integer, got {type(workers)}"
    assert isinstance(save_interval, int), f"Save interval must be an integer, got {type(save_interval)}"
    
    # Setup logging
    global logger
    logger = setup_vocab_logging(output_dir, 'repo2jsonl')
    
    logger.info("=" * 80)
    logger.info("X-SPANFORMER REPO2JSONL PIPELINE (GIT EXPORT)")
    logger.info("=" * 80)
    logger.info(f"Repository URL: {url}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Save interval: {save_interval}")
    logger.info(f"Branch: {branch or 'default'}")
    logger.info("-" * 80)
    
    console.print("[bold cyan]═══ X-Spanformer REPO2JSONL Pipeline (Git Export) ═══[/bold cyan]")
    console.print("[green]✔ Initializing repository export pipeline[/green]")

    # Validate arguments
    if save_interval < 0:
        console.print(f"[red]Error: --save-interval cannot be negative. Use 0 to disable incremental saving.[/red]")
        return

    # Create directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test Ollama connection
    console.print("[cyan]🔍 Testing Ollama connection...[/cyan]")
    agent_config = load_judge_config()
    model = agent_config.get("judge", {}).get("model_name", "llama3.2:1b")
    
    try:
        if not asyncio.run(check_ollama_connection(model)):
            console.print(f"[red]❌ Ollama connection failed! Please ensure Ollama is running and model '{model}' is available.[/red]")
            return
        else:
            console.print(f"[green]✅ Ollama is running and accessible (model: {model})[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error testing Ollama connection: {str(e)}[/red]")
        return

    console.print()

    # Step 1: Export repository via git clone
    console.print(f"[bold blue]━━━ Step 1: Repository Export ━━━[/bold blue]")
    
    exporter = GitRepoExporter(config)
    try:
        repo_path = exporter.export_repository(url, input_dir, branch=branch, force=force)
        console.print(f"[green]✅ Repository exported to: {repo_path}[/green]")
        logger.info(f"Repository successfully exported to: {repo_path}")
    except Exception as e:
        console.print(f"[red]❌ Failed to export repository: {e}[/red]")
        logger.error(f"Repository export failed: {e}")
        return

    console.print()

    # Step 2: Extract code files
    console.print(f"[bold blue]━━━ Step 2: Code File Extraction ━━━[/bold blue]")
    
    code_extractor = CodeFileExtractor(config)
    
    try:
        csv_files = code_extractor.extract_to_csv(repo_path, output_dir, force=force)
        console.print(f"[green]✅ Extracted {len(csv_files)} CSV files[/green]")
        logger.info(f"Code extraction complete: {len(csv_files)} CSV files generated")
        
        for csv_file in csv_files:
            console.print(f"[dim]  Generated: {csv_file.name}[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Failed to extract code files: {e}[/red]")
        logger.error(f"Code file extraction failed: {e}")
        return

    if not csv_files:
        console.print(f"[red]⚠ No CSV files generated[/red]")
        return

    console.print()

    # Step 3: Process CSV files through judge evaluation
    console.print(f"[bold blue]━━━ Step 3: Judge Evaluation ━━━[/bold blue]")
    console.print(f"[white]📊 Processing {len(csv_files)} CSV files for AI evaluation[/white]")

    # Create mapping for source attribution
    repo_mapping = {}
    for csv_file in csv_files:
        repo_mapping[csv_file.name] = f"{url}#{csv_file.stem}"

    # Load existing records if any
    existing_records = load_existing_records(output_dir, name, force)

    # Process through our shared CSV processor
    try:
        all_records = process_all_csvs(
            csv_files=csv_files,
            col="content",  # Code content column
            w=workers,
            cfg={"content_type": "code"},  # Specify this is code content for proper text processing
            save_interval=save_interval,
            output_path=output_dir,
            base_name=name,
            pdf_mapping=repo_mapping,
            existing_records=existing_records
        )
        
        # Update extracted_by field to reflect this pipeline
        for record in all_records:
            record.meta.extracted_by = "repo2jsonl"
            record.meta.extracted_by = "repo2jsonl"
        
        if not all_records:
            console.print(f"[red]⚠ No valid records processed[/red]")
            return

        # Final summary
        total = len(all_records)
        kept = sum(1 for r in all_records if r.meta.status == "keep")
        ratio = round(kept / total * 100, 2) if total > 0 else 0
        
        console.print("[bold green]═══ Final Summary ═══[/bold green]")
        console.print(f"[white]Repository:[/white] {url}")
        console.print(f"[white]Total records processed:[/white] {total}")
        console.print(f"[green]Kept:[/green] {kept} [cyan]({ratio}%)[/cyan]")
        console.print(f"[yellow]Discarded:[/yellow] {total - kept}")

        # Optional pretty JSON output
        if pretty:
            jsonl_dir = output_dir / "jsonl"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            pretty_file = jsonl_dir / f"{name}.json"
            with pretty_file.open("w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in all_records], f, ensure_ascii=False, indent=2)
            console.print(f"[cyan]📋 Pretty JSON → {pretty_file.name}[/cyan]")

        logger.info(f"Pipeline completed successfully: {total} records, {kept} kept ({ratio}%)")

    except Exception as e:
        console.print(f"[red]❌ Processing failed: {e}[/red]")
        logger.error(f"CSV processing failed: {e}")
        return

    console.print()
    console.print("[bold green]✅ Repository export pipeline completed![/bold green]")

def load_existing_records(output_dir: Path, base_name: str, force: bool) -> List[PretrainRecord]:
    """Load existing records from previous runs unless force mode is enabled."""
    if force:
        # Clean existing files in force mode
        jsonl_dir = output_dir / "jsonl"
        if jsonl_dir.exists():
            for file in [f"{base_name}.jsonl", "discard.jsonl"]:
                file_path = jsonl_dir / file
                if file_path.exists():
                    file_path.unlink()
                    console.print(f"[yellow]🔄 Force mode: Removed existing {file}[/yellow]")
        return []

    existing_records = []
    jsonl_dir = output_dir / "jsonl"
    
    # Load from both dataset and discard files
    for filename in [f"{base_name}.jsonl", "discard.jsonl"]:
        file_path = jsonl_dir / filename
        if file_path.exists():
            try:
                count = 0
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            existing_records.append(PretrainRecord.model_validate_json(line))
                            count += 1
                
                status = 'keep' if 'discard' not in filename else 'discard'
                console.print(f"[cyan]✔ Loaded {count} {status} records from {filename}[/cyan]")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not load {filename}: {e}[/yellow]")
    
    return existing_records

def main():
    """Main entry point."""
    args = parse_args()
    
    run(
        url=args.url,
        input_dir=args.input,
        output_dir=args.output,
        name=args.name,
        workers=args.workers,
        save_interval=args.save_interval,
        force=args.force,
        pretty=args.pretty,
        branch=args.branch
    )

if __name__ == "__main__":
    main()
