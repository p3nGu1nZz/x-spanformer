#!/usr/bin/env python3
"""
X-Spanformer Span Annotator Pipeline

Three-turn hierarchical annotation pipeline with robust async session management.

Usage:
    python -m x_spanformer.pipelines.span_annotator \
        --corpus data/vocab/corpus.jsonl \
        --output data/annotations \
        --range 1-100

Key Features:
    - Three-turn conversation strategy: word-level -> phrase-level -> clause-level
    - Async batch processing with concurrency control
    - Resume capability and progress tracking
    - Comprehensive validation and error handling
    - Real-time telemetry and statistics
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Core imports
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord, SpanAnnotation
from x_spanformer.agents.session.span_annotator_session import SpanAnnotatorSession
from x_spanformer.agents.ollama_client import check_ollama_connection

# Constants
DEFAULT_MODEL = "llama3.2:3b"
MAX_TOTAL_FAILURES = 3  # Exit pipeline after this many total failures

# Initialize logger
logger = logging.getLogger(__name__)


def format_eta_time(eta_minutes: float) -> str:
    """
    Format ETA time in a human-readable format.
    
    Args:
        eta_minutes: ETA in minutes
        
    Returns:
        Formatted time string (e.g., "2h 30m", "1d 5h 20m")
    """
    if eta_minutes < 60:
        return f"{eta_minutes:.1f} min"
    
    hours = int(eta_minutes // 60)
    minutes = int(eta_minutes % 60)
    
    if hours < 24:
        return f"{hours}h {minutes}m"
    
    days = hours // 24
    remaining_hours = hours % 24
    
    if remaining_hours == 0 and minutes == 0:
        return f"{days}d"
    elif minutes == 0:
        return f"{days}d {remaining_hours}h"
    else:
        return f"{days}d {remaining_hours}h {minutes}m"


class SpanAnnotatorPipeline:
    """
    Three-turn span annotation pipeline with robust session management.
    
    Combines hierarchical annotation strategy with comprehensive
    error handling and progress tracking.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        conversation_timeout: float = 180.0,
        max_retries: int = 3
    ):
        """Initialize the span annotation pipeline."""
        self.model_name = model_name
        self.temperature = temperature
        self.conversation_timeout = conversation_timeout
        self.max_retries = max_retries
        
        # Initialize session
        self.session = SpanAnnotatorSession(
            model_name=model_name,
            temperature=temperature,
            conversation_timeout=conversation_timeout,
            max_retries=max_retries
        )
        
        # Processing statistics
        self.pipeline_stats = {
            "total_sequences": 0,
            "processed_sequences": 0,
            "successful_annotations": 0,
            "failed_annotations": 0,
            "total_spans": 0,
            "started_at": None,
            "completed_at": None
        }
    
    def get_sequence_number(self, sequence: PretrainRecord) -> int:
        """Extract sequence number from PretrainRecord, checking meta field first."""
        # Check meta field first (same logic as in filtering)
        if hasattr(sequence, 'meta') and sequence.meta:
            if hasattr(sequence.meta, 'sequence_number'):
                return sequence.meta.sequence_number or 0
            elif isinstance(sequence.meta, dict) and 'sequence_number' in sequence.meta:
                return sequence.meta['sequence_number'] or 0
        
        # Fallback to direct sequence_number field
        return getattr(sequence, 'sequence_number', 0)
    
    def parse_range_specification(self, range_spec: str) -> List[int]:
        """Parse range specification into list of sequence numbers."""
        sequence_ids = []
        for part in range_spec.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                sequence_ids.extend(range(start, end + 1))
            else:
                sequence_ids.append(int(part))
        return sorted(set(sequence_ids))
    
    def load_sequences(
        self, 
        corpus_file: Path, 
        range_spec: Optional[str] = None
    ) -> List[PretrainRecord]:
        """Load sequences from corpus file with optional filtering."""
        sequences = []
        
        logger.info(f"Loading corpus: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    sequence = PretrainRecord(**data)
                    
                    # Ensure sequence_number is set from meta if not present at root level
                    if sequence.sequence_number is None and sequence.meta:
                        if hasattr(sequence.meta, 'sequence_number') and sequence.meta.sequence_number:
                            sequence.sequence_number = sequence.meta.sequence_number
                        elif isinstance(sequence.meta, dict) and 'sequence_number' in sequence.meta:
                            sequence.sequence_number = sequence.meta['sequence_number']
                    
                    sequences.append(sequence)
                except Exception as e:
                    logger.warning(f"Parse error line {line_num}: {e}")
        
        logger.info(f"Loaded {len(sequences)} sequences")
        
        # Apply range filtering if specified
        if range_spec:
            target_sequence_ids = self.parse_range_specification(range_spec)
            original_count = len(sequences)
            
            filtered_sequences = []
            for seq in sequences:
                # Get sequence number from meta field
                seq_num = None
                if hasattr(seq, 'meta') and seq.meta:
                    if hasattr(seq.meta, 'sequence_number'):
                        seq_num = seq.meta.sequence_number
                    elif isinstance(seq.meta, dict) and 'sequence_number' in seq.meta:
                        seq_num = seq.meta['sequence_number']
                
                if seq_num and seq_num in target_sequence_ids:
                    filtered_sequences.append(seq)
            
            logger.info(f"Filtered to {len(filtered_sequences)}/{original_count} sequences")
            sequences = filtered_sequences
        
        return sequences
    
    def ensure_output_structure(self, output_dir: Path):
        """Ensure output directory structure exists."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "working").mkdir(exist_ok=True)
    
    def load_existing_results(self, output_dir: Path) -> Dict[int, str]:
        """Load existing annotation results for resume capability with span count validation."""
        existing_results = {}
        working_dir = output_dir / "working"
        
        if not working_dir.exists():
            return existing_results
        
        working_files = list(working_dir.glob("*.json"))
        logger.info(f"Checking {len(working_files)} existing results")
        
        total_existing_spans = 0
        completed_sequences = 0
        failed_sequences = 0
        
        for working_file in working_files:
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sequence_number = data.get("sequence_number", 0)
                span_annotations = data.get("span_annotations", [])
                
                if span_annotations:
                    status = "completed"
                    span_count = len(span_annotations)
                    total_existing_spans += span_count
                    completed_sequences += 1
                    existing_results[sequence_number] = status
                else:
                    # Remove working file if no spans - force retry
                    logger.info(f"Removing empty working file for sequence {sequence_number}: {working_file.name}")
                    working_file.unlink()
                    failed_sequences += 1
                
            except Exception as e:
                logger.warning(f"Load error {working_file.name}: {e}")
                # Remove corrupted working file - force retry
                logger.info(f"Removing corrupted working file: {working_file.name}")
                try:
                    working_file.unlink()
                except:
                    pass
                failed_sequences += 1
        
        if existing_results:
            logger.info(f"Found {len(existing_results)} existing sequences:")
            logger.info(f"  - {completed_sequences} completed sequences")
            logger.info(f"  - {failed_sequences} failed sequences") 
            logger.info(f"  - {total_existing_spans} total spans in working files")
            
            # Update pipeline stats with existing span count
            self.pipeline_stats["total_spans"] = total_existing_spans
            
        return existing_results
    
    def save_working_file(
        self, 
        output_dir: Path, 
        sequence: PretrainRecord,
        annotation_record: Optional[Any] = None,
        error_message: Optional[str] = None
    ):
        """Save annotation result to working file."""
        working_dir = output_dir / "working"
        
        # Get sequence number using consistent helper method
        sequence_number = self.get_sequence_number(sequence)
        
        working_file = working_dir / f"sequence-{sequence_number:08d}.json"
        
        working_data = {
            "sequence_number": sequence_number,
            "raw_text": sequence.raw,
            "domain_type": getattr(sequence, 'type', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if annotation_record else "failed",
            "error_message": error_message
        }
        
        if annotation_record:
            working_data.update({
                "span_annotations": [
                    {
                        "start_pos": span.start_pos,
                        "end_pos": span.end_pos,
                        "xbar_label": span.xbar_label,
                        "text": getattr(span, 'linguistic_features', {}).get('extracted_text', '')
                    }
                    for span in annotation_record.span_annotations
                ],
                "total_spans": len(annotation_record.span_annotations),
                "agent_metadata": annotation_record.agent_metadata if hasattr(annotation_record, 'agent_metadata') else {}
            })
        
        with open(working_file, 'w', encoding='utf-8') as f:
            json.dump(working_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved working file: sequence {sequence_number}")
    
    def consolidate_results(self, output_dir: Path):
        """Consolidate working files into final annotation format with one span per record."""
        working_dir = output_dir / "working"
        annotations_file = output_dir / "annotations.jsonl"  # Save in same dir as metadata.json
        
        working_files = list(working_dir.glob("*.json"))
        logger.info(f"Consolidating {len(working_files)} working files")
        
        # Define label hierarchy for deduplication (higher value = more specific)
        label_hierarchy = {
            # Clause-level (most specific)
            "main_clause": 3, "subordinate_clause": 3, "relative_clause": 3,
            "if_statement": 3, "loop_statement": 3, "function_definition": 3,
            "class_definition": 3, "import_statement": 3, "return_statement": 3,
            "sentence": 3, "fragment": 3,
            
            # Phrase-level (medium specificity)
            "noun_phrase": 2, "verb_phrase": 2, "adjective_phrase": 2, "adverb_phrase": 2,
            "prepositional_phrase": 2, "expression": 2, "function_call": 2, "assignment": 2,
            "parameter_list": 2, "argument_list": 2, "code_block": 2, "documentation_comment": 2,
            
            # Word-level (least specific)
            "noun": 1, "verb": 1, "adjective": 1, "adverb": 1, "preposition": 1,
            "determiner": 1, "pronoun": 1, "conjunction": 1, "keyword": 1, "identifier": 1,
            "operator": 1, "literal": 1, "inline_code": 1, "punctuation": 1, "proper_noun": 1
        }
        
        # Collect all valid annotations with deduplication
        all_annotations = {}  # Key: (sequence_number, start_pos, end_pos, text) -> annotation
        
        for working_file in sorted(working_files):
            try:
                with open(working_file, 'r', encoding='utf-8') as inf:
                    data = json.load(inf)
                
                if data.get("status") == "completed" and data.get("span_annotations"):
                    for span_annotation in data["span_annotations"]:
                        # Validate span positions before processing
                        start_pos = span_annotation["start_pos"]
                        end_pos = span_annotation["end_pos"]
                        expected_text = span_annotation["text"]
                        raw_text = data["raw_text"]
                        
                        # Check if positions are valid and match expected text
                        if start_pos >= 0 and end_pos <= len(raw_text) and start_pos < end_pos:
                            # Check if the extracted text matches (use exclusive end for extraction)
                            actual_text = raw_text[start_pos:end_pos]
                            if actual_text == expected_text:
                                # Create deduplication key
                                dup_key = (data["sequence_number"], start_pos, end_pos, expected_text)
                                
                                # Create flattened record with id first (will be set during output)
                                flattened_record = {
                                    "sequence_number": data["sequence_number"],
                                    "raw": data["raw_text"],
                                    "domain_type": data["domain_type"],
                                    "start_pos": start_pos,
                                    "end_pos": end_pos,
                                    "xbar_label": span_annotation["xbar_label"],
                                    "text": expected_text,
                                    "model": self.model_name,
                                    "timestamp": data["timestamp"]
                                }
                                
                                # Apply deduplication logic
                                if dup_key in all_annotations:
                                    # Keep the annotation with higher hierarchy (more specific label)
                                    existing_label = all_annotations[dup_key]["xbar_label"]
                                    new_label = span_annotation["xbar_label"]
                                    
                                    existing_priority = label_hierarchy.get(existing_label, 0)
                                    new_priority = label_hierarchy.get(new_label, 0)
                                    
                                    if new_priority > existing_priority:
                                        all_annotations[dup_key] = flattened_record
                                        logger.debug(f"Updated {existing_label} to {new_label}: '{expected_text[:20]}...'")
                                    else:
                                        logger.debug(f"Kept {existing_label} over {new_label}: '{expected_text[:20]}...'")
                                else:
                                    all_annotations[dup_key] = flattened_record
                            else:
                                logger.warning(f"Text mismatch seq {data['sequence_number']}: expected '{expected_text}' at {start_pos}-{end_pos}")
                        else:
                            logger.warning(f"Invalid span bounds seq {data['sequence_number']}: {start_pos}-{end_pos} (text len {len(raw_text)})")
            
            except Exception as e:
                logger.warning(f"Consolidation error {working_file.name}: {e}")
        
        # Write deduplicated annotations to file with unique IDs
        total_annotations = len(all_annotations)
        with open(annotations_file, 'w', encoding='utf-8') as outf:
            for span_id, annotation in enumerate(all_annotations.values()):
                # Create ordered record with id first
                ordered_record = {
                    "id": span_id,
                    "sequence_number": annotation["sequence_number"],
                    "raw": annotation["raw"],
                    "domain_type": annotation["domain_type"],
                    "start_pos": annotation["start_pos"],
                    "end_pos": annotation["end_pos"],
                    "xbar_label": annotation["xbar_label"],
                    "text": annotation["text"],
                    "model": annotation["model"],
                    "timestamp": annotation["timestamp"]
                }
                outf.write(json.dumps(ordered_record, ensure_ascii=False) + '\n')
        
        logger.info(f"Consolidated {total_annotations} spans into {annotations_file.name}")
    
    def update_metadata(self, output_dir: Path):
        """Update global metadata file."""
        metadata_file = output_dir / "metadata.json"
        
        # Calculate actual stats from working files
        working_dir = output_dir / "working"
        working_files = list(working_dir.glob("*.json"))
        
        successful_count = 0
        failed_count = 0
        total_spans = 0
        
        for working_file in working_files:
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get("status") == "completed":
                    successful_count += 1
                    total_spans += data.get("total_spans", 0)
                else:
                    failed_count += 1
            except:
                failed_count += 1
        
        metadata = {
            "pipeline": "unified_span_annotator",
            "model": self.model_name,
            "last_updated": datetime.now().isoformat(),
            "processing_stats": {
                "total_sequences": self.pipeline_stats["total_sequences"],
                "successful_annotations": successful_count,
                "failed_annotations": failed_count,
                "total_spans": total_spans,
                "success_rate": successful_count / max(1, successful_count + failed_count)
            },
            "session_stats": self.session.get_statistics()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    async def process_sequences(
        self,
        corpus_file: Path,
        output_dir: Path,
        range_spec: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process sequences with comprehensive annotation.
        
        Args:
            corpus_file: Path to corpus.jsonl file
            output_dir: Output directory for results
            range_spec: Range specification (e.g., "1-100")
            stream: Stream results to console in real-time
            
        Returns:
            Processing statistics
        """
        self.pipeline_stats["started_at"] = datetime.now().isoformat()
        
        # Ensure output structure
        self.ensure_output_structure(output_dir)
        
        # Load sequences
        sequences = self.load_sequences(corpus_file, range_spec)
        self.pipeline_stats["total_sequences"] = len(sequences)
        
        if not sequences:
            logger.warning("No sequences found to process")
            return self.pipeline_stats
        
        # Load existing results for resume (always enabled)
        existing_results = self.load_existing_results(output_dir)
        
        # Filter sequences to process
        sequences_to_process = []
        for seq in sequences:
            # Get sequence number using consistent helper method
            seq_id = self.get_sequence_number(seq)
                    
            if seq_id not in existing_results or existing_results[seq_id] != "completed":
                sequences_to_process.append(seq)
        
        skipped_count = len(sequences) - len(sequences_to_process)
        if skipped_count > 0:
            logger.info(f"Processing {len(sequences_to_process)} sequences (skipped {skipped_count} completed)")
        else:
            logger.info(f"Processing {len(sequences_to_process)} sequences")
        
        # Progress tracking
        async def progress_callback(progress_info):
            pass  # Simplified - main progress logging handled in loop
        
        # Process sequences individually (sequential processing)
        successful_count = 0
        failed_count = 0
        session_start_time = datetime.now()
        total_completed_already = len(existing_results)
        
        for i, sequence in enumerate(sequences_to_process):
            # Get sequence number using consistent helper method
            seq_id = self.get_sequence_number(sequence)
            
            # Calculate current position in total sequences (already completed + current position)
            current_position = total_completed_already + i + 1
            total_sequences = len(sequences)
            
            logger.info(f"Processing {current_position}/{total_sequences}: sequence {seq_id}")
            
            try:
                # Annotate single sequence
                result = await self.session.annotate_single_sequence(sequence, progress_callback=progress_callback)
                
                if result.success and result.annotation_record:
                    annotation_record = result.annotation_record
                    self.save_working_file(output_dir, sequence, annotation_record)
                    successful_count += 1
                    
                    span_count = len(annotation_record.span_annotations)
                    # Update total span count in pipeline stats
                    self.pipeline_stats["total_spans"] += span_count
                    logger.info(f"SUCCESS: Sequence {seq_id}: {span_count} spans annotated")
                    
                    # Calculate and log progress summary
                    current_time = datetime.now()
                    elapsed_seconds = (current_time - session_start_time).total_seconds()
                    completed_count = successful_count + failed_count
                    remaining_count = len(sequences_to_process) - completed_count
                    total_completed_overall = total_completed_already + completed_count
                    
                    if completed_count > 0 and elapsed_seconds > 0:
                        avg_time_per_sequence = elapsed_seconds / completed_count
                        sequences_per_minute = (completed_count / elapsed_seconds) * 60
                        eta_seconds = remaining_count * avg_time_per_sequence
                        eta_minutes = eta_seconds / 60
                        eta_formatted = format_eta_time(eta_minutes)
                        
                        logger.info("=" * 80)
                        logger.info(f"{total_completed_overall}/{total_sequences} sequences completed | "
                                  f"Avg: {sequences_per_minute:.1f} seq/min | "
                                  f"ETA: {eta_formatted}")
                        logger.info(f"Total spans annotated: {self.pipeline_stats['total_spans']}")
                        logger.info("=" * 80)
                else:
                    error_msg = result.error_message or "Annotation failed"
                    failed_count += 1
                    logger.warning(f"FAILED: Sequence {seq_id}: {error_msg}")
                    
                    # Check if we've exceeded the maximum failure limit
                    if failed_count >= MAX_TOTAL_FAILURES:
                        logger.critical(f"CRITICAL: Reached maximum failure limit ({MAX_TOTAL_FAILURES} failures). Exiting pipeline.")
                        logger.critical(f"Failed sequences will be retried automatically on next run.")
                        break
                    
                    # Calculate and log progress summary
                    current_time = datetime.now()
                    elapsed_seconds = (current_time - session_start_time).total_seconds()
                    completed_count = successful_count + failed_count
                    remaining_count = len(sequences_to_process) - completed_count
                    total_completed_overall = total_completed_already + completed_count
                    
                    if completed_count > 0 and elapsed_seconds > 0:
                        avg_time_per_sequence = elapsed_seconds / completed_count
                        sequences_per_minute = (completed_count / elapsed_seconds) * 60
                        eta_seconds = remaining_count * avg_time_per_sequence
                        eta_minutes = eta_seconds / 60
                        eta_formatted = format_eta_time(eta_minutes)
                        
                        logger.info("=" * 80)
                        logger.info(f"{total_completed_overall}/{total_sequences} sequences completed | "
                                  f"Avg: {sequences_per_minute:.1f} seq/min | "
                                  f"ETA: {eta_formatted}")
                        logger.info(f"Total spans annotated: {self.pipeline_stats['total_spans']}")
                        logger.info("=" * 80)
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"ERROR: Sequence {seq_id}: {str(e)}")
                
                # Check if we've exceeded the maximum failure limit
                if failed_count >= MAX_TOTAL_FAILURES:
                    logger.critical(f"CRITICAL: Reached maximum failure limit ({MAX_TOTAL_FAILURES} failures). Exiting pipeline.")
                    logger.critical(f"Failed sequences will be retried automatically on next run.")
                    break
                
                # Calculate and log progress summary
                current_time = datetime.now()
                elapsed_seconds = (current_time - session_start_time).total_seconds()
                completed_count = successful_count + failed_count
                remaining_count = len(sequences_to_process) - completed_count
                total_completed_overall = total_completed_already + completed_count
                
                if completed_count > 0 and elapsed_seconds > 0:
                    avg_time_per_sequence = elapsed_seconds / completed_count
                    sequences_per_minute = (completed_count / elapsed_seconds) * 60
                    eta_seconds = remaining_count * avg_time_per_sequence
                    eta_minutes = eta_seconds / 60
                    eta_formatted = format_eta_time(eta_minutes)
                    
                    logger.info("=" * 80)
                    logger.info(f"{total_completed_overall}/{total_sequences} sequences completed | "
                              f"Avg: {sequences_per_minute:.1f} seq/min | "
                              f"ETA: {eta_formatted}")
                    logger.info(f"Total spans annotated: {self.pipeline_stats['total_spans']}")
                    logger.info("=" * 80)
        
        # Update statistics
        self.pipeline_stats["processed_sequences"] = len(sequences_to_process)
        self.pipeline_stats["successful_annotations"] = successful_count
        self.pipeline_stats["failed_annotations"] = failed_count
        self.pipeline_stats["completed_at"] = datetime.now().isoformat()
        
        # Check if pipeline exited early due to failures
        processed_count = successful_count + failed_count
        if processed_count < len(sequences_to_process):
            remaining_count = len(sequences_to_process) - processed_count
            logger.warning(f"Pipeline exited early: {remaining_count} sequences not processed due to failure limit")
            self.pipeline_stats["early_exit"] = True
            self.pipeline_stats["remaining_sequences"] = remaining_count
        else:
            self.pipeline_stats["early_exit"] = False
        
        # Consolidate and save metadata
        self.consolidate_results(output_dir)
        self.update_metadata(output_dir)
        
        # Final status summary
        if failed_count >= MAX_TOTAL_FAILURES:
            logger.critical(f"Pipeline terminated due to {failed_count} failures. Failed sequences will be retried on next run.")
        else:
            logger.info(f"Pipeline completed: {successful_count}/{len(sequences_to_process)} successful")
        
        logger.info(f"Results saved to: {output_dir}")
        
        return self.pipeline_stats


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="X-Spanformer Span Annotator Pipeline")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to corpus.jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for annotations")
    parser.add_argument("--range", type=str, help="Range specification (e.g., '1-100', '5,10,15')")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature")
    parser.add_argument("--timeout", type=float, default=180.0, help="Conversation timeout (seconds)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.output / "annotations.log" if args.output else "annotations.log"
    
    # Ensure output directory exists for log file
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    logger.info("="*60)
    logger.info("X-SPANFORMER SPAN ANNOTATOR PIPELINE")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Range: {args.range or 'ALL SEQUENCES'}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Resume: ENABLED")
    
    try:
        # Check Ollama connection
        logger.info("Checking Ollama connection...")
        if not await check_ollama_connection(DEFAULT_MODEL):
            logger.error("ERROR: Ollama service unavailable at http://localhost:11434")
            logger.error("Please ensure Ollama is running and try again.")
            sys.exit(1)
        logger.info("Ollama connection successful")
        
        # Initialize pipeline
        pipeline = SpanAnnotatorPipeline(
            model_name=args.model,
            temperature=args.temperature,
            conversation_timeout=args.timeout
        )
        
        # Run processing
        stats = await pipeline.process_sequences(
            corpus_file=args.corpus,
            output_dir=args.output,
            range_spec=args.range
        )
        
        # Display final results
        logger.info("="*60)
        logger.info("PIPELINE COMPLETION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total sequences: {stats['total_sequences']}")
        logger.info(f"Processed: {stats['processed_sequences']}")
        logger.info(f"Successful: {stats['successful_annotations']}")
        logger.info(f"Failed: {stats['failed_annotations']}")
        
        session_stats = pipeline.session.get_statistics()
        logger.info(f"Total spans: {session_stats.get('total_spans', 0)}")
        logger.info(f"Success rate: {session_stats.get('success_rate', 0):.2%}")
        logger.info(f"Avg spans/sequence: {session_stats.get('avg_spans_per_sequence', 0):.1f}")
        
        logger.info(f"Results: {args.output}")
        logger.info(f"Working files: {args.output / 'working'}")
        logger.info(f"Annotations: {args.output / 'annotations.jsonl'}")
        
        # Check if pipeline failed due to too many failures
        if stats.get('early_exit', False) and stats['failed_annotations'] >= MAX_TOTAL_FAILURES:
            logger.critical(f"Pipeline terminated due to failure limit. Exiting with error code.")
            sys.exit(1)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Stats: {stats['successful_annotations']}/{stats['total_sequences']} sequences annotated")
        logger.info(f"Output: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
