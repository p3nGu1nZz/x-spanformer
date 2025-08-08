#!/usr/bin/env python3
"""
Unified Span Annotator Pipeline for X-Spanformer

Production-ready implementation combining three-turn hierarchical annotation
with robust async session management and comprehensive error handling.

Usage:
    python -m x_spanformer.pipelines.unified_span_annotator_pipeline \
        --corpus data/vocab/corpus.jsonl \
        --output data/annotations \
        --range 1-100

Key Features:
    - Three-turn conversation strategy: word-level → phrase-level → clause-level
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

# Initialize logger
logger = logging.getLogger(__name__)


class SpanAnnotatorPipeline:
    """
    Production-ready unified span annotation pipeline.
    
    Combines the three-turn annotation strategy with robust session management,
    error handling, and comprehensive progress tracking.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        conversation_timeout: float = 180.0,
        max_retries: int = 3
    ):
        """Initialize the unified pipeline."""
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
        
        logger.info(f"Loading sequences from {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    sequence = PretrainRecord(**data)
                    sequences.append(sequence)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
        
        logger.info(f"Loaded {len(sequences)} total sequences")
        
        # Apply range filtering if specified
        if range_spec:
            target_sequence_ids = self.parse_range_specification(range_spec)
            logger.info(f"Filtering for sequence numbers: {target_sequence_ids}")
            
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
                    logger.debug(f"Included sequence {seq_num}: {seq.id.id if seq.id else 'unknown'}")
            
            logger.info(f"Filtered to {len(filtered_sequences)} sequences")
            sequences = filtered_sequences
        
        return sequences
    
    def ensure_output_structure(self, output_dir: Path):
        """Ensure output directory structure exists."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "working").mkdir(exist_ok=True)
    
    def load_existing_results(self, output_dir: Path) -> Dict[int, str]:
        """Load existing annotation results for resume capability."""
        existing_results = {}
        working_dir = output_dir / "working"
        
        if not working_dir.exists():
            return existing_results
        
        working_files = list(working_dir.glob("*.json"))
        logger.info(f"Loading existing results from {len(working_files)} working files")
        
        for working_file in working_files:
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sequence_number = data.get("sequence_number", 0)
                status = "completed" if data.get("span_annotations") else "failed"
                existing_results[sequence_number] = status
                
            except Exception as e:
                logger.warning(f"Failed to load working file {working_file}: {e}")
        
        logger.info(f"Found {len(existing_results)} existing results")
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
        
        # Get sequence number directly from the sequence
        sequence_number = sequence.sequence_number or 0
        
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
        
        logger.debug(f"Saved working file for sequence {sequence_number}")
    
    def consolidate_results(self, output_dir: Path):
        """Consolidate working files into final annotation format."""
        working_dir = output_dir / "working"
        annotations_file = output_dir / "annotations.jsonl"  # Save in same dir as metadata.json
        
        working_files = list(working_dir.glob("*.json"))
        logger.info(f"Consolidating {len(working_files)} working files")
        
        total_annotations = 0
        with open(annotations_file, 'w', encoding='utf-8') as outf:
            for working_file in sorted(working_files):
                try:
                    with open(working_file, 'r', encoding='utf-8') as inf:
                        data = json.load(inf)
                    
                    if data.get("status") == "completed" and data.get("span_annotations"):
                        # Write consolidated annotation record
                        consolidated_record = {
                            "sequence_number": data["sequence_number"],
                            "raw": data["raw_text"],
                            "domain_type": data["domain_type"],
                            "span_annotations": data["span_annotations"],
                            "total_spans": data["total_spans"],
                            "metadata": {
                                "annotation_strategy": "three_turn_unified",
                                "model": self.model_name,
                                "timestamp": data["timestamp"]
                            }
                        }
                        
                        outf.write(json.dumps(consolidated_record, ensure_ascii=False) + '\n')
                        total_annotations += data["total_spans"]
                
                except Exception as e:
                    logger.warning(f"Failed to consolidate {working_file}: {e}")
        
        logger.info(f"Consolidated {total_annotations} total annotations to {annotations_file}")
    
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
            logger.warning("No sequences to process")
            return self.pipeline_stats
        
        # Load existing results for resume (always enabled)
        existing_results = self.load_existing_results(output_dir)
        
        # Filter sequences to process
        sequences_to_process = []
        for seq in sequences:
            # Get sequence number directly from the sequence
            seq_id = seq.sequence_number or 0
                    
            if seq_id not in existing_results or existing_results[seq_id] != "completed":
                sequences_to_process.append(seq)
        
        logger.info(f"Processing {len(sequences_to_process)} sequences (skipped {len(sequences) - len(sequences_to_process)} completed)")
        
        # Progress tracking
        async def progress_callback(progress_info):
            logger.info(f"[PROGRESS] Sequence {progress_info['sequence_number']}: {progress_info.get('phase', 'unknown')}")
            if progress_info.get('total_spans'):
                logger.info(f"[PROGRESS] Total spans: {progress_info['total_spans']}")
        
        # Process sequences individually (sequential processing)
        successful_count = 0
        failed_count = 0
        
        for i, sequence in enumerate(sequences_to_process):
            # Get sequence number directly from the sequence
            seq_id = sequence.sequence_number or 0
            
            logger.info(f"Processing sequence {i+1}/{len(sequences_to_process)}: ID {seq_id}")
            
            try:
                # Annotate single sequence
                result = await self.session.annotate_single_sequence(sequence, progress_callback=progress_callback)
                
                if result.success and result.annotation_record:
                    annotation_record = result.annotation_record
                    self.save_working_file(output_dir, sequence, annotation_record)
                    successful_count += 1
                    
                    span_count = len(annotation_record.span_annotations)
                    logger.info(f"Successfully annotated sequence {seq_id} with {span_count} spans")
                else:
                    error_msg = result.error_message or "Annotation returned None"
                    self.save_working_file(output_dir, sequence, error_message=error_msg)
                    failed_count += 1
                    logger.warning(f"Failed to annotate sequence {seq_id}: {error_msg}")
                    
            except Exception as e:
                self.save_working_file(output_dir, sequence, error_message=str(e))
                failed_count += 1
                logger.error(f"Failed to annotate sequence {seq_id}: {e}", exc_info=True)
        
        # Update statistics
        self.pipeline_stats["processed_sequences"] = len(sequences_to_process)
        self.pipeline_stats["successful_annotations"] = successful_count
        self.pipeline_stats["failed_annotations"] = failed_count
        self.pipeline_stats["completed_at"] = datetime.now().isoformat()
        
        # Consolidate and save metadata
        self.consolidate_results(output_dir)
        self.update_metadata(output_dir)
        
        logger.info(f"Pipeline completed: {successful_count}/{len(sequences_to_process)} successful")
        logger.info(f"Results saved to: {output_dir}")
        
        return self.pipeline_stats


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Unified X-bar Span Annotator Pipeline")
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
    log_file = args.output / "annotation_pipeline.log" if args.output else "annotation_pipeline.log"
    
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
    logger.info("UNIFIED X-SPANFORMER SPAN ANNOTATOR PIPELINE")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Range: {args.range or 'ALL SEQUENCES'}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Resume: ENABLED (always)")
    
    try:
        # Check Ollama connection
        logger.info("Checking Ollama connection...")
        if not await check_ollama_connection(DEFAULT_MODEL):
            logger.error("ERROR: Ollama service not available at http://localhost:11434")
            logger.error("Please ensure Ollama is running and try again.")
            sys.exit(1)
        logger.info("SUCCESS: Ollama connection successful")
        
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
        logger.info(f"SUCCESS: Total sequences: {stats['total_sequences']}")
        logger.info(f"SUCCESS: Processed: {stats['processed_sequences']}")
        logger.info(f"SUCCESS: Successful: {stats['successful_annotations']}")
        logger.info(f"ERROR: Failed: {stats['failed_annotations']}")
        
        session_stats = pipeline.session.get_statistics()
        logger.info(f"STATS: Total spans: {session_stats.get('total_spans', 0)}")
        logger.info(f"STATS: Success rate: {session_stats.get('success_rate', 0):.2%}")
        logger.info(f"STATS: Avg spans/sequence: {session_stats.get('avg_spans_per_sequence', 0):.1f}")
        
        logger.info(f"OUTPUT: Results: {args.output}")
        logger.info(f"OUTPUT: Working files: {args.output / 'working'}")
        logger.info(f"OUTPUT: Annotations: {args.output / 'annotations.jsonl'}")
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Stats: {stats['successful_annotations']}/{stats['total_sequences']} sequences annotated")
        logger.info(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
