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
from x_spanformer.xbar.analyze_annotations import AnnotationAnalyzer
from x_spanformer.xbar.xbar_dict import get_global_dict
from x_spanformer.xbar.xbar_map import XBarLabelMap

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
        
        # Track failed sequences for priority retry
        self.failed_sequence_ids = set()
    
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
            
            # Show range-specific message
            logger.info(f"Filtered to {len(filtered_sequences)}/{original_count} sequences")
            if len(filtered_sequences) == 1:
                seq_nums = sorted(target_sequence_ids)
                logger.info(f"Selected sequence {seq_nums[0]}")
            else:
                seq_nums = sorted([seq.meta.sequence_number if hasattr(seq.meta, 'sequence_number') else seq.meta.get('sequence_number', '?') for seq in filtered_sequences])
                min_seq = min(seq_nums) if seq_nums else None
                max_seq = max(seq_nums) if seq_nums else None
                logger.info(f"Selected {len(seq_nums)} sequences ({min_seq} to {max_seq}) out of {len(target_sequence_ids)} requested")
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
        self.failed_sequence_ids = set()  # Track failed sequences for priority retry
        
        for working_file in working_files:
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sequence_number = data.get("sequence_number", 0)
                span_annotations = data.get("span_annotations", [])
                status = data.get("status", "unknown")
                
                # Only consider it completed if it has spans AND status is completed
                if span_annotations and len(span_annotations) > 0 and status == "completed":
                    span_count = len(span_annotations)
                    total_existing_spans += span_count
                    completed_sequences += 1
                    existing_results[sequence_number] = "completed"
                else:
                    # Track for retry: either failed, no spans, or empty status
                    self.failed_sequence_ids.add(sequence_number)
                    # Remove working file to force retry
                    if not span_annotations or len(span_annotations) == 0:
                        logger.info(f"Removing empty working file for sequence {sequence_number}: {working_file.name}")
                    else:
                        logger.info(f"Removing failed working file for sequence {sequence_number}: {working_file.name}")
                    working_file.unlink()
                    failed_sequences += 1
                
            except Exception as e:
                # Extract sequence number from filename for tracking
                try:
                    seq_num = int(working_file.stem.split('-')[-1])
                    self.failed_sequence_ids.add(seq_num)
                except (ValueError, IndexError):
                    pass
                    
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
            
        # Log failed sequences that will be prioritized
        if hasattr(self, 'failed_sequence_ids') and self.failed_sequence_ids:
            logger.info(f"Will prioritize retry of {len(self.failed_sequence_ids)} previously failed sequences")
            
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
        """Build X-bar dictionaries and generate annotations.jsonl from working files."""
        working_dir = output_dir / "working"
        annotations_file = output_dir / "annotations.jsonl"
        
        working_files = list(working_dir.glob("*.json"))
        logger.info(f"Building X-bar dictionaries from {len(working_files)} working files")
        
        # Get global dictionary instance
        xbar_dict = get_global_dict()
        
        # Dictionary building structures
        domain_spans = {}  # domain -> level -> list of spans
        total_spans_processed = 0
        
        # Collect all annotations for annotations.jsonl
        all_annotations = []
        
        for working_file in sorted(working_files):
            try:
                with open(working_file, 'r', encoding='utf-8') as inf:
                    data = json.load(inf)
                
                if data.get("status") == "completed" and data.get("span_annotations"):
                    domain_type = data.get("domain_type", "unknown")
                    
                    # Initialize domain spans tracking
                    if domain_type not in domain_spans:
                        domain_spans[domain_type] = {
                            "word_level": [],
                            "phrase_level": [],
                            "clause_level": []
                        }
                    
                    for span_annotation in data["span_annotations"]:
                        # Validate span positions before processing
                        start_pos = span_annotation["start_pos"]
                        end_pos = span_annotation["end_pos"]
                        expected_text = span_annotation["text"]
                        raw_text = data["raw_text"]
                        xbar_label = span_annotation["xbar_label"]
                        
                        # Check if positions are valid and match expected text
                        if start_pos >= 0 and end_pos <= len(raw_text) and start_pos < end_pos:
                            # Check if the extracted text matches (use exclusive end for extraction)
                            actual_text = raw_text[start_pos:end_pos]
                            if actual_text == expected_text:
                                total_spans_processed += 1
                                
                                # Create annotation record for annotations.jsonl
                                annotation_record = {
                                    "id": len(all_annotations),
                                    "sequence_number": data["sequence_number"],
                                    "raw": data["raw_text"],
                                    "domain_type": domain_type,
                                    "start_pos": start_pos,
                                    "end_pos": end_pos,
                                    "xbar_label": xbar_label,
                                    "text": expected_text,
                                    "model": self.model_name,
                                    "timestamp": data.get("timestamp", "")
                                }
                                all_annotations.append(annotation_record)
                                
                                # Add to dictionary building - determine hierarchical level
                                hierarchical_level = self._determine_hierarchical_level(xbar_label)
                                if hierarchical_level:
                                    domain_spans[domain_type][hierarchical_level].append(expected_text)
                                    
                            else:
                                logger.warning(f"Text mismatch seq {data['sequence_number']}: expected '{expected_text}' at {start_pos}-{end_pos}")
                        else:
                            logger.warning(f"Invalid span bounds seq {data['sequence_number']}: {start_pos}-{end_pos} (text len {len(raw_text)})")
            
            except Exception as e:
                logger.warning(f"Consolidation error {working_file.name}: {e}")
        
        # Clean and validate labels before saving
        from x_spanformer.xbar.xbar_map import XBarLabelMap
        cleaned_annotations, mapping_stats = XBarLabelMap.clean_and_validate_labels(all_annotations)
        
        logger.info(f"Label cleaning results:")
        logger.info(f"  Valid labels (unchanged): {mapping_stats['valid']}")
        logger.info(f"  Invalid labels mapped: {mapping_stats['mapped']}")
        logger.info(f"  Invalid labels removed: {mapping_stats['removed']}")
        
        # Write annotations.jsonl file
        with open(annotations_file, 'w', encoding='utf-8') as f:
            for annotation in cleaned_annotations:
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        logger.info(f"Generated {len(cleaned_annotations)} annotation records in annotations.jsonl")
        
        # Build dictionaries from collected spans
        logger.info("Building X-bar dictionaries from processed spans...")
        total_new_spans = 0
        for domain_type, levels in domain_spans.items():
            new_counts = xbar_dict.add_sequence_spans(
                domain_type=domain_type,
                word_spans=levels["word_level"],
                phrase_spans=levels["phrase_level"],
                clause_spans=levels["clause_level"]
            )
            domain_new = sum(new_counts.values())
            total_new_spans += domain_new
            if domain_new > 0:
                logger.info(f"Added {domain_new} new unique spans for domain '{domain_type}' "
                           f"(word: {new_counts['word_level']}, phrase: {new_counts['phrase_level']}, "
                           f"clause: {new_counts['clause_level']})")
        
        logger.info(f"Processed {total_spans_processed} spans for dictionary building")
        logger.info(f"Dictionary building: {total_new_spans} new unique spans added across all domains")
        
        # Save dictionaries to spans.jsonl
        xbar_dict.save_dictionaries(output_dir)
        
        # Log dictionary statistics
        xbar_dict.log_statistics()
    
    def _determine_hierarchical_level(self, xbar_label: str) -> Optional[str]:
        """
        Determine hierarchical level from X-bar label.
        
        Args:
            xbar_label: The X-bar label
            
        Returns:
            Hierarchical level or None if unknown
        """
        # Word-level labels
        word_labels = {
            'noun', 'verb', 'adjective', 'adverb', 'determiner', 'preposition', 
            'pronoun', 'conjunction', 'punctuation', 'keyword', 'identifier', 
            'operator', 'literal', 'delimiter', 'type_name', 'comment'
        }
        
        # Phrase-level labels
        phrase_labels = {
            'noun_phrase', 'verb_phrase', 'adjective_phrase', 'adverb_phrase', 
            'prepositional_phrase', 'expression', 'function_call', 'assignment', 
            'parameter_list', 'argument_list', 'inline_code', 'code_block'
        }
        
        # Clause-level labels
        clause_labels = {
            'main_clause', 'subordinate_clause', 'relative_clause', 'if_statement', 
            'loop_statement', 'function_definition', 'class_definition', 
            'import_statement', 'return_statement', 'documentation_comment'
        }
        
        # Normalize label for checking
        normalized_label = xbar_label.lower().strip()
        
        if normalized_label in word_labels:
            return "word_level"
        elif normalized_label in phrase_labels:
            return "phrase_level"
        elif normalized_label in clause_labels:
            return "clause_level"
        else:
            # Use the helper function from xbar_map for unknown labels
            mapped_level = XBarLabelMap.get_hierarchical_level(xbar_label)
            if mapped_level:
                logger.debug(f"Mapped unknown label '{xbar_label}' to '{mapped_level}'")
                return mapped_level
            else:
                logger.warning(f"Unknown hierarchical level for label: {xbar_label}")
                return None
    
    def update_metadata(self, output_dir: Path):
        """Update global metadata file with dictionary statistics."""
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
        
        # Get dictionary statistics
        from x_spanformer.xbar.xbar_dict import get_global_dict
        xbar_dict = get_global_dict()
        
        # Get comprehensive dictionary stats
        all_stats = xbar_dict.get_all_stats()
        
        dict_stats = {
            "total_unique_spans": all_stats.get("total_unique_spans", 0),
            "domain_distribution": all_stats.get("domain_totals", {}),
            "level_distribution": all_stats.get("level_totals", {}),
            "detailed_breakdown": all_stats.get("detailed_breakdown", {})
        }
        
        metadata = {
            "pipeline": "dictionary_builder",
            "model": self.model_name,
            "last_updated": datetime.now().isoformat(),
            "processing_stats": {
                "total_sequences": self.pipeline_stats["total_sequences"],
                "successful_annotations": successful_count,
                "failed_annotations": failed_count,
                "total_raw_spans": total_spans,
                "success_rate": successful_count / max(1, successful_count + failed_count)
            },
            "dictionary_stats": dict_stats,
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
        
        # Initialize or load global dictionary
        xbar_dict = get_global_dict()
        logger.info("Initializing X-bar dictionary system...")
        
        # Try to load existing dictionaries if they exist
        try:
            xbar_dict.load_dictionaries(output_dir)
            logger.info("Loaded existing X-bar dictionaries")
        except Exception as e:
            logger.debug(f"No existing dictionaries to load: {e}")
        
        # Load sequences
        sequences = self.load_sequences(corpus_file, range_spec)
        self.pipeline_stats["total_sequences"] = len(sequences)
        
        if not sequences:
            logger.warning("No sequences found to process")
            return self.pipeline_stats
        
        # Load existing results for resume (always enabled)
        existing_results = self.load_existing_results(output_dir)
        
        # Filter sequences to process with priority for previously failed sequences
        sequences_to_process = []
        failed_sequences_to_retry = []
        new_sequences_to_process = []
        
        for seq in sequences:
            # Get sequence number using consistent helper method
            seq_id = self.get_sequence_number(seq)
                    
            if seq_id not in existing_results or existing_results[seq_id] != "completed":
                # Prioritize previously failed sequences
                if hasattr(self, 'failed_sequence_ids') and seq_id in self.failed_sequence_ids:
                    failed_sequences_to_retry.append(seq)
                else:
                    new_sequences_to_process.append(seq)
        
        # Process failed sequences first, then new sequences
        sequences_to_process = failed_sequences_to_retry + new_sequences_to_process
        
        skipped_count = len(sequences) - len(sequences_to_process)
        if skipped_count > 0:
            logger.info(f"Processing {len(sequences_to_process)} sequences (skipped {skipped_count} completed)")
            if failed_sequences_to_retry:
                logger.info(f"  - {len(failed_sequences_to_retry)} previously failed sequences (prioritized)")
            if new_sequences_to_process:
                logger.info(f"  - {len(new_sequences_to_process)} new sequences")
        else:
            logger.info(f"Processing {len(sequences_to_process)} sequences")
        
        # Progress tracking
        async def progress_callback(progress_info):
            pass  # Simplified - main progress logging handled in loop
        
        # Process sequences individually (sequential processing)
        successful_count = 0
        failed_count = 0
        skipped_count = 0  # Track sequences skipped due to errors (including JSON parsing)
        session_start_time = datetime.now()
        total_completed_already = len(existing_results)
        
        for i, sequence in enumerate(sequences_to_process):
            # Get sequence number using consistent helper method
            seq_id = self.get_sequence_number(sequence)
            
            # Calculate current position in total sequences (already completed + current position)
            current_position = total_completed_already + i + 1
            total_sequences_to_process = len(sequences_to_process)
            total_sequences_overall = self.pipeline_stats["total_sequences"]  # Use the original total from load_sequences
            
            logger.info(f"Processing {current_position}/{total_sequences_overall}: sequence {seq_id}")
            
            try:
                # Annotate single sequence
                result = await self.session.annotate_single_sequence(sequence, progress_callback=progress_callback)
                
                if result.success and result.annotation_record:
                    annotation_record = result.annotation_record
                    span_count = len(annotation_record.span_annotations)
                    
                    # Check if any spans were actually extracted
                    if span_count > 0:
                        self.save_working_file(output_dir, sequence, annotation_record)
                        successful_count += 1
                        
                        # Update total span count in pipeline stats
                        self.pipeline_stats["total_spans"] += span_count
                        logger.info(f"SUCCESS: Sequence {seq_id}: {span_count} spans annotated")
                    else:
                        # No spans extracted - treat as skipped for retry
                        skipped_count += 1
                        logger.warning(f"SKIPPED: Sequence {seq_id}: No spans extracted - will retry on next run")
                        # Don't save working file so sequence will be retried
                        continue
                    
                    # Calculate and log progress summary
                    current_time = datetime.now()
                    elapsed_seconds = (current_time - session_start_time).total_seconds()
                    completed_count = successful_count + failed_count  # Exclude skipped_count from ETA calculation
                    remaining_count = len(sequences_to_process) - completed_count  # Skipped sequences still need to be processed
                    total_completed_overall = total_completed_already + completed_count
                    
                    if completed_count > 0 and elapsed_seconds > 0:
                        avg_time_per_sequence = elapsed_seconds / completed_count
                        sequences_per_minute = (completed_count / elapsed_seconds) * 60
                        eta_seconds = remaining_count * avg_time_per_sequence
                        eta_minutes = eta_seconds / 60
                        eta_formatted = format_eta_time(eta_minutes)
                        
                        logger.info("=" * 80)
                        logger.info(f"{total_completed_overall}/{total_sequences_overall} sequences completed | "
                                  f"Avg: {sequences_per_minute:.1f} seq/min | "
                                  f"ETA: {eta_formatted}")
                        logger.info(f"Total spans annotated: {self.pipeline_stats['total_spans']} | "
                                  f"Skipped: {skipped_count}")
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
                    completed_count = successful_count + failed_count  # Exclude skipped_count from ETA calculation
                    remaining_count = len(sequences_to_process) - completed_count  # Skipped sequences still need to be processed
                    total_completed_overall = total_completed_already + completed_count
                    
                    if completed_count > 0 and elapsed_seconds > 0:
                        avg_time_per_sequence = elapsed_seconds / completed_count
                        sequences_per_minute = (completed_count / elapsed_seconds) * 60
                        eta_seconds = remaining_count * avg_time_per_sequence
                        eta_minutes = eta_seconds / 60
                        eta_formatted = format_eta_time(eta_minutes)
                        
                        logger.info("=" * 80)
                        logger.info(f"{total_completed_overall}/{total_sequences_overall} sequences completed | "
                                  f"Avg: {sequences_per_minute:.1f} seq/min | "
                                  f"ETA: {eta_formatted}")
                        logger.info(f"Total spans annotated: {self.pipeline_stats['total_spans']} | "
                                  f"Skipped: {skipped_count}")
                        logger.info("=" * 80)
                    
            except ValueError as e:
                # JSON parsing errors - skip sequence without saving working file
                skipped_count += 1
                logger.warning(f"SKIPPED: Sequence {seq_id}: {str(e)}")
                logger.info(f"Sequence {seq_id} will be retried on next pipeline run")
                # Don't increment failed_count or save working file - just continue to next sequence
                continue
                
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
                    logger.info(f"{total_completed_overall}/{total_sequences_overall} sequences completed | "
                              f"Avg: {sequences_per_minute:.1f} seq/min | "
                              f"ETA: {eta_formatted}")
                    logger.info(f"Total spans annotated: {self.pipeline_stats['total_spans']} | "
                              f"Skipped: {skipped_count}")
                    logger.info("=" * 80)
        
        # Update statistics
        self.pipeline_stats["processed_sequences"] = len(sequences_to_process)
        self.pipeline_stats["successful_annotations"] = successful_count
        self.pipeline_stats["failed_annotations"] = failed_count
        self.pipeline_stats["skipped_annotations"] = skipped_count
        self.pipeline_stats["completed_at"] = datetime.now().isoformat()
        
        # Check if pipeline exited early due to failures
        processed_count = successful_count + failed_count + skipped_count
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
        
        # Run annotation analysis if annotations were created
        annotations_file = output_dir / "annotations.jsonl"
        if annotations_file.exists():
            try:
                logger.info("Running annotation analysis...")
                analyzer = AnnotationAnalyzer(str(annotations_file))
                analyzer.analyze_and_report(str(output_dir))
            except Exception as e:
                logger.warning(f"Annotation analysis failed: {e}")
        
        # Final status summary
        if failed_count >= MAX_TOTAL_FAILURES:
            logger.critical(f"Pipeline terminated due to {failed_count} failures. Failed sequences will be retried on next run.")
        else:
            logger.info(f"Pipeline completed: {successful_count} successful, {failed_count} failed, {skipped_count} skipped")
        
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
        
        # Display final results with clearer distinction between total and session counts
        logger.info("="*60)
        logger.info("PIPELINE COMPLETION SUMMARY")
        logger.info("="*60)
        
        # Get total stats for clarity
        total_in_range = stats['total_sequences']
        processed_this_session = stats['processed_sequences'] 
        already_completed = total_in_range - processed_this_session
        successful_this_session = stats['successful_annotations']
        failed_this_session = stats['failed_annotations']
        skipped_this_session = stats.get('skipped_annotations', 0)
        
        # Show range statistics
        logger.info(f"Sequences in range: {total_in_range}")
        logger.info(f"Already completed: {already_completed}")
        logger.info(f"Processed this session: {processed_this_session}")
        logger.info(f"  - Successful: {successful_this_session}")
        logger.info(f"  - Failed: {failed_this_session}")
        logger.info(f"  - Skipped: {skipped_this_session}")
        
        # Calculate span count for sequences in the requested range
        total_spans_in_range = 0
        working_dir = args.output / "working"
        
        if working_dir.exists():
            # Load sequences to get the actual sequence numbers in range
            sequences = pipeline.load_sequences(args.corpus, args.range)
            requested_seq_ids = {pipeline.get_sequence_number(seq) for seq in sequences}
            
            # Count spans only from sequences in the requested range
            for working_file in working_dir.glob("*.json"):
                try:
                    with open(working_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    seq_id = data.get("sequence_number")
                    if seq_id in requested_seq_ids and data.get("status") == "completed":
                        span_count = len(data.get("span_annotations", []))
                        total_spans_in_range += span_count
                except Exception:
                    continue  # Skip invalid files
        
        # Session success rate (for sequences processed this session)
        session_success_rate = (successful_this_session / processed_this_session) if processed_this_session > 0 else 0
        
        # Average spans per sequence (for successfully processed sequences this session)
        session_total_spans = stats.get('total_spans', 0)  # Spans from this session only
        avg_spans_this_session = (session_total_spans / successful_this_session) if successful_this_session > 0 else 0
        
        # Average spans across all completed sequences in range
        completed_sequences_in_range = already_completed + successful_this_session
        avg_spans_overall = (total_spans_in_range / completed_sequences_in_range) if completed_sequences_in_range > 0 else 0
        
        logger.info(f"Total spans in requested range: {total_spans_in_range}")
        logger.info(f"Session success rate: {session_success_rate:.2%}")
        logger.info(f"Avg spans/sequence (this session): {avg_spans_this_session:.1f}")
        logger.info(f"Avg spans/sequence (overall range): {avg_spans_overall:.1f}")
        
        logger.info(f"Results: {args.output}")
        logger.info(f"Working files: {args.output / 'working'}")
        logger.info(f"Annotations: {args.output / 'annotations.jsonl'}")
        
        # Check if pipeline failed due to too many failures
        if stats.get('early_exit', False) and stats['failed_annotations'] >= MAX_TOTAL_FAILURES:
            logger.critical(f"Pipeline terminated due to failure limit. Exiting with error code.")
            sys.exit(1)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Session stats: {successful_this_session}/{processed_this_session} sequences processed successfully")
        logger.info(f"Overall progress: {completed_sequences_in_range}/{total_in_range} sequences complete")
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
