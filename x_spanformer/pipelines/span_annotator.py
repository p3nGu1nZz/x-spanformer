"""
Span Annotator Pipeline for X-Spanformer

Implements agentic X-bar span annotation for generating supervised training data
for the factorized pointer network span predictor. Processes raw Unicode sequences
from corpus.jsonl through multi-turn conversations with LLM agents to generate
hierarchical X-bar span boundary annotations.

Usage:
    python -m x_spanformer.pipelines.span_annotator \
        --corpus data/vocab/corpus.jsonl \
        --output data/annotations \
        --range 1-100 \
        --config config/pipelines/span_annotator.yaml \
        --agent config/agents/span_annotator_agent.yaml
"""

import asyncio
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sys

from x_spanformer.agents.session.span_annotator_session import SpanAnnotatorSession
from x_spanformer.agents.ollama_client import check_ollama_connection
from x_spanformer.xbar.position_mapper import PositionMapper
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord, AnnotationBatch
from x_spanformer.config.span_annotator_config_loader import load_config
from x_spanformer.pipelines.shared.annotation_processor import AnnotationProcessor
from x_spanformer.pipelines.shared.pipeline_telemetry import SpanAnnotationTelemetry
from x_spanformer.pipelines.shared.pipeline_logging import PipelineLogger

logger = logging.getLogger(__name__)


class SpanAnnotatorPipeline:
    """
    Main pipeline for span annotation using async LLM agents.
    
    Processes corpus.jsonl sequences through comprehensive X-bar span annotation
    using multi-turn conversations with language models.
    """
    
    def __init__(self, config_path: Optional[str] = None, agent_config_path: Optional[str] = None):
        """Initialize span annotator pipeline."""
        # Load pipeline configuration
        self.config = load_config(config_path or "config/pipelines/span_annotator.yaml")
        self.annotation_processor = AnnotationProcessor()
        
        # Load agent configuration if provided
        agent_config_path = agent_config_path or "config/agents/span_annotator_agent.yaml"
        agent_config_file = Path(agent_config_path)
        
        if not agent_config_file.exists():
            raise FileNotFoundError(f"Agent configuration file not found: {agent_config_file}")
        
        try:
            import yaml
            with open(agent_config_file, 'r', encoding='utf-8') as f:
                agent_config = yaml.safe_load(f)
            
            if not agent_config:
                raise ValueError(f"Agent configuration file is empty or invalid: {agent_config_file}")
            
            model_config = agent_config.get("model", {})
            if not model_config:
                raise ValueError(f"Missing 'model' section in agent config: {agent_config_file}")
            
            model_name = model_config.get("name")
            if not model_name:
                raise ValueError(f"Missing 'model.name' in agent config: {agent_config_file}")
            
            temperature = model_config.get("temperature")
            if temperature is None:
                raise ValueError(f"Missing 'model.temperature' in agent config: {agent_config_file}")
            
            # Extract dialogue configuration
            dialogue_config = agent_config.get("dialogue", {})
            if not dialogue_config:
                raise ValueError(f"Missing 'dialogue' section in agent config: {agent_config_file}")
            
            max_turns = dialogue_config.get("max_turns")
            if max_turns is None:
                raise ValueError(f"Missing 'dialogue.max_turns' in agent config: {agent_config_file}")
            
            # Extract early termination config
            early_termination_config = agent_config.get("agent", {}).get("early_termination")
            
            # Extract agent behavior settings
            agent_behavior = agent_config.get("agent", {})
            max_spans_per_sequence = agent_behavior.get("max_spans_per_sequence", 64)  # Default to 64
            
        except Exception as e:
            raise RuntimeError(f"Failed to load agent configuration from {agent_config_file}: {e}")
        
        # Initialize span annotator session
        self.agent = SpanAnnotatorSession(
            model_name=model_name,
            max_concurrent=1,  # Process one at a time
            max_retries=self.config.processing.max_retries,
            conversation_timeout=self.config.processing.conversation_timeout,
            max_turns=max_turns,
            temperature=temperature,
            early_termination_config=early_termination_config,
            max_spans_per_sequence=max_spans_per_sequence
        )
        
        # Processing statistics
        self.stats = {
            "total_sequences": 0,
            "processed_sequences": 0,
            "successful_annotations": 0,
            "failed_annotations": 0,
            "total_spans": 0,
            "processing_time": 0.0,
            "started_at": None,
            "completed_at": None,
            "consecutive_failures": 0,  # Track consecutive failures
            "max_consecutive_failures": 3  # Exit after 3 consecutive failures
        }
        
        # Initialize shared telemetry
        self.telemetry = SpanAnnotationTelemetry()
        
        # Telemetry display settings
        self.last_telemetry_display = None
        self.telemetry_display_interval = 30  # Show telemetry every 30 seconds
        self.sequences_per_telemetry_display = 5  # Or every 5 sequences
    
    def should_display_telemetry(self, force: bool = False) -> bool:
        """
        Determine if telemetry should be displayed based on time or sequence count.
        
        Args:
            force: Force display regardless of timing
            
        Returns:
            True if telemetry should be displayed
        """
        current_time = datetime.now()
        
        if force:
            return True
        
        # Display every N sequences
        processed_sequences = self.telemetry.telemetry["completed_sequences"] + self.telemetry.telemetry["failed_sequences"]
        if processed_sequences > 0 and processed_sequences % self.sequences_per_telemetry_display == 0:
            return True
        
        # Display at time intervals
        if self.last_telemetry_display is None:
            return True
        
        time_since_last = (current_time - self.last_telemetry_display).total_seconds()
        if time_since_last >= self.telemetry_display_interval:
            return True
        
        return False
    
    def display_telemetry_if_needed(self, force: bool = False, output_dir: Optional[Path] = None):
        """
        Display telemetry panel if conditions are met and optionally save to metadata.
        
        Args:
            force: Force display regardless of timing
            output_dir: Output directory to save telemetry to metadata.json (optional)
        """
        should_display = self.should_display_telemetry(force)
        processed_sequences = self.telemetry.telemetry["completed_sequences"] + self.telemetry.telemetry["failed_sequences"]
        
        # Debug logging to understand why telemetry isn't displaying
        if force or processed_sequences > 0:
            logger.info(f"Telemetry check: processed={processed_sequences}, force={force}, should_display={should_display}")
        
        if should_display:
            logger.info("Displaying telemetry progress panel...")
            self.telemetry.display_progress_panel()
            self.last_telemetry_display = datetime.now()
            
            # Save telemetry to metadata.json if output directory provided
            if output_dir:
                try:
                    metadata_file = output_dir / "metadata.json"
                    self.telemetry.save_telemetry_to_metadata(metadata_file)
                except Exception as e:
                    logger.warning(f"Failed to save telemetry to metadata: {e}")
        elif processed_sequences > 0:  # Only log when we've actually processed sequences
            logger.info(f"Telemetry not displayed: processed={processed_sequences}, last_display={self.last_telemetry_display}")
    
    def display_current_stats(self):
        """
        Display current processing statistics on demand.
        Useful for debugging or monitoring.
        """
        logger.info("=" * 60)
        logger.info("CURRENT PIPELINE STATUS")
        logger.info("=" * 60)
        self.telemetry.display_progress_panel()
        
        # Additional pipeline-specific stats
        stats = self.telemetry.get_statistics()
        logger.info(f"Pipeline Stats:")
        logger.info(f"  - Total sequences to process: {stats['total_sequences']}")
        logger.info(f"  - Completed this session: {len(self.telemetry.telemetry['sequence_times'])}")
        logger.info(f"  - Success rate this session: {stats['success_rate_percent']:.1f}%")
        logger.info(f"  - Average time per sequence: {stats['average_sequence_time_seconds']:.1f}s")
        
        if stats['processing_rate_per_min'] > 0:
            logger.info(f"  - Current processing rate: {stats['processing_rate_per_min']:.2f} seq/min")
        
        logger.info("=" * 60)
    
    def get_all_corpus_sequences(self, corpus_file: Path) -> List[int]:
        """
        Get all sequence numbers from the corpus file.
        
        Args:
            corpus_file: Path to corpus.jsonl file
            
        Returns:
            List of all sequence IDs in the corpus
        """
        sequence_ids = []
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Try to get sequence_number from data, fallback to line number
                    seq_num = data.get("meta", {}).get("sequence_number", line_num)
                    sequence_ids.append(seq_num)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        return sorted(sequence_ids)
    
    def parse_range_specification(self, range_spec: str) -> List[int]:
        """
        Parse range specification into list of sequence IDs.
        
        Supports formats:
        - Single: "42"
        - List: "1,5,10"  
        - Range: "1-100"
        - Mixed: "1-10,15,20-25"
        
        Args:
            range_spec: Range specification string
            
        Returns:
            List of sequence IDs to process
        """
        sequence_ids = []
        
        for part in range_spec.split(','):
            part = part.strip()
            
            if '-' in part:
                # Range specification
                start, end = part.split('-', 1)
                sequence_ids.extend(range(int(start), int(end) + 1))
            else:
                # Single sequence ID
                sequence_ids.append(int(part))
        
        return sorted(list(set(sequence_ids)))  # Remove duplicates and sort
        """
        Parse range specification into list of sequence IDs.
        
        Supports formats:
        - Single: "42"
        - List: "1,5,10"  
        - Range: "1-100"
        - Mixed: "1-10,15,20-25"
        
        Args:
            range_spec: Range specification string
            
        Returns:
            List of sequence IDs to process
        """
        sequence_ids = []
        
        for part in range_spec.split(','):
            part = part.strip()
            
            if '-' in part:
                # Range specification
                start, end = part.split('-', 1)
                sequence_ids.extend(range(int(start), int(end) + 1))
            else:
                # Single sequence ID
                sequence_ids.append(int(part))
        
        return sorted(list(set(sequence_ids)))  # Remove duplicates and sort
    
    def load_all_sequences(self, corpus_file: Path) -> List[PretrainRecord]:
        """
        Load all sequences from corpus file.
        
        Args:
            corpus_file: Path to corpus.jsonl file
            
        Returns:
            List of all PretrainRecord objects in the corpus
        """
        logger.info(f"Loading all sequences from {corpus_file}")
        
        all_sequences = []
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    record = PretrainRecord(**data)
                    
                    # Ensure sequence_number is set for embedding alignment
                    if record.sequence_number is None:
                        record.sequence_number = line_num
                    
                    all_sequences.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(all_sequences)} sequences for processing")
        return all_sequences

    def load_target_sequences(
        self, 
        corpus_file: Path, 
        sequence_ids: List[int]
    ) -> List[PretrainRecord]:
        """
        Load target sequences from corpus file.
        
        Args:
            corpus_file: Path to corpus.jsonl file
            sequence_ids: List of sequence IDs to load
            
        Returns:
            List of PretrainRecord objects for target sequences
        """
        logger.info(f"Loading {len(sequence_ids)} target sequences from {corpus_file}")
        
        target_sequences = []
        sequence_id_set = set(sequence_ids)
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    record = PretrainRecord(**data)
                    
                    # Check if this sequence is in our target set
                    seq_num = record.sequence_number or line_num
                    
                    if seq_num in sequence_id_set:
                        # Ensure sequence_number is set for embedding alignment
                        if record.sequence_number is None:
                            record.sequence_number = seq_num
                        
                        target_sequences.append(record)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(target_sequences)} sequences for processing")
        return target_sequences
    
    def ensure_output_structure(self, output_dir: Path):
        """Ensure output directory structure exists."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "working").mkdir(exist_ok=True)
        
        # Initialize annotations file if it doesn't exist
        annotations_file = output_dir / "annotations.jsonl"
        if not annotations_file.exists():
            annotations_file.touch()
        
        # Initialize metadata file if it doesn't exist
        metadata_file = output_dir / "metadata.json"
        if not metadata_file.exists():
            initial_metadata = {
                "pipeline_version": "1.0",
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_sequences": 0,
                "processed_sequences": 0,
                "successful_annotations": 0,
                "failed_annotations": 0,
                "processing_stats": {},
                "active_processes": []
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(initial_metadata, f, indent=2)
    
    def load_existing_results(self, output_dir: Path) -> Dict[int, str]:
        """
        Load existing annotation results to enable resume functionality.
        
        Args:
            output_dir: Output directory containing working files
            
        Returns:
            Dict mapping sequence_id -> status (completed, failed, partial)
        """
        existing_results = {}
        working_dir = output_dir / "working"
        
        if not working_dir.exists():
            logger.info("No working directory found - starting fresh")
            return existing_results
        
        working_files = list(working_dir.glob("corpus-seq-*.json"))
        if not working_files:
            logger.info("No existing working files found - starting fresh")
            return existing_results
        
        logger.info(f"Loading existing results from {len(working_files)} working files...")
        
        completed_count = 0
        failed_count = 0
        incomplete_count = 0
        invalid_count = 0

        for working_file in working_files:
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sequence_number = data.get("sequence_number")
                annotation_session = data.get("annotation_session", {})
                annotation_status = annotation_session.get("annotation_status", "unknown")
                spans_extracted = annotation_session.get("spans_extracted", 0)
                
                if sequence_number is not None:
                    # Validate that "completed" sequences actually have spans
                    if annotation_status == "completed" and spans_extracted > 0:
                        existing_results[sequence_number] = "completed"
                        completed_count += 1
                    elif annotation_status == "completed" and spans_extracted == 0:
                        # Mark as failed if no spans were extracted despite "completed" status
                        existing_results[sequence_number] = "failed"
                        failed_count += 1
                        logger.warning(f"Sequence {sequence_number} marked as completed but has 0 spans - treating as failed")
                    elif annotation_status == "failed":
                        existing_results[sequence_number] = "failed"
                        failed_count += 1
                    elif annotation_status.startswith("in_progress"):
                        # Mark incomplete sequences as failed to retry them
                        existing_results[sequence_number] = "failed"
                        incomplete_count += 1
                        logger.info(f"Sequence {sequence_number} was incomplete ({annotation_status}) - will retry")
                    else:
                        # Any other status - treat as failed to be safe
                        existing_results[sequence_number] = "failed"
                        failed_count += 1
                        logger.warning(f"Sequence {sequence_number} has unclear status '{annotation_status}' - treating as failed")
                else:
                    invalid_count += 1
                    logger.warning(f"Working file {working_file.name} missing sequence_number - skipping")
                    
            except Exception as e:
                invalid_count += 1
                logger.warning(f"Failed to load existing result from {working_file}: {e}")
        
        # Summary logging
        if existing_results:
            min_seq = min(existing_results.keys())
            max_seq = max(existing_results.keys())
            logger.info(f"Resume Summary:")
            logger.info(f"  - Found {completed_count} completed sequences")
            logger.info(f"  - Found {failed_count} failed sequences (will retry)")
            logger.info(f"  - Found {incomplete_count} incomplete sequences (will retry)")
            logger.info(f"  - Found {invalid_count} invalid/unreadable files")
            logger.info(f"  - Sequence range: {min_seq} to {max_seq}")
            logger.info(f"  - Total sequences to process: {len(existing_results)}")
        
        return existing_results
    
    def find_missing_sequences(self, target_sequences: List[PretrainRecord], existing_results: Dict[int, str]) -> List[int]:
        """
        Find sequences that are missing within the processed range (gaps).
        Only considers sequences as missing if they fall within the range of processed sequences.
        
        Args:
            target_sequences: List of all sequences to process
            existing_results: Dict mapping sequence_id -> status for sequences with working files
            
        Returns:
            List of sequence numbers that represent gaps in processing
        """
        if not existing_results:
            # Fresh run - no sequences are "missing", they're just unprocessed
            logger.info("Fresh run detected - no existing working files found")
            return []
        
        # Find the highest processed sequence ID
        max_processed_id = max(existing_results.keys()) if existing_results else 0
        
        # Get all target sequence IDs within the processed range
        target_sequence_ids = {record.sequence_number for record in target_sequences 
                             if record.sequence_number is not None and record.sequence_number <= max_processed_id}
        
        # Find gaps: sequences within processed range that don't have working files
        missing_sequence_numbers = []
        for seq_id in target_sequence_ids:
            if seq_id not in existing_results:
                missing_sequence_numbers.append(seq_id)
        
        if missing_sequence_numbers:
            logger.info(f"Found {len(missing_sequence_numbers)} gap sequences (within processed range 1-{max_processed_id})")
            if len(missing_sequence_numbers) <= 20:
                logger.info(f"Gap sequences: {sorted(missing_sequence_numbers)}")
            else:
                sample = sorted(missing_sequence_numbers)[:10]
                logger.info(f"Gap sequences (sample): {sample}... and {len(missing_sequence_numbers) - 10} more")
        else:
            logger.info(f"No gaps found within processed range (1-{max_processed_id})")
        
        return missing_sequence_numbers
    
    def save_working_file(
        self, 
        output_dir: Path, 
        record: PretrainRecord, 
        annotation_result: Optional[AnnotationRecord] = None,
        error_message: Optional[str] = None,
        status: str = "in_progress",
        skip_annotations_write: bool = False
    ):
        """Save annotation result to individual working file and append to consolidated annotations."""
        working_file = output_dir / "working" / f"corpus-seq-{record.sequence_number:08d}.json"
        
        # Ensure working directory exists
        working_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine final status
        if annotation_result is not None:
            final_status = "completed"
        elif error_message is not None:
            final_status = "failed"
        else:
            final_status = status
        
        working_data = {
            "corpus_id": record.id.id if record.id else f"corpus-seq-{record.sequence_number:08d}",
            "sequence_number": record.sequence_number,
            "raw_sequence": record.raw,
            "domain_type": record.type,
            "source_meta": {
                "status": record.meta.status if record.meta else "unknown",
                "source": record.meta.source_file if record.meta else "unknown"
            },
            "annotation_session": {
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat() if final_status in ["completed", "failed"] else None,
                "annotation_status": final_status,
                "model": self.agent.model_name,
                "error_message": error_message,
                "processing_time": annotation_result.agent_metadata.get("processing_time", 0) if annotation_result and annotation_result.agent_metadata else 0,
                "turns_used": annotation_result.agent_metadata.get("turns_required", 0) if annotation_result and annotation_result.agent_metadata else 0,
                "spans_extracted": len(annotation_result.span_annotations) if annotation_result else 0
            }
        }
        
        if annotation_result:
            # Store full annotation details for working file including conversation and spans
            working_data["annotation_details"] = {
                "conversation_turns": annotation_result.conversation_turns if hasattr(annotation_result, 'conversation_turns') else [],
                "individual_spans": [
                    {
                        "start_pos": span.start_pos,
                        "end_pos": span.end_pos,
                        "xbar_class": span.xbar_class,
                        "confidence": span.confidence,
                        "text": span.linguistic_features.get("text", "") if span.linguistic_features else "",
                        "length": span.linguistic_features.get("length", span.end_pos - span.start_pos) if span.linguistic_features else span.end_pos - span.start_pos,
                        "linguistic_features": span.linguistic_features or {}
                    }
                    for span in annotation_result.span_annotations
                ],
                "agent_metadata": annotation_result.agent_metadata or {}
            }
            
            # Also keep the summary for quick overview
            working_data["annotation_summary"] = {
                "total_spans": len(annotation_result.span_annotations),
                "span_types": list(set(span.xbar_class for span in annotation_result.span_annotations)),
                "coverage_stats": {
                    "text_length": annotation_result.total_positions,
                    "annotated_characters": sum(span.linguistic_features.get("length", 0) if span.linguistic_features else 0 for span in annotation_result.span_annotations),
                    "coverage_ratio": sum(span.linguistic_features.get("length", 0) if span.linguistic_features else 0 for span in annotation_result.span_annotations) / annotation_result.total_positions if annotation_result.total_positions > 0 else 0
                },
                "conversation_efficiency": {
                    "turns_required": annotation_result.agent_metadata.get("turns_required", 0) if annotation_result.agent_metadata else 0,
                    "spans_per_turn": len(annotation_result.span_annotations) / max(1, annotation_result.agent_metadata.get("turns_required", 1)) if annotation_result.agent_metadata else 0,
                    "processing_time_per_span": annotation_result.agent_metadata.get("processing_time", 0) / max(1, len(annotation_result.span_annotations)) if annotation_result.agent_metadata else 0
                }
            }
            
            # Only append to annotations file when completely finished and not skipping
            if final_status == "completed" and not skip_annotations_write:
                self.append_to_annotations_file(output_dir, annotation_result)
        
        with open(working_file, 'w', encoding='utf-8') as f:
            json.dump(working_data, f, indent=2, ensure_ascii=False)
            
        logger.debug(f"Saved working file for sequence {record.sequence_number} with status: {final_status}")
    
    def append_to_annotations_file(self, output_dir: Path, annotation_result: AnnotationRecord):
        """Append unique span records to the main annotations.jsonl file with automatic deduplication."""
        annotations_file = output_dir / "annotations.jsonl"
        
        # Load existing spans for this sequence to check for duplicates
        existing_spans = set()
        if annotations_file.exists():
            with open(annotations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            if record.get('sequence_id') == annotation_result.sequence_id:
                                # Create a key based on position to identify duplicates
                                span_key = (
                                    record['span_annotation']['start_pos'],
                                    record['span_annotation']['end_pos']
                                )
                                existing_spans.add(span_key)
                        except json.JSONDecodeError:
                            continue
        
        # Extract essential information for each span
        raw_text = annotation_result.raw
        sequence_id = annotation_result.sequence_id
        
        # Get domain type from annotation metadata - check multiple sources
        domain_type = "unknown"
        
        # First try to get from meta tags
        if annotation_result.meta and annotation_result.meta.tags:
            for tag in annotation_result.meta.tags:
                if tag in ["natural", "code", "mixed"]:
                    domain_type = tag
                    break
        
        # If not found in tags, check agent metadata or fall back to "mixed"
        if domain_type == "unknown" and annotation_result.agent_metadata:
            domain_type = annotation_result.agent_metadata.get("domain_type", "mixed")
        
        # Final fallback
        if domain_type == "unknown":
            domain_type = "mixed"
        
        # Deduplicate spans by position - keep unique (start_pos, end_pos) combinations
        unique_spans = {}
        duplicate_count = 0
        
        for span in annotation_result.span_annotations:
            span_key = (span.start_pos, span.end_pos)
            
            if span_key not in unique_spans:
                unique_spans[span_key] = span
            else:
                duplicate_count += 1
                # Keep the span with higher confidence if available
                existing_span = unique_spans[span_key]
                if (hasattr(span, 'confidence') and hasattr(existing_span, 'confidence') and 
                    span.confidence > existing_span.confidence):
                    unique_spans[span_key] = span
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate spans for sequence {sequence_id} (kept {len(unique_spans)} unique)")
        
        # Create one record per unique span and append to file (skip existing duplicates)
        new_spans_added = 0
        with open(annotations_file, 'a', encoding='utf-8') as f:
            for span in unique_spans.values():
                span_key = (span.start_pos, span.end_pos)
                
                # Skip if this span already exists in the file for this sequence
                if span_key in existing_spans:
                    continue
                
                # Extract text and length from linguistic features if available
                linguistic_features = span.linguistic_features or {}
                span_text = linguistic_features.get("text", "")
                span_length = linguistic_features.get("length", span.end_pos - span.start_pos)
                
                # Log single-character spans for analysis
                if span_length == 1:
                    logger.debug(f"Single-char span: '{span_text}' ({span.xbar_class}) at pos {span.start_pos}-{span.end_pos}")
                
                span_record = {
                    "raw": raw_text,
                    "sequence_id": sequence_id,
                    "type": domain_type,
                    "span_annotation": {
                        "start_pos": span.start_pos,
                        "end_pos": span.end_pos,
                        "xbar_class": span.xbar_class,
                        "text": span_text,
                        "length": span_length
                    },
                    "total_positions": annotation_result.total_positions
                }
                f.write(json.dumps(span_record, ensure_ascii=False) + '\n')
                new_spans_added += 1
                
        logger.info(f"Appended {new_spans_added} unique span records to {annotations_file}")
    
    def update_global_metadata(
        self, 
        output_dir: Path, 
        processed_count: int, 
        successful_count: int,
        failed_count: int,
        total_spans: int
    ):
        """Update global metadata file with processing progress and quality metrics."""
        metadata_file = output_dir / "metadata.json"
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata: Dict[str, Any] = json.load(f)
        except:
            metadata: Dict[str, Any] = {
                "pipeline_version": "1.0",
                "started_at": datetime.now().isoformat()
            }
        
        # Always recalculate from actual working files to ensure accuracy
        actual_stats = self._calculate_actual_stats(output_dir)
        
        # Use actual stats instead of incremental counters
        processed_count = actual_stats["total_files"]
        successful_count = actual_stats["successful_count"] 
        failed_count = actual_stats["failed_count"]
        total_spans = actual_stats["total_spans"]
        
        # Calculate quality metrics
        avg_spans_per_sequence = total_spans / max(1, successful_count)
        success_rate = successful_count / max(1, processed_count)
        
        # Update counts and quality metrics
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["processed_sequences"] = processed_count
        metadata["successful_annotations"] = successful_count
        metadata["failed_annotations"] = failed_count
        metadata["total_spans"] = total_spans
        metadata["quality_metrics"] = {
            "avg_spans_per_sequence": round(avg_spans_per_sequence, 2),
            "success_rate": round(success_rate, 4),
            "annotation_density": round(total_spans / max(1, processed_count), 2),
            "pipeline_efficiency": "real_time_writing_enabled"
        }
        metadata["agent_stats"] = self.agent.get_statistics()
        
        # Save telemetry data to metadata as well (integrated approach)
        self.telemetry.save_telemetry_to_metadata(metadata_file)
        
        # Note: telemetry save will overwrite the file, so we need to merge manually
        # Let's reload and ensure our data is preserved
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                updated_metadata = json.load(f)
            
            # Merge our metadata with telemetry data
            for key, value in metadata.items():
                if key != "telemetry":  # Don't overwrite telemetry section
                    updated_metadata[key] = value
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(updated_metadata, f, indent=2)
                
        except Exception as e:
            # Fallback to original approach if telemetry merge fails
            logger.warning(f"Failed to merge telemetry data, using fallback: {e}")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
    
    def _calculate_actual_stats(self, output_dir: Path) -> Dict[str, int]:
        """
        Calculate actual statistics from working files to ensure metadata accuracy.
        
        Args:
            output_dir: Directory containing working files
            
        Returns:
            Dictionary with actual counts from working files
        """
        working_dir = output_dir / "working"
        
        if not working_dir.exists():
            return {
                "total_files": 0,
                "successful_count": 0,
                "failed_count": 0,
                "total_spans": 0
            }
        
        total_files = 0
        successful_count = 0
        failed_count = 0
        total_spans = 0
        
        for working_file in working_dir.glob("corpus-seq-*.json"):
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                total_files += 1
                annotation_session = data.get("annotation_session", {})
                status = annotation_session.get("annotation_status", "unknown")
                spans = annotation_session.get("spans_extracted", 0)
                
                if status == "completed" and spans > 0:
                    successful_count += 1
                    total_spans += spans
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to read working file {working_file}: {e}")
                total_files += 1
                failed_count += 1
        
        return {
            "total_files": total_files,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_spans": total_spans
        }

    def fix_existing_metadata(self, output_dir: Path):
        """
        Fix existing metadata.json by recalculating from working files.
        This can be called to repair incorrect metadata without reprocessing.
        """
        logger.info("Fixing metadata.json by recalculating from working files...")
        
        # This will automatically recalculate from working files
        self.update_global_metadata(output_dir, 0, 0, 0, 0)
        
        # Verify the fix
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"Metadata fixed:")
            logger.info(f"  - Processed sequences: {metadata.get('processed_sequences', 0)}")
            logger.info(f"  - Successful annotations: {metadata.get('successful_annotations', 0)}")
            logger.info(f"  - Failed annotations: {metadata.get('failed_annotations', 0)}")
            logger.info(f"  - Total spans: {metadata.get('total_spans', 0)}")
            logger.info(f"  - Success rate: {metadata.get('quality_metrics', {}).get('success_rate', 0):.1%}")
        else:
            logger.error("Failed to fix metadata - file not found")
    
    async def process_sequence_range(
        self,
        corpus_file: Path,
        output_dir: Path,
        range_spec: Optional[str] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process a specific range of sequences with comprehensive annotation.
        
        Args:
            corpus_file: Path to corpus.jsonl file
            output_dir: Output directory for results
            range_spec: Range specification (e.g., "1-100", "1,5,10") or None for all sequences
            resume: Whether to resume from existing results
            
        Returns:
            Processing statistics
        """
        self.stats["started_at"] = datetime.now().isoformat()
        
        # Parse target sequences - if no range specified, process all sequences
        if range_spec:
            sequence_ids = self.parse_range_specification(range_spec)
            target_sequences = self.load_target_sequences(corpus_file, sequence_ids)
            logger.info(f"Processing specified range: {range_spec}")
        else:
            target_sequences = self.load_all_sequences(corpus_file)
            logger.info(f"Processing all sequences in corpus")
        
        self.stats["total_sequences"] = len(target_sequences)
        
        # Ensure output structure
        self.ensure_output_structure(output_dir)
        
        # Load existing results for resume
        existing_results = self.load_existing_results(output_dir) if resume else {}
        
        # Fix metadata on resume to ensure accuracy from the start
        # Initialize telemetry once at the beginning
        if resume and existing_results:
            logger.info("Resume mode detected - correcting metadata from working files...")
            actual_stats = self.annotation_processor.fix_metadata_from_working_files(output_dir, self.agent.get_statistics())
            
            # Try to load telemetry state from metadata.json
            metadata_file = output_dir / "metadata.json"
            telemetry_loaded = self.telemetry.load_telemetry_from_metadata(metadata_file)
            
            if telemetry_loaded:
                # Update telemetry with current target sequences (might be different range)
                self.telemetry.telemetry["total_sequences"] = len(target_sequences)
                logger.info(f"Telemetry loaded successfully - updated target sequences to {len(target_sequences)}")
            else:
                # If telemetry loading failed, initialize with corrected metadata for accurate progress tracking
                self.telemetry.initialize(
                    total_sequences=len(target_sequences),
                    existing_completed=actual_stats["successful_count"],
                    existing_failed=actual_stats["failed_count"]
                )
                logger.info("Telemetry initialized from corrected metadata")
            
            # Log metadata correction
            logger.info("Metadata corrected from working files:")
            logger.info(f"  - Processed sequences: {actual_stats['total_files']}")
            logger.info(f"  - Successful annotations: {actual_stats['successful_count']}")
            logger.info(f"  - Failed annotations: {actual_stats['failed_count']}")
            logger.info(f"  - Total spans: {actual_stats['total_spans']}")
            logger.info(f"  - Success rate: {actual_stats['successful_count']/max(1, actual_stats['total_files'])*100:.1f}%")
        else:
            # Initialize telemetry for fresh run
            self.telemetry.initialize(total_sequences=len(target_sequences))
        
        # Display initial telemetry panel
        self.display_telemetry_if_needed(force=True, output_dir=output_dir)
        
        # Find gap sequences - those missing within the processed range
        gap_sequences = self.find_missing_sequences(target_sequences, existing_results)
        
        # Categorize sequences for processing
        sequences_to_process = []
        failed_sequences_to_retry = []
        gap_sequences_to_process = []
        completed_count = 0
        new_sequences_count = 0
        
        # Get highest processed sequence ID for categorization
        max_processed_id = max(existing_results.keys()) if existing_results else 0
        
        for record in target_sequences:
            if record.sequence_number in existing_results:
                status = existing_results[record.sequence_number]
                if status == "completed":
                    self.stats["processed_sequences"] += 1
                    self.stats["successful_annotations"] += 1
                    completed_count += 1
                    continue
                elif status == "failed":
                    failed_sequences_to_retry.append(record)
                    continue
            elif record.sequence_number in gap_sequences:
                # This is a gap within the processed range
                gap_sequences_to_process.append(record)
                continue
            elif record.sequence_number is not None and record.sequence_number > max_processed_id:
                # This is a new sequence beyond the processed range
                sequences_to_process.append(record)
                new_sequences_count += 1
                continue
            
            # Should not reach here, but add to new sequences as fallback
            sequences_to_process.append(record)
            new_sequences_count += 1
        
        # Process in priority order: failed sequences first, then gaps, then new sequences
        # This ensures we fill gaps in completed sequences before continuing with new ones
        all_sequences_to_process = failed_sequences_to_retry + gap_sequences_to_process + sequences_to_process
        
        # Improved logging with proper categorization
        if existing_results:
            logger.info(f"Resume mode: Found {len(existing_results)} existing working files (highest ID: {max_processed_id})")
        else:
            logger.info(f"Fresh run: No existing working files found")
        
        logger.info(f"Processing {len(all_sequences_to_process)} sequences:")
        logger.info(f"  - Retrying {len(failed_sequences_to_retry)} failed sequences")
        logger.info(f"  - Processing {len(gap_sequences_to_process)} gap sequences (within processed range)")
        logger.info(f"  - Processing {new_sequences_count} new sequences (beyond processed range)")
        logger.info(f"  - Skipped {completed_count} completed sequences")
        
        # Process sequences one at a time (batch_size = 1)
        batch_size = 1  # Force single sequence processing for reliability
        successful_count = self.stats["successful_annotations"]  # Include already completed
        failed_count = 0
        total_spans = 0
        
        # Track sequence start times for telemetry
        sequence_start_times = {}
        
        for i in range(0, len(all_sequences_to_process), batch_size):
            batch = all_sequences_to_process[i:i + batch_size]
            current_sequence = batch[0]  # Since batch_size = 1, this is the single sequence
            
            # Record start time for this sequence
            sequence_start_time = datetime.now()
            sequence_start_times[current_sequence.sequence_number] = sequence_start_time
            
            logger.info(f"Processing sequence {current_sequence.sequence_number} ({i + 1}/{len(all_sequences_to_process)})")
            
            try:
                # Create progress callback to save working files incrementally
                async def progress_callback(progress_info):
                    """Save intermediate progress to working files."""
                    sequence_id = progress_info.get("sequence_id")
                    phase = progress_info.get("phase", "")
                    
                    # Find the record for this sequence
                    for record in batch:
                        if record.sequence_number == sequence_id:
                            if phase == "completed":
                                # Sequence completed - save final result and write to annotations.jsonl
                                annotation_record = progress_info.get("annotation_record")
                                if annotation_record and hasattr(annotation_record, 'span_annotations') and len(annotation_record.span_annotations) > 0:
                                    # Only process as truly completed if spans were extracted
                                    self.save_working_file(output_dir, record, annotation_record)
                                    # Update telemetry
                                    sequence_start_time = sequence_start_times.get(sequence_id, datetime.now())
                                    self.telemetry.update_on_completion(annotation_record, sequence_start_time)
                                    
                                    # Update metadata immediately after each sequence
                                    self.update_global_metadata(
                                        output_dir,
                                        self.stats["processed_sequences"] + 1,  
                                        self.stats["successful_annotations"] + 1,
                                        self.stats["failed_annotations"], 
                                        self.stats["total_spans"] + len(annotation_record.span_annotations)
                                    )
                                    
                                    # Display telemetry panel conditionally (time-based or sequence-count-based)
                                    self.display_telemetry_if_needed(output_dir=output_dir)
                                    logger.info(f"[COMPLETED] Sequence {sequence_id} - {len(annotation_record.span_annotations)} spans extracted")
                                else:
                                    # Sequence marked as completed but no spans extracted - treat as failed
                                    self.save_working_file(output_dir, record, error_message="No spans extracted despite completion", status="failed")
                                    # Update telemetry for failure
                                    sequence_start_time = sequence_start_times.get(sequence_id, datetime.now())
                                    self.telemetry.update_on_failure(sequence_start_time)
                                    logger.warning(f"[FAILED] Sequence {sequence_id} - marked completed but no spans extracted")
                            else:
                                # Save working file with current progress
                                self.save_working_file(
                                    output_dir, 
                                    record, 
                                    status=f"in_progress_turn_{progress_info.get('turn', 0)}"
                                )
                            break
                
                # Process the sequence
                sequence_id = batch[0].sequence_number  # Since batch_size = 1
                logger.info(f"Starting annotation for sequence {sequence_id}...")
                
                # Process the batch (single sequence)
                annotation_batch = await self.agent.annotate_batch(batch, progress_callback=progress_callback)
                
                logger.info(f"Completed annotation for sequence {sequence_id}")
                
                # Save final results and track consecutive failures
                batch_successful = 0
                batch_failed = 0
                
                # Check results from the batch and update statistics
                # Note: Progress callback already handled saving and logging
                for j, record in enumerate(batch):
                    if j < len(annotation_batch.records):
                        annotation_result = annotation_batch.records[j]
                        # Find matching annotation result by sequence number
                        matching_result = None
                        for result in annotation_batch.records:
                            # Check if this result matches the current record
                            if (hasattr(result, 'sequence_id') and 
                                isinstance(result.sequence_id, str) and 
                                result.sequence_id.endswith(f"{record.sequence_number:08d}")):
                                matching_result = result
                                break
                            elif (hasattr(result, 'sequence_id') and 
                                  result.sequence_id == record.sequence_number):
                                matching_result = result
                                break
                        
                        # Default to direct indexing if no match found
                        if not matching_result:
                            matching_result = annotation_batch.records[j]
                        
                        # Check if sequence was successful (progress callback already saved the result)
                        if matching_result and hasattr(matching_result, 'span_annotations') and len(matching_result.span_annotations) > 0:
                            # Success - progress callback already handled saving and logging
                            successful_count += 1
                            batch_successful += 1
                            total_spans += len(matching_result.span_annotations)
                            # Reset consecutive failures on success
                            self.stats["consecutive_failures"] = 0
                        else:
                            # Failure - progress callback already handled saving and logging
                            failed_count += 1
                            batch_failed += 1
                            self.stats["consecutive_failures"] += 1
                    else:
                        # Batch processing incomplete - this shouldn't happen with progress callback
                        self.save_working_file(output_dir, record, error_message="Batch processing incomplete", status="failed")
                        failed_count += 1
                        batch_failed += 1
                        self.stats["consecutive_failures"] += 1
                        # Update telemetry for failure
                        sequence_start_time = sequence_start_times.get(record.sequence_number)
                        self.telemetry.update_on_failure(sequence_start_time)
                        logger.warning(f"Sequence {record.sequence_number} failed: batch processing incomplete (consecutive failures: {self.stats['consecutive_failures']})")
                
                # Check for consecutive failure exit condition
                if self.stats["consecutive_failures"] >= self.stats["max_consecutive_failures"]:
                    logger.error(f"Exiting due to {self.stats['consecutive_failures']} consecutive failures")
                    break
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Save failed results and track consecutive failures
                for record in batch:
                    self.save_working_file(output_dir, record, error_message=str(e), status="failed")
                    failed_count += 1
                    self.stats["consecutive_failures"] += 1
                    # Update telemetry for failure
                    sequence_start_time = sequence_start_times.get(record.sequence_number)
                    self.telemetry.update_on_failure(sequence_start_time)
                    logger.warning(f"Sequence {record.sequence_number} failed with exception: {e} (consecutive failures: {self.stats['consecutive_failures']})")
                
                # Check for consecutive failure exit condition
                if self.stats["consecutive_failures"] >= self.stats["max_consecutive_failures"]:
                    logger.error(f"Exiting due to {self.stats['consecutive_failures']} consecutive failures")
                    break
            
            # Update progress after each sequence
            self.stats["processed_sequences"] += len(batch)  # len(batch) = 1
            self.stats["successful_annotations"] = successful_count
            self.stats["failed_annotations"] = failed_count
            self.stats["total_spans"] = total_spans
            
            # Update global metadata after each sequence
            self.update_global_metadata(
                output_dir, 
                self.stats["processed_sequences"],
                successful_count,
                failed_count,
                total_spans
            )
            
            logger.info(f"Sequence {current_sequence.sequence_number} completed: {batch_successful} successful, {batch_failed} failed")
        
        self.stats["completed_at"] = datetime.now().isoformat()
        
        # Display final telemetry panel
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETION SUMMARY")
        logger.info("=" * 60)
        self.display_telemetry_if_needed(force=True, output_dir=output_dir)
        
        return self.stats
    
    def consolidate_results(self, output_dir: Path):
        """
        Legacy consolidation method - now annotations are written in real-time with deduplication.
        This method reports the final count.
        """
        annotations_file = output_dir / "annotations.jsonl"
        
        if annotations_file.exists():
            # Count lines in the file
            with open(annotations_file, 'r', encoding='utf-8') as f:
                span_count = sum(1 for line in f if line.strip())
            
            logger.info(f"Real-time annotations completed: {span_count} span records in {annotations_file}")
        else:
            logger.info("No annotations file found - no spans were processed")
        
        logger.info("Annotations written in compact JSONL format with automatic deduplication")
    
    def deduplicate_annotations_file(self, output_dir: Path) -> Dict[str, int]:
        """
        Post-process existing annotations.jsonl to remove duplicates.
        
        Returns:
            Dictionary with deduplication statistics
        """
        annotations_file = output_dir / "annotations.jsonl"
        backup_file = output_dir / "annotations_backup.jsonl"
        
        if not annotations_file.exists():
            logger.warning(f"No annotations file found at {annotations_file}")
            return {"original_count": 0, "deduplicated_count": 0, "removed_count": 0}
        
        # Read all annotations
        annotations = []
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line.strip()))
        
        original_count = len(annotations)
        
        # Group by sequence_id and deduplicate within each sequence
        by_sequence = {}
        for ann in annotations:
            seq_id = ann['sequence_id']
            if seq_id not in by_sequence:
                by_sequence[seq_id] = []
            by_sequence[seq_id].append(ann)
        
        deduplicated_annotations = []
        total_removed = 0
        
        for seq_id, seq_annotations in by_sequence.items():
            # Deduplicate by position within sequence
            unique_positions = {}
            seq_removed = 0
            
            for ann in seq_annotations:
                pos_key = (ann['span_annotation']['start_pos'], ann['span_annotation']['end_pos'])
                
                if pos_key not in unique_positions:
                    unique_positions[pos_key] = ann
                else:
                    seq_removed += 1
                    # Keep annotation with higher confidence if both have confidence
                    existing = unique_positions[pos_key]
                    current = ann
                    
                    # Simple heuristic: prefer longer xbar_class names (more specific)
                    if len(current['span_annotation']['xbar_class']) > len(existing['span_annotation']['xbar_class']):
                        unique_positions[pos_key] = current
            
            deduplicated_annotations.extend(unique_positions.values())
            total_removed += seq_removed
            
            if seq_removed > 0:
                logger.info(f"Sequence {seq_id}: removed {seq_removed} duplicates, kept {len(unique_positions)} unique spans")
        
        deduplicated_count = len(deduplicated_annotations)
        
        if total_removed > 0:
            # Create backup of original file
            import shutil
            shutil.copy2(annotations_file, backup_file)
            logger.info(f"Created backup at {backup_file}")
            
            # Write deduplicated annotations
            with open(annotations_file, 'w', encoding='utf-8') as f:
                for ann in deduplicated_annotations:
                    f.write(json.dumps(ann, ensure_ascii=False) + '\n')
            
            logger.info(f"Deduplication complete: {original_count} -> {deduplicated_count} spans ({total_removed} removed)")
        else:
            logger.info(f"No duplicates found in {original_count} spans")
        
        return {
            "original_count": original_count,
            "deduplicated_count": deduplicated_count,
            "removed_count": total_removed
        }


def main():
    """Main entry point for span annotator pipeline."""
    parser = argparse.ArgumentParser(description="X-Spanformer Span Annotator Pipeline")
    
    parser.add_argument(
        "--corpus", 
        type=Path,
        required=True,
        help="Path to corpus.jsonl file from jsonl2vocab pipeline"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path, 
        required=True,
        help="Output directory for annotation results"
    )
    
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Range specification (e.g., '1-100', '1,5,10', '42'). If not specified, processes all sequences in corpus."
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/pipelines/span_annotator.yaml)"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Path to agent configuration file (default: config/agents/span_annotator_agent.yaml)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline and load configuration
    pipeline = SpanAnnotatorPipeline(args.config, args.agent)
    
    # Setup logging using shared logging utilities
    from x_spanformer.pipelines.shared.pipeline_logging import PipelineLogger
    
    # Configure logging from pipeline config
    log_file_path = None
    if pipeline.config.logging.log_to_file:
        args.output.mkdir(parents=True, exist_ok=True)
        log_file_path = args.output / "span_annotator.log"
    
    logger = PipelineLogger.setup_pipeline_logging(
        pipeline_name="Span Annotation Pipeline",
        log_level=pipeline.config.logging.level.upper(),
        log_format=pipeline.config.logging.format,
        log_to_file=pipeline.config.logging.log_to_file,
        log_file_path=log_file_path,
        console_output=True
    )
    
    # Configure the module-level logger to use the same handlers and settings
    module_logger = logging.getLogger(__name__)
    module_logger.handlers.clear()
    for handler in logger.handlers:
        module_logger.addHandler(handler)
    module_logger.setLevel(logger.level)
    module_logger.propagate = False  # Prevent duplicate messages
    
    # Configure the session logger to use the same handlers and settings
    PipelineLogger.configure_module_logger(logger, "x_spanformer.agents.session.span_annotator_session")
    
    # Also configure other related loggers
    PipelineLogger.configure_all_module_loggers(logger, [
        "x_spanformer.agents.dialogue",
        "x_spanformer.agents.ollama_client",
        "x_spanformer.pipelines.shared.annotation_processor"
    ])
    
    # Configure the session logger to use the same handlers and settings
    session_logger = PipelineLogger.configure_module_logger(logger, "x_spanformer.agents.session.span_annotator_session")
    
    # Log pipeline startup information
    PipelineLogger.log_pipeline_start(
        logger,
        "X-Spanformer Span Annotator Pipeline",
        corpus=str(args.corpus),
        output=str(args.output),
        range=args.range if args.range else 'ALL SEQUENCES',
        config_path=args.config if args.config else 'default (config/pipelines/span_annotator.yaml)',
        agent_config_path=args.agent if args.agent else 'default (config/agents/span_annotator_agent.yaml)',
        batch_size=pipeline.config.processing.batch_size,
        model=pipeline.agent.model_name,
        max_turns=pipeline.agent.max_turns
    )
    logger.info(f"Loaded temperature: {pipeline.agent.temperature}")
    logger.info(f"Loaded timeout: {pipeline.config.processing.conversation_timeout}")
    logger.info(f"Loaded max_retries: {pipeline.config.processing.max_retries}")
    
    # Check Ollama connection with retry logic before starting processing
    async def check_ollama_with_retry():
        logger.info("Testing Ollama connection...")
        model_name = "phi4-mini"  # Default model - should match agent config
        max_retries = pipeline.config.processing.max_retries
        retry_delay = 3  # seconds between retries
        connection_timeout = 10  # seconds timeout per attempt
        
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging
                connection_result = await asyncio.wait_for(
                    check_ollama_connection(model_name), 
                    timeout=connection_timeout
                )
                
                if connection_result:
                    logger.info(f"[SUCCESS] Ollama connection successful (model: {model_name})")
                    return True
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"[FAILED] Ollama connection failed (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"[FAILED] Ollama connection failed after {max_retries} attempts!")
                        
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"⏱️ Ollama connection timeout (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"⏱️ Ollama connection timeout after {max_retries} attempts!")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[ERROR] Error testing Ollama connection (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"[ERROR] Error testing Ollama connection after {max_retries} attempts: {str(e)}")
        
        return False

    # Run annotation pipeline
    async def run_pipeline():
        # Check Ollama connection first
        if not await check_ollama_with_retry():
            logger.error("[FATAL] Cannot connect to Ollama after maximum retry attempts!")
            logger.error("[INFO] Please ensure Ollama is running: ollama serve")
            logger.error(f"[INFO] Please ensure model is available: ollama run phi4-mini")
            logger.error("[INFO] Check Ollama is accessible and the model is loaded")
            sys.exit(1)
        
        stats = await pipeline.process_sequence_range(
            corpus_file=args.corpus,
            output_dir=args.output,
            range_spec=args.range,  # Can be None now
            resume=True  # Always resume based on existing working files
        )
        
        logger.info("Pipeline completed!")
        logger.info(f"Statistics: {stats}")
        
        # Consolidate results
        pipeline.consolidate_results(args.output)
    
    # Run async pipeline
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
