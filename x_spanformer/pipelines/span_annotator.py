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
        if agent_config_path:
            # For now, we'll pass individual parameters to the agent
            # TODO: Create agent config loader if needed
            agent_config_file = Path(agent_config_path)
            if agent_config_file.exists():
                import yaml
                with open(agent_config_file, 'r', encoding='utf-8') as f:
                    agent_config = yaml.safe_load(f)
                
                model_config = agent_config.get("model", {})
                model_name = model_config.get("name", "phi4-mini")
                temperature = model_config.get("temperature", 0.1)
                
                # Extract dialogue configuration
                dialogue_config = agent_config.get("dialogue", {})
                max_turns = dialogue_config.get("max_turns", 16)
                
                # Extract early termination config
                early_termination_config = agent_config.get("agent", {}).get("early_termination")
            else:
                model_name = "phi4-mini"
                temperature = 0.1
                max_turns = 16
                early_termination_config = None
        else:
            model_name = "phi4-mini"
            temperature = 0.1
            max_turns = 16
            early_termination_config = None
        
        # Initialize span annotator session
        self.agent = SpanAnnotatorSession(
            model_name=model_name,
            max_concurrent=1,  # Process one at a time
            max_retries=self.config.processing.max_retries,
            conversation_timeout=self.config.processing.conversation_timeout,
            max_turns=max_turns,
            early_termination_config=early_termination_config
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
            "completed_at": None
        }
        
        # Telemetry tracking
        self.telemetry = {
            "start_time": None,
            "completed_sequences": 0,
            "total_sequences": 0,
            "spans_by_type": {},
            "spans_by_modality": {},
            "sequence_times": [],
            "last_sequence_time": None
        }
    
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
            return existing_results
        
        for working_file in working_dir.glob("corpus-seq-*.json"):
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sequence_number = data.get("sequence_number")
                annotation_status = data.get("annotation_session", {}).get("annotation_status", "unknown")
                
                if sequence_number is not None:
                    existing_results[sequence_number] = annotation_status
                    
            except Exception as e:
                logger.warning(f"Failed to load existing result from {working_file}: {e}")
        
        return existing_results
    
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
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
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
        
        # Initialize telemetry
        self.telemetry["start_time"] = datetime.now()
        self.telemetry["total_sequences"] = 0
        self.telemetry["completed_sequences"] = 0
        self.telemetry["spans_by_type"] = {}
        self.telemetry["spans_by_modality"] = {}
        self.telemetry["sequence_times"] = []
        
        # Parse target sequences - if no range specified, process all sequences
        if range_spec:
            sequence_ids = self.parse_range_specification(range_spec)
            target_sequences = self.load_target_sequences(corpus_file, sequence_ids)
            logger.info(f"Processing specified range: {range_spec}")
        else:
            target_sequences = self.load_all_sequences(corpus_file)
            logger.info(f"Processing all sequences in corpus")
        
        self.stats["total_sequences"] = len(target_sequences)
        self.telemetry["total_sequences"] = len(target_sequences)
        
        # Ensure output structure
        self.ensure_output_structure(output_dir)
        
        # Load existing results for resume
        existing_results = self.load_existing_results(output_dir) if resume else {}
        
        # Filter sequences to process
        sequences_to_process = []
        for record in target_sequences:
            if record.sequence_number in existing_results:
                status = existing_results[record.sequence_number]
                if status == "completed":
                    logger.info(f"Skipping completed sequence {record.sequence_number}")
                    self.stats["processed_sequences"] += 1
                    continue
            
            sequences_to_process.append(record)
        
        logger.info(f"Processing {len(sequences_to_process)} sequences (skipped {len(target_sequences) - len(sequences_to_process)} completed)")
        
        # Process sequences in batches
        batch_size = self.config.processing.batch_size
        successful_count = 0
        failed_count = 0
        total_spans = 0
        
        # Track sequence start times for telemetry
        sequence_start_times = {}
        
        for i in range(0, len(sequences_to_process), batch_size):
            batch = sequences_to_process[i:i + batch_size]
            
            # Record start times for this batch
            batch_start_time = datetime.now()
            for record in batch:
                sequence_start_times[record.sequence_number] = batch_start_time
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(sequences_to_process) + batch_size - 1)//batch_size}")
            
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
                                if annotation_record:
                                    self.save_working_file(output_dir, record, annotation_record)
                                    # Update telemetry
                                    sequence_start_time = sequence_start_times.get(sequence_id, batch_start_time)
                                    self.update_telemetry_on_completion(annotation_record, sequence_start_time)
                                    # Display telemetry panel
                                    self.display_telemetry_panel()
                                    logger.info(f"[COMPLETED] Sequence {sequence_id} - wrote to annotations.jsonl")
                                else:
                                    self.save_working_file(output_dir, record, status="completed")
                            else:
                                # Save working file with current progress
                                self.save_working_file(
                                    output_dir, 
                                    record, 
                                    status=f"in_progress_turn_{progress_info.get('turn', 0)}"
                                )
                            break
                
                # Annotate batch with progress tracking
                annotation_batch = await self.agent.annotate_batch(batch, progress_callback=progress_callback)
                
                # Save final results
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
                        
                        # Skip annotations write since it was already done in progress callback
                        self.save_working_file(output_dir, record, matching_result, skip_annotations_write=True)
                        successful_count += 1
                        total_spans += len(matching_result.span_annotations)
                    else:
                        self.save_working_file(output_dir, record, error_message="Batch processing incomplete", status="failed")
                        failed_count += 1
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Save failed results
                for record in batch:
                    self.save_working_file(output_dir, record, error_message=str(e), status="failed")
                    failed_count += 1
            
            # Update progress
            self.stats["processed_sequences"] += len(batch)
            self.stats["successful_annotations"] = successful_count
            self.stats["failed_annotations"] = failed_count
            self.stats["total_spans"] = total_spans
            
            # Update global metadata
            self.update_global_metadata(
                output_dir, 
                self.stats["processed_sequences"],
                successful_count,
                failed_count,
                total_spans
            )
        
        self.stats["completed_at"] = datetime.now().isoformat()
        
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
    
    def display_telemetry_panel(self):
        """Display telemetry panel with progress and statistics in logger-compatible format."""
        current_time = datetime.now()
        
        # Calculate progress metrics
        progress_pct = (self.telemetry["completed_sequences"] / max(self.telemetry["total_sequences"], 1)) * 100
        
        # Calculate timing metrics
        elapsed_time = 0
        sequences_per_min = 0
        eta_minutes = 0
        eta_display = "calculating..."
        
        if self.telemetry["start_time"]:
            elapsed_seconds = (current_time - self.telemetry["start_time"]).total_seconds()
            elapsed_time = elapsed_seconds / 60  # Convert to minutes
            
            if elapsed_seconds > 0 and self.telemetry["completed_sequences"] > 0:
                sequences_per_min = (self.telemetry["completed_sequences"] * 60) / elapsed_seconds
                
                remaining_sequences = self.telemetry["total_sequences"] - self.telemetry["completed_sequences"]
                if sequences_per_min > 0:
                    eta_minutes = remaining_sequences / sequences_per_min
                    
                    # Format ETA display
                    if eta_minutes >= 60:
                        eta_hours = int(eta_minutes // 60)
                        eta_mins = int(eta_minutes % 60)
                        eta_display = f"{eta_hours}h {eta_mins}m"
                    else:
                        eta_display = f"{eta_minutes:.1f} minutes"
        
        # Calculate span statistics
        total_spans = sum(self.telemetry["spans_by_type"].values())
        span_types_summary = ", ".join([f"{type_name}: {count}" for type_name, count in sorted(self.telemetry["spans_by_type"].items())])
        modality_summary = ", ".join([f"{mod}: {count}" for mod, count in sorted(self.telemetry["spans_by_modality"].items())])
        
        # Calculate average sequence processing time
        avg_seq_time = 0
        if self.telemetry["sequence_times"]:
            avg_seq_time = sum(self.telemetry["sequence_times"]) / len(self.telemetry["sequence_times"])
        
        # Display telemetry panel
        logger.info("=" * 80)
        logger.info("TELEMETRY PANEL - Span Annotation Progress")
        logger.info("=" * 80)
        logger.info(f"Progress: {self.telemetry['completed_sequences']}/{self.telemetry['total_sequences']} sequences ({progress_pct:.1f}%)")
        logger.info(f"Processing Rate: {sequences_per_min:.2f} sequences/min")
        logger.info(f"Elapsed Time: {elapsed_time:.1f} minutes")
        logger.info(f"ETA: {eta_display}")
        logger.info(f"Average Sequence Time: {avg_seq_time:.1f} seconds")
        logger.info("-" * 40)
        logger.info(f"Total Spans Extracted: {total_spans}")
        if span_types_summary:
            logger.info(f"Span Types: {span_types_summary}")
        if modality_summary:
            logger.info(f"Modalities: {modality_summary}")
        logger.info("=" * 80)

    def update_telemetry_on_completion(self, annotation_result: AnnotationRecord, sequence_start_time: datetime):
        """Update telemetry data when a sequence is completed."""
        current_time = datetime.now()
        
        # Update completion count
        self.telemetry["completed_sequences"] += 1
        
        # Track sequence processing time
        if sequence_start_time:
            sequence_time = (current_time - sequence_start_time).total_seconds()
            self.telemetry["sequence_times"].append(sequence_time)
            self.telemetry["last_sequence_time"] = sequence_time
        
        # Track span statistics
        if hasattr(annotation_result, 'span_annotations') and annotation_result.span_annotations:
            for span in annotation_result.span_annotations:
                # Track by xbar_class (type)
                span_type = span.xbar_class if hasattr(span, 'xbar_class') else 'unknown'
                self.telemetry["spans_by_type"][span_type] = self.telemetry["spans_by_type"].get(span_type, 0) + 1
                
                # Track by modality (inferred from span properties)
                modality = self._infer_span_modality(span)
                self.telemetry["spans_by_modality"][modality] = self.telemetry["spans_by_modality"].get(modality, 0) + 1
    
    def _infer_span_modality(self, span) -> str:
        """Infer the modality of a span based on its properties."""
        span_type = span.xbar_class if hasattr(span, 'xbar_class') else ''
        
        # Basic modality classification based on X-bar class
        if any(keyword in span_type.lower() for keyword in ['punct', 'symbol', 'operator', 'delim']):
            return 'punctuation'
        elif any(keyword in span_type.lower() for keyword in ['noun', 'verb', 'adj', 'adv', 'det', 'prep']):
            return 'lexical'
        elif any(keyword in span_type.lower() for keyword in ['phrase', 'clause', "'"]):
            return 'syntactic'
        elif any(keyword in span_type.lower() for keyword in ['sentence', 'block', 'root']):
            return 'structural'
        else:
            return 'other'

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
    
    # Setup logging based on configuration
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if pipeline.config.logging.log_to_file:
        args.output.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(args.output / "span_annotator.log"))
    
    logging.basicConfig(
        level=getattr(logging, pipeline.config.logging.level.upper()),
        format=pipeline.config.logging.format,
        handlers=log_handlers
    )
    
    logger.info(f"Starting X-Spanformer Span Annotator Pipeline")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Range: {args.range if args.range else 'ALL SEQUENCES'}")
    
    # Initialize pipeline
    pipeline = SpanAnnotatorPipeline(args.config, args.agent)
    
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
