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
            else:
                model_name = "phi4-mini"
                temperature = 0.1
        else:
            model_name = "phi4-mini"
            temperature = 0.1
        
        # Initialize span annotator session
        self.agent = SpanAnnotatorSession(
            model_name=model_name,
            max_concurrent=1,  # Process one at a time
            max_retries=self.config.processing.max_retries,
            conversation_timeout=self.config.processing.conversation_timeout
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
        (output_dir / "consolidated").mkdir(exist_ok=True)
        
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
        annotation_result: Optional[AnnotationRecord],
        error_message: Optional[str] = None
    ):
        """Save annotation result to individual working file."""
        working_file = output_dir / "working" / f"corpus-seq-{record.sequence_number:08d}.json"
        
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
                "completed_at": datetime.now().isoformat(),
                "annotation_status": "completed" if annotation_result else "failed",
                "model": self.agent.model_name,
                "error_message": error_message
            }
        }
        
        if annotation_result:
            working_data["annotation_result"] = annotation_result.model_dump()
        
        with open(working_file, 'w', encoding='utf-8') as f:
            json.dump(working_data, f, indent=2, ensure_ascii=False)
    
    def update_global_metadata(
        self, 
        output_dir: Path, 
        processed_count: int, 
        successful_count: int,
        failed_count: int,
        total_spans: int
    ):
        """Update global metadata file with processing progress."""
        metadata_file = output_dir / "metadata.json"
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except:
            metadata = {}
        
        # Update counts
        metadata.update({
            "last_updated": datetime.now().isoformat(),
            "processed_sequences": processed_count,
            "successful_annotations": successful_count,
            "failed_annotations": failed_count,
            "total_spans": total_spans,
            "agent_stats": self.agent.get_statistics()
        })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    async def process_sequence_range(
        self,
        corpus_file: Path,
        output_dir: Path,
        range_spec: str,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process a specific range of sequences with comprehensive annotation.
        
        Args:
            corpus_file: Path to corpus.jsonl file
            output_dir: Output directory for results
            range_spec: Range specification (e.g., "1-100", "1,5,10")
            resume: Whether to resume from existing results
            
        Returns:
            Processing statistics
        """
        self.stats["started_at"] = datetime.now().isoformat()
        
        # Parse target sequences
        sequence_ids = self.parse_range_specification(range_spec)
        target_sequences = self.load_target_sequences(corpus_file, sequence_ids)
        
        self.stats["total_sequences"] = len(target_sequences)
        
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
        
        for i in range(0, len(sequences_to_process), batch_size):
            batch = sequences_to_process[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(sequences_to_process) + batch_size - 1)//batch_size}")
            
            try:
                # Annotate batch
                annotation_batch = await self.agent.annotate_batch(batch)
                
                # Save results
                for j, record in enumerate(batch):
                    if j < len(annotation_batch.records):
                        annotation_result = annotation_batch.records[j]
                        self.save_working_file(output_dir, record, annotation_result)
                        successful_count += 1
                        total_spans += len(annotation_result.span_annotations)
                    else:
                        self.save_working_file(output_dir, record, None, "Batch processing failed")
                        failed_count += 1
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Save failed results
                for record in batch:
                    self.save_working_file(output_dir, record, None, str(e))
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
        """Consolidate working files into final training data."""
        logger.info("Consolidating annotation results...")
        
        working_dir = output_dir / "working"
        consolidated_dir = output_dir / "consolidated"
        consolidated_dir.mkdir(exist_ok=True)
        
        annotations_file = consolidated_dir / "annotations.jsonl"
        
        successful_annotations = []
        
        for working_file in working_dir.glob("corpus-seq-*.json"):
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get("annotation_session", {}).get("annotation_status") == "completed":
                    annotation_result = data.get("annotation_result")
                    if annotation_result:
                        successful_annotations.append(annotation_result)
                        
            except Exception as e:
                logger.warning(f"Failed to load {working_file}: {e}")
        
        # Write consolidated annotations
        with open(annotations_file, 'w', encoding='utf-8') as f:
            for annotation in successful_annotations:
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        logger.info(f"Consolidated {len(successful_annotations)} annotations to {annotations_file}")


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
        "--output",
        type=Path, 
        required=True,
        help="Output directory for annotation results"
    )
    
    parser.add_argument(
        "--range",
        type=str,
        required=True,
        help="Range specification (e.g., '1-100', '1,5,10', '42')"
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
    
    parser.add_argument(
        "--consolidate-only",
        action="store_true", 
        help="Only consolidate existing results without processing"
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
    logger.info(f"Range: {args.range}")
    
    # Initialize pipeline
    pipeline = SpanAnnotatorPipeline(args.config, args.agent)
    
    if args.consolidate_only:
        # Only consolidate existing results
        pipeline.consolidate_results(args.output)
        return
    
    # Run annotation pipeline
    async def run_pipeline():
        stats = await pipeline.process_sequence_range(
            corpus_file=args.corpus,
            output_dir=args.output,
            range_spec=args.range,
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
