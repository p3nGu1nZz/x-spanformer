"""
Shared telemetry tracking for X-Spanformer pipelines.

Provides centralized telemetry functionality for tracking pipeline progress,
performance metrics, ETA calculations, and status reporting across different
pipeline types.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineTelemetry:
    """
    Centralized telemetry tracking for X-Spanformer pipelines.
    
    Tracks progress, timing, success rates, and provides formatted reporting
    for any pipeline that processes sequences in batches.
    """
    
    def __init__(self, pipeline_name: str = "Pipeline"):
        """
        Initialize telemetry tracking.
        
        Args:
            pipeline_name: Name of the pipeline for display purposes
        """
        self.pipeline_name = pipeline_name
        self.telemetry = {
            "start_time": None,
            "completed_sequences": 0,
            "failed_sequences": 0,
            "total_sequences": 0,
            "spans_by_type": {},
            "spans_by_modality": {},
            "sequence_times": [],
            "last_sequence_time": None
        }
    
    def initialize(self, total_sequences: int, existing_completed: int = 0, existing_failed: int = 0):
        """
        Initialize telemetry with starting values.
        
        Args:
            total_sequences: Total number of sequences to process
            existing_completed: Number of sequences already completed (for resume)
            existing_failed: Number of sequences that previously failed (for resume)
        """
        self.telemetry["start_time"] = datetime.now()
        self.telemetry["total_sequences"] = total_sequences
        self.telemetry["completed_sequences"] = existing_completed
        self.telemetry["failed_sequences"] = existing_failed
        self.telemetry["spans_by_type"] = {}
        self.telemetry["spans_by_modality"] = {}
        self.telemetry["sequence_times"] = []
        
        logger.info(f"{self.pipeline_name} telemetry initialized:")
        logger.info(f"  - Total sequences: {total_sequences}")
        logger.info(f"  - Already completed: {existing_completed}")
        logger.info(f"  - Previously failed: {existing_failed}")
    
    def update_on_completion(self, annotation_result=None, sequence_start_time: Optional[datetime] = None):
        """
        Update telemetry when a sequence is completed successfully.
        
        Args:
            annotation_result: Result object with span annotations (optional)
            sequence_start_time: When processing started for this sequence
        """
        current_time = datetime.now()
        
        # Update completion count
        self.telemetry["completed_sequences"] += 1
        
        # Track sequence processing time
        if sequence_start_time:
            sequence_time = (current_time - sequence_start_time).total_seconds()
            self.telemetry["sequence_times"].append(sequence_time)
            self.telemetry["last_sequence_time"] = sequence_time
        
        # Track span statistics if annotation result provided
        if annotation_result and hasattr(annotation_result, 'span_annotations') and annotation_result.span_annotations:
            for span in annotation_result.span_annotations:
                # Track by xbar_class (type)
                span_type = span.xbar_class if hasattr(span, 'xbar_class') else 'unknown'
                self.telemetry["spans_by_type"][span_type] = self.telemetry["spans_by_type"].get(span_type, 0) + 1
                
                # Track by modality (inferred from span properties)
                modality = self._infer_span_modality(span)
                self.telemetry["spans_by_modality"][modality] = self.telemetry["spans_by_modality"].get(modality, 0) + 1
    
    def update_on_failure(self, sequence_start_time: Optional[datetime] = None):
        """
        Update telemetry when a sequence fails.
        
        Args:
            sequence_start_time: When processing started for this sequence
        """
        current_time = datetime.now()
        
        # Update failure count
        self.telemetry["failed_sequences"] += 1
        
        # Track sequence processing time even for failures
        if sequence_start_time:
            sequence_time = (current_time - sequence_start_time).total_seconds()
            self.telemetry["sequence_times"].append(sequence_time)
            self.telemetry["last_sequence_time"] = sequence_time
    
    def display_progress_panel(self):
        """Display comprehensive telemetry panel with progress and statistics."""
        current_time = datetime.now()
        
        # Calculate progress metrics
        processed_sequences = self.telemetry["completed_sequences"] + self.telemetry["failed_sequences"]
        progress_pct = (self.telemetry["completed_sequences"] / max(self.telemetry["total_sequences"], 1)) * 100
        success_rate = (self.telemetry["completed_sequences"] / max(processed_sequences, 1)) * 100
        
        # Calculate timing metrics
        elapsed_time = 0
        sequences_per_min = 0
        eta_display = "calculating..."
        
        if self.telemetry["start_time"]:
            elapsed_seconds = (current_time - self.telemetry["start_time"]).total_seconds()
            elapsed_time = elapsed_seconds / 60  # Convert to minutes
            
            if elapsed_seconds > 0 and processed_sequences > 0:
                sequences_per_min = (processed_sequences * 60) / elapsed_seconds
                
                # Calculate remaining work: 
                # - Unprocessed sequences from total corpus
                # - Failed sequences that need retry
                remaining_new_sequences = self.telemetry["total_sequences"] - self.telemetry["completed_sequences"]
                remaining_work = remaining_new_sequences + self.telemetry["failed_sequences"]
                
                if sequences_per_min > 0 and remaining_work > 0:
                    eta_minutes = remaining_work / sequences_per_min
                    eta_display = self._format_eta(eta_minutes)
                elif remaining_work == 0:
                    eta_display = "Complete!"
        
        # Calculate span statistics
        total_spans = sum(self.telemetry["spans_by_type"].values())
        span_types_summary = self._format_span_summary(self.telemetry["spans_by_type"])
        modality_summary = self._format_span_summary(self.telemetry["spans_by_modality"])
        
        # Calculate average sequence processing time
        avg_seq_time = 0
        if self.telemetry["sequence_times"]:
            avg_seq_time = sum(self.telemetry["sequence_times"]) / len(self.telemetry["sequence_times"])
        
        # Calculate remaining work breakdown for display
        remaining_new_sequences = self.telemetry["total_sequences"] - self.telemetry["completed_sequences"] 
        total_remaining_work = remaining_new_sequences + self.telemetry["failed_sequences"]
        
        # Display telemetry panel
        logger.info("=" * 80)
        logger.info(f"TELEMETRY PANEL - {self.pipeline_name} Progress")
        logger.info("=" * 80)
        logger.info(f"Progress: {processed_sequences}/{self.telemetry['total_sequences']} sequences ({progress_pct:.1f}%)")
        logger.info(f"Success Rate: {self.telemetry['completed_sequences']}/{processed_sequences} successful ({success_rate:.1f}%)")
        logger.info(f"Failed Sequences: {self.telemetry['failed_sequences']} (need retry)")
        logger.info(f"Remaining Work: {total_remaining_work} sequences ({remaining_new_sequences} new + {self.telemetry['failed_sequences']} retries)")
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current telemetry statistics.
        
        Returns:
            Dictionary of telemetry statistics
        """
        processed_sequences = self.telemetry["completed_sequences"] + self.telemetry["failed_sequences"]
        success_rate = (self.telemetry["completed_sequences"] / max(processed_sequences, 1)) * 100
        
        # Calculate timing metrics
        elapsed_time = 0
        sequences_per_min = 0
        
        if self.telemetry["start_time"]:
            elapsed_seconds = (datetime.now() - self.telemetry["start_time"]).total_seconds()
            elapsed_time = elapsed_seconds / 60
            
            if elapsed_seconds > 0 and processed_sequences > 0:
                sequences_per_min = (processed_sequences * 60) / elapsed_seconds
        
        # Calculate average sequence processing time
        avg_seq_time = 0
        if self.telemetry["sequence_times"]:
            avg_seq_time = sum(self.telemetry["sequence_times"]) / len(self.telemetry["sequence_times"])
        
        return {
            "total_sequences": self.telemetry["total_sequences"],
            "completed_sequences": self.telemetry["completed_sequences"],
            "failed_sequences": self.telemetry["failed_sequences"],
            "processed_sequences": processed_sequences,
            "success_rate_percent": success_rate,
            "elapsed_time_minutes": elapsed_time,
            "processing_rate_per_min": sequences_per_min,
            "average_sequence_time_seconds": avg_seq_time,
            "total_spans": sum(self.telemetry["spans_by_type"].values()),
            "spans_by_type": dict(self.telemetry["spans_by_type"]),
            "spans_by_modality": dict(self.telemetry["spans_by_modality"])
        }
    
    def _format_eta(self, eta_minutes: float) -> str:
        """Format ETA for display."""
        if eta_minutes >= 60:
            eta_hours = int(eta_minutes // 60)
            eta_mins = int(eta_minutes % 60)
            return f"{eta_hours}h {eta_mins}m"
        else:
            return f"{eta_minutes:.1f} minutes"
    
    def _format_span_summary(self, span_dict: Dict[str, int], max_items: int = 20) -> str:
        """Format span statistics for display."""
        if not span_dict:
            return ""
        
        # Sort by count descending
        sorted_items = sorted(span_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Limit display items
        if len(sorted_items) > max_items:
            displayed = sorted_items[:max_items]
            remaining = len(sorted_items) - max_items
            summary_parts = [f"{name}: {count}" for name, count in displayed]
            summary_parts.append(f"... and {remaining} more")
        else:
            summary_parts = [f"{name}: {count}" for name, count in sorted_items]
        
        return ", ".join(summary_parts)
    
    def _infer_span_modality(self, span) -> str:
        """Infer the modality of a span based on its properties."""
        # Handle None or missing xbar_class attribute
        if not hasattr(span, 'xbar_class') or span.xbar_class is None:
            return 'other'
        
        span_type = str(span.xbar_class).lower()
        
        # Basic modality classification based on X-bar class
        if any(keyword in span_type for keyword in ['punct', 'symbol', 'operator', 'delim']):
            return 'punctuation'
        elif any(keyword in span_type for keyword in ['noun', 'verb', 'adj', 'adv', 'det', 'prep']):
            return 'lexical'
        elif any(keyword in span_type for keyword in ['phrase', 'clause', 'np', 'vp', 'pp', 'adjp', 'advp']):
            return 'syntactic'
        elif any(keyword in span_type for keyword in ['sentence', 'block', 'root']):
            return 'structural'
        else:
            return 'other'


class SpanAnnotationTelemetry(PipelineTelemetry):
    """
    Specialized telemetry for span annotation pipelines.
    
    Extends base telemetry with span-specific metrics and reporting.
    """
    
    def __init__(self):
        super().__init__("Span Annotation")
    
    def update_on_completion(self, annotation_result, sequence_start_time: Optional[datetime] = None):
        """Update telemetry with span annotation specific details."""
        # Call parent method for basic tracking
        super().update_on_completion(annotation_result, sequence_start_time)
        
        # Additional span annotation specific logging
        if annotation_result and hasattr(annotation_result, 'span_annotations'):
            span_count = len(annotation_result.span_annotations)
            sequence_id = getattr(annotation_result, 'sequence_id', 'unknown')
            logger.debug(f"Sequence {sequence_id}: extracted {span_count} spans")
