"""
Shared telemetry tracking for X-Spanformer pipelines.

Provides centralized telemetry functionality for tracking pipeline progress,
performance metrics, ETA calculations, and status reporting across different
pipeline types.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Use the same logger name as the main pipeline for consistent output
logger = logging.getLogger("Span Annotation Pipeline")


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
        self._historical_total_spans = 0  # Total spans from all sessions (loaded from metadata)
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
            self.telemetry["last_sequence_time"] = sequence_time  # Store duration in seconds
        
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
            self.telemetry["last_sequence_time"] = sequence_time  # Store duration in seconds
    
    def display_progress_panel(self):
        """Display comprehensive telemetry panel with progress and statistics."""
        try:
            current_time = datetime.now()
            
            # Calculate progress metrics
            processed_sequences = self.telemetry["completed_sequences"] + self.telemetry["failed_sequences"]
            progress_pct = (self.telemetry["completed_sequences"] / max(self.telemetry["total_sequences"], 1)) * 100
            success_rate = (self.telemetry["completed_sequences"] / max(processed_sequences, 1)) * 100
            
            # Calculate timing metrics
            elapsed_time = 0
            sequences_per_min = 0
            eta_display = "calculating..."
            current_session_processed = len(self.telemetry["sequence_times"])
            
            if self.telemetry["start_time"]:
                elapsed_seconds = (current_time - self.telemetry["start_time"]).total_seconds()
                elapsed_time = elapsed_seconds / 60  # Convert to minutes
            
                # Calculate processing rate - use total progress for more stable ETA
                # Use the higher of current session processing rate or overall processing rate
                sequences_per_min = 0
                eta_display = "Calculating..."
                
                # Current session rate (for immediate feedback)
                current_session_rate = 0
                if elapsed_seconds > 0 and current_session_processed > 0:
                    current_session_rate = (current_session_processed * 60) / elapsed_seconds
                
                # Overall rate based on total completed sequences
                # This provides more stable ETA estimates, especially after restarts
                total_processed = self.telemetry["completed_sequences"] + current_session_processed
                overall_rate = 0
                if elapsed_seconds > 0 and total_processed > 0:
                    overall_rate = (total_processed * 60) / elapsed_seconds
                
                # Use the more reliable rate (prefer overall rate for stability, but use current session if higher)
                if total_processed >= 5:  # Only use overall rate if we have sufficient data
                    sequences_per_min = max(overall_rate, current_session_rate * 0.8)  # Slight preference for overall rate
                elif current_session_processed >= 3:  # Use current session if we have some data
                    sequences_per_min = current_session_rate
                
                # Calculate remaining work and ETA
                if sequences_per_min > 0:
                    remaining_new_sequences = self.telemetry["total_sequences"] - self.telemetry["completed_sequences"]
                    remaining_work = remaining_new_sequences + self.telemetry["failed_sequences"]
                    
                    if remaining_work > 0:
                        eta_minutes = remaining_work / sequences_per_min
                        eta_display = self._format_eta(eta_minutes)
                    else:
                        eta_display = "Complete!"
            
            # Calculate span statistics - combine historical and current session data
            total_spans = sum(self.telemetry["spans_by_type"].values())
            span_types_summary = self._format_span_summary(self.telemetry["spans_by_type"])
            modality_summary = self._format_span_summary(self.telemetry["spans_by_modality"])
            
            # Calculate average sequence processing time for current session
            avg_seq_time = 0
            if self.telemetry["sequence_times"]:
                avg_seq_time = sum(self.telemetry["sequence_times"]) / len(self.telemetry["sequence_times"])
            
            # Calculate remaining work breakdown for display
            remaining_new_sequences = self.telemetry["total_sequences"] - self.telemetry["completed_sequences"] 
            total_remaining_work = remaining_new_sequences + self.telemetry["failed_sequences"]
            
            # Display telemetry panel
            logger.info("=" * 80)
            logger.info(f"[TELEMETRY] {self.pipeline_name} Progress Panel")
            logger.info("=" * 80)
            logger.info(f"[TELEMETRY] Overall Progress: {processed_sequences}/{self.telemetry['total_sequences']} sequences ({progress_pct:.1f}%)")
            logger.info(f"[TELEMETRY] Success Rate: {self.telemetry['completed_sequences']}/{processed_sequences} successful ({success_rate:.1f}%)")
            logger.info(f"[TELEMETRY] Failed Sequences: {self.telemetry['failed_sequences']} (need retry)")
            logger.info(f"[TELEMETRY] Remaining Work: {total_remaining_work} sequences ({remaining_new_sequences} new + {self.telemetry['failed_sequences']} retries)")
            logger.info("-" * 40)
            
            # Show current session performance metrics
            if current_session_processed > 0:
                logger.info(f"[TELEMETRY] Current Session: {current_session_processed} sequences processed")
                if elapsed_seconds > 0:
                    current_session_rate = (current_session_processed * 60) / elapsed_seconds
                    total_processed = self.telemetry["completed_sequences"] + current_session_processed
                    if total_processed > 0:
                        overall_rate = (total_processed * 60) / elapsed_seconds
                        logger.info(f"[TELEMETRY] Current Session Rate: {current_session_rate:.2f} sequences/min")
                        logger.info(f"[TELEMETRY] Overall Processing Rate: {overall_rate:.2f} sequences/min")
                    else:
                        logger.info(f"[TELEMETRY] Current Session Rate: {current_session_rate:.2f} sequences/min")
                logger.info(f"[TELEMETRY] Session Average Time: {avg_seq_time:.1f} seconds per sequence")
                logger.info(f"[TELEMETRY] Session Duration: {elapsed_time:.1f} minutes")
                logger.info(f"[TELEMETRY] ETA (based on optimal rate): {eta_display}")
            else:
                logger.info("[TELEMETRY] Current Session: No sequences processed yet")
                logger.info(f"[TELEMETRY] Session Duration: {elapsed_time:.1f} minutes")
                logger.info("[TELEMETRY] ETA: Calculating...")
            
            logger.info("-" * 40)
            
            # Calculate span statistics for display
            current_session_spans = sum(self.telemetry["spans_by_type"].values())
            if self._historical_total_spans > 0:
                # Show both historical total and current session breakdown
                total_all_sessions = self._historical_total_spans + current_session_spans
                logger.info(f"[TELEMETRY] Total Spans Extracted (All Sessions): {total_all_sessions}")
                logger.info(f"[TELEMETRY] Current Session Spans: {current_session_spans}")
                logger.info(f"[TELEMETRY] Previous Sessions Spans: {self._historical_total_spans}")
            else:
                logger.info(f"[TELEMETRY] Total Spans Extracted: {current_session_spans}")
                
            if span_types_summary:
                logger.info(f"[TELEMETRY] Span Types (Current Session): {span_types_summary}")
            if modality_summary:
                logger.info(f"[TELEMETRY] Modalities (Current Session): {modality_summary}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"[TELEMETRY] Error displaying progress panel: {e}")
            logger.error(f"[TELEMETRY] Exception type: {type(e).__name__}")
            logger.error(f"[TELEMETRY] Telemetry data keys: {list(self.telemetry.keys())}")
            # Display basic fallback information
            logger.info("=" * 80)
            logger.info(f"[TELEMETRY] {self.pipeline_name} Progress (Basic View)")
            logger.info("=" * 80)
            logger.info(f"[TELEMETRY] Progress: {self.telemetry.get('completed_sequences', 0)} completed, {self.telemetry.get('failed_sequences', 0)} failed")
            logger.info(f"[TELEMETRY] Total sequences: {self.telemetry.get('total_sequences', 0)}")
            logger.info("=" * 80)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current telemetry statistics.
        
        Returns:
            Dictionary of telemetry statistics
        """
        processed_sequences = self.telemetry["completed_sequences"] + self.telemetry["failed_sequences"]
        success_rate = (self.telemetry["completed_sequences"] / max(processed_sequences, 1)) * 100
        
        # Calculate timing metrics with improved ETA logic
        elapsed_time = 0
        sequences_per_min = 0
        eta_minutes = 0
        current_session_processed = len(self.telemetry["sequence_times"])
        
        if self.telemetry["start_time"]:
            elapsed_seconds = (datetime.now() - self.telemetry["start_time"]).total_seconds()
            elapsed_time = elapsed_seconds / 60
            
            if elapsed_seconds > 0:
                # Current session rate
                current_session_rate = 0
                if current_session_processed > 0:
                    current_session_rate = (current_session_processed * 60) / elapsed_seconds
                
                # Overall rate based on total completed sequences
                total_processed = self.telemetry["completed_sequences"] + current_session_processed
                overall_rate = 0
                if total_processed > 0:
                    overall_rate = (total_processed * 60) / elapsed_seconds
                
                # Use the more reliable rate for ETA calculation
                if total_processed >= 5:  # Prefer overall rate for stability
                    sequences_per_min = max(overall_rate, current_session_rate * 0.8)
                elif current_session_processed >= 3:  # Use current session if we have some data
                    sequences_per_min = current_session_rate
                else:
                    sequences_per_min = current_session_rate  # Use what we have
        
        # Calculate average sequence processing time for current session
        avg_seq_time = 0
        if self.telemetry["sequence_times"]:
            avg_seq_time = sum(self.telemetry["sequence_times"]) / len(self.telemetry["sequence_times"])
        
        # Calculate ETA based on remaining work and optimal processing rate
        remaining_new_sequences = self.telemetry["total_sequences"] - self.telemetry["completed_sequences"]
        remaining_work = remaining_new_sequences + self.telemetry["failed_sequences"]
        if sequences_per_min > 0 and remaining_work > 0:
            eta_minutes = remaining_work / sequences_per_min
        
        return {
            "total_sequences": self.telemetry["total_sequences"],
            "completed_sequences": self.telemetry["completed_sequences"],
            "failed_sequences": self.telemetry["failed_sequences"],
            "processed_sequences": processed_sequences,
            "success_rate_percent": success_rate,
            "elapsed_time_minutes": elapsed_time,
            "processing_rate_per_min": sequences_per_min,
            "average_sequence_time_seconds": avg_seq_time,
            "eta_minutes": eta_minutes,
            "current_session_processed": current_session_processed,
            "total_spans": sum(self.telemetry["spans_by_type"].values()),
            "historical_total_spans": self._historical_total_spans,
            "spans_by_type": dict(self.telemetry["spans_by_type"]),
            "spans_by_modality": dict(self.telemetry["spans_by_modality"])
        }
    
    def save_telemetry_to_metadata(self, metadata_filepath: Path):
        """
        Save current telemetry state to the metadata.json file.
        
        Args:
            metadata_filepath: Path to metadata.json file
        """
        try:
            # Load existing metadata
            if metadata_filepath.exists():
                with open(metadata_filepath, 'r', encoding='utf-8') as f:
                    metadata: Dict[str, Any] = json.load(f)
            else:
                metadata: Dict[str, Any] = {
                    "pipeline_version": "1.0",
                    "started_at": datetime.now().isoformat()
                }
            
            # Update metadata with telemetry data
            telemetry_data = dict(self.telemetry)
            
            # Convert datetime objects to ISO strings for JSON serialization
            if telemetry_data["start_time"]:
                if isinstance(telemetry_data["start_time"], datetime):
                    telemetry_data["start_time"] = telemetry_data["start_time"].isoformat()
                else:
                    logger.warning(f"start_time is not datetime: {type(telemetry_data['start_time'])}")
            
            if telemetry_data["last_sequence_time"]:
                if isinstance(telemetry_data["last_sequence_time"], (int, float)):
                    # last_sequence_time is now stored as duration in seconds, no conversion needed
                    pass
                elif isinstance(telemetry_data["last_sequence_time"], datetime):
                    # Handle legacy case where it was stored as datetime - convert to None
                    logger.warning("Converting legacy datetime last_sequence_time to None")
                    telemetry_data["last_sequence_time"] = None
                else:
                    logger.warning(f"last_sequence_time is unexpected type: {type(telemetry_data['last_sequence_time'])}")
                    telemetry_data["last_sequence_time"] = None
            
            # Add comprehensive telemetry section
            metadata["telemetry"] = {
                "pipeline_name": self.pipeline_name,
                "session_data": telemetry_data,
                "current_statistics": self.get_statistics(),
                "last_updated": datetime.now().isoformat()
            }
            
            # Also update the legacy fields for backward compatibility
            stats = self.get_statistics()
            metadata["processed_sequences"] = stats["processed_sequences"]
            metadata["total_sequences"] = stats["total_sequences"]
            
            # Update total spans with combined total (historical + current session)
            current_session_spans = sum(self.telemetry["spans_by_type"].values())
            combined_total_spans = self._historical_total_spans + current_session_spans
            metadata["total_spans"] = combined_total_spans
            
            metadata["last_updated"] = datetime.now().isoformat()
            
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Telemetry state saved to {metadata_filepath}")
        except Exception as e:
            logger.error(f"Failed to save telemetry state: {e}")

    def save_telemetry_state(self, filepath: Path):
        """
        Save current telemetry state to a JSON file (legacy method).
        
        Args:
            filepath: Path to save telemetry state
        """
        try:
            state = {
                "pipeline_name": self.pipeline_name,
                "telemetry_snapshot": dict(self.telemetry),
                "statistics": self.get_statistics(),
                "saved_at": datetime.now().isoformat()
            }
            
            # Convert datetime objects to ISO strings for JSON serialization
            if state["telemetry_snapshot"]["start_time"]:
                state["telemetry_snapshot"]["start_time"] = state["telemetry_snapshot"]["start_time"].isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
            logger.debug(f"Telemetry state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save telemetry state: {e}")
    
    def load_telemetry_from_metadata(self, metadata_filepath: Path) -> bool:
        """
        Load telemetry state from the metadata.json file.
        
        Args:
            metadata_filepath: Path to metadata.json file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not metadata_filepath.exists():
                return False
                
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check if telemetry data exists in metadata
            if "telemetry" not in metadata:
                logger.info("No telemetry data found in metadata.json - starting fresh")
                return False
            
            telemetry_section = metadata["telemetry"]
            telemetry_data = telemetry_section.get("session_data", {})
            
            if not telemetry_data:
                logger.info("Empty telemetry session data - starting fresh")
                return False
            
            # Convert ISO string back to datetime for start_time
            if telemetry_data.get("start_time"):
                telemetry_data["start_time"] = datetime.fromisoformat(telemetry_data["start_time"])
            
            # Ensure last_sequence_time is properly typed as float
            if "last_sequence_time" in telemetry_data:
                try:
                    telemetry_data["last_sequence_time"] = float(telemetry_data["last_sequence_time"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid last_sequence_time value, removing: {telemetry_data.get('last_sequence_time')}")
                    telemetry_data.pop("last_sequence_time", None)
            
            # Load historical span statistics from metadata root level (legacy compatibility)
            # This includes the total spans calculated from all working files
            if "total_spans" in metadata:
                historical_total_spans = metadata["total_spans"]
                logger.debug(f"Found total spans in metadata: {historical_total_spans}")
                
                # Calculate current session spans from loaded telemetry data
                current_session_spans = sum(telemetry_data.get("spans_by_type", {}).values())
                
                if current_session_spans > 0:
                    # If we have current session data, the historical total should be
                    # the metadata total minus the current session spans
                    self._historical_total_spans = max(0, historical_total_spans - current_session_spans)
                    logger.debug(f"Calculated historical spans: {self._historical_total_spans} (metadata total: {historical_total_spans} - current session: {current_session_spans})")
                else:
                    # If no current session data, all spans are historical
                    self._historical_total_spans = historical_total_spans
                    logger.info(f"No current session spans found, treating all {historical_total_spans} spans as historical")
                    logger.info("Note: Detailed span type breakdown only available for current session")
            
            # Restore telemetry data
            self.telemetry = telemetry_data
            
            # Log resume information
            stats = telemetry_section.get("current_statistics", {})
            logger.info(f"Telemetry state loaded from {metadata_filepath}")
            logger.info(f"Resuming from: {stats.get('processed_sequences', 0)} processed sequences")
            logger.info(f"Previous session success rate: {stats.get('success_rate_percent', 0):.1f}%")
            
            # Add historical context to display
            # Note: _historical_total_spans is already calculated above based on metadata and current session
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load telemetry state from metadata: {e}")
            return False

    def load_telemetry_state(self, filepath: Path) -> bool:
        """
        Load telemetry state from a JSON file (legacy method).
        
        Args:
            filepath: Path to load telemetry state from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not filepath.exists():
                return False
                
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore telemetry data
            telemetry_data = state["telemetry_snapshot"]
            
            # Convert ISO string back to datetime
            if telemetry_data["start_time"]:
                telemetry_data["start_time"] = datetime.fromisoformat(telemetry_data["start_time"])
            
            self.telemetry = telemetry_data
            
            logger.info(f"Telemetry state loaded from {filepath}")
            logger.info(f"Resuming from: {state['statistics']['processed_sequences']} processed sequences")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load telemetry state: {e}")
            return False
    
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
