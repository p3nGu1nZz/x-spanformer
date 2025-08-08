"""
Unified Span Annotator Session for X-Spanformer

Consolidated async session management for hierarchical X-bar span annotation
combining the best features from the previous implementations with enhanced
three-turn conversation strategy and robust error handling.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Core X-Spanformer imports
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.annotation_record import AnnotationRecord, SpanAnnotation
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.xbar.xbar_annotator import XBarAnnotator, ModelConfig
from x_spanformer.xbar.xbar_map import DomainType

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class AnnotationTask:
    """Single annotation task for async processing."""
    sequence_number: int
    text: str
    pretrain_record: PretrainRecord
    priority: int = 0
    retry_count: int = 0


@dataclass
class AnnotationResult:
    """Result of span annotation processing."""
    sequence_number: int
    annotation_record: Optional[AnnotationRecord]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    turns_used: int = 3  # Always 3 for three-turn annotation


class SpanAnnotatorSession:
    """
    Unified async span annotator session using three-turn X-bar theory.
    
    Combines the robust async processing from the original session with
    the simplified three-turn approach of the unified annotator.
    
    Features:
    - Three-turn hierarchical annotation (word → phrase → clause)
    - Async batch processing with concurrency control
    - Early termination and error handling
    - Progress tracking and statistics
    - Resume capability and validation
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        max_retries: int = 3,
        conversation_timeout: float = 180.0,
        temperature: float = 0.2,
        early_termination_config: Optional[Dict[str, Any]] = None,
        max_spans_per_sequence: int = 64
    ):
        """
        Initialize unified span annotator session.
        
        Args:
            model_name: LLM model for linguistic analysis
            max_retries: Maximum retry attempts per sequence
            conversation_timeout: Timeout for single conversation (seconds)
            temperature: Temperature parameter for model inference
            early_termination_config: Early termination settings (legacy, not used in three-turn)
            max_spans_per_sequence: Maximum spans to generate per sequence
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.conversation_timeout = conversation_timeout
        self.temperature = temperature
        self.max_spans_per_sequence = max_spans_per_sequence
        
        # Initialize span annotator
        model_config = ModelConfig(
            name=model_name,
            temperature=temperature,
            timeout=conversation_timeout
        )
        self.annotator = XBarAnnotator(model_config)
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_spans": 0,
            "total_turns": 0,  # Always 3 per successful sequence
            "total_time": 0.0
        }
    
    async def annotate_single_sequence(
        self, 
        pretrain_record: PretrainRecord,
        progress_callback: Optional[Any] = None
    ) -> AnnotationResult:
        """
        Annotate a single text sequence with three-turn X-bar spans.
        
        Args:
            pretrain_record: PretrainRecord to process
            progress_callback: Optional progress callback function
            
        Returns:
            Annotation result with spans or error information
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Starting annotation: sequence {pretrain_record.sequence_number}")
        logger.info(f"Text length: {len(pretrain_record.raw)} chars")
        logger.info(f"Preview: {pretrain_record.raw[:100]}{'...' if len(pretrain_record.raw) > 100 else ''}")
        
        try:
            logger.info(f"Processing sequence {pretrain_record.sequence_number}")
            
            # Call progress callback if provided
            if progress_callback:
                await progress_callback({
                    "sequence_number": pretrain_record.sequence_number,
                    "turn": 0,
                    "total_spans": 0,
                    "phase": "starting"
                })
            
            # Use the three-turn annotator
            annotation_record = await asyncio.wait_for(
                self.annotator.annotate_sequence(pretrain_record),
                timeout=self.conversation_timeout
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            if annotation_record is not None:
                # Update statistics
                self.stats["total_processed"] += 1
                self.stats["successful"] += 1
                self.stats["total_spans"] += len(annotation_record.span_annotations)
                self.stats["total_turns"] += 3  # Always 3 turns
                self.stats["total_time"] += processing_time
                
                logger.info(f"SUCCESS: Sequence {pretrain_record.sequence_number} annotated")
                logger.info(f"Extracted {len(annotation_record.span_annotations)} spans (3 turns)")
                
                # Final progress callback
                if progress_callback:
                    await progress_callback({
                        "sequence_number": pretrain_record.sequence_number,
                        "turn": 3,
                        "total_spans": len(annotation_record.span_annotations),
                        "phase": "completed",
                        "annotation_record": annotation_record,
                        "processing_time": processing_time
                    })
                
                return AnnotationResult(
                    sequence_number=pretrain_record.sequence_number or 0,
                    annotation_record=annotation_record,
                    success=True,
                    processing_time=processing_time,
                    turns_used=3
                )
            else:
                # Failed annotation
                self.stats["total_processed"] += 1
                self.stats["failed"] += 1
                self.stats["total_time"] += processing_time
                
                error_msg = f"Three-turn annotation failed for sequence {pretrain_record.sequence_number or 0}"
                logger.warning(f"{error_msg}")
                
                return AnnotationResult(
                    sequence_number=pretrain_record.sequence_number or 0,
                    annotation_record=None,
                    success=False,
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
        except asyncio.TimeoutError:
            processing_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Annotation timeout after {self.conversation_timeout}s for sequence {pretrain_record.sequence_number or 0}"
            logger.error(f"{error_msg}")
            
            self.stats["total_processed"] += 1
            self.stats["failed"] += 1
            self.stats["total_time"] += processing_time
            
            return AnnotationResult(
                sequence_number=pretrain_record.sequence_number or 0,
                annotation_record=None,
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
            
        except ValueError as e:
            # Re-raise ValueError (JSON parsing errors) to allow pipeline-level sequence skipping
            logger.error(f"JSON parsing error for sequence {pretrain_record.sequence_number or 0}: {str(e)}")
            raise
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Failed to annotate sequence {pretrain_record.sequence_number or 0}: {str(e)}"
            logger.error(f"{error_msg}")
            
            self.stats["total_processed"] += 1
            self.stats["failed"] += 1
            self.stats["total_time"] += processing_time
            
            return AnnotationResult(
                sequence_number=pretrain_record.sequence_number or 0,
                annotation_record=None,
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    async def stream_annotations(
        self,
        pretrain_records: List[PretrainRecord]
    ) -> AsyncGenerator[AnnotationResult, None]:
        """
        Stream annotation results as they complete.
        
        Args:
            pretrain_records: List of pretrain records to annotate
            
        Yields:
            AnnotationResult for each completed sequence
        """
        # Create and start tasks
        tasks = [
            asyncio.create_task(
                self.annotate_single_sequence(record)
            )
            for record in pretrain_records
        ]
        
        # Yield results as they complete
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                yield result
            except Exception as e:
                logger.error(f"Stream annotation error: {e}")
                yield AnnotationResult(
                    sequence_number=-1,
                    annotation_record=None,
                    success=False,
                    error_message=str(e)
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = dict(self.stats)
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total_processed"]
            stats["avg_processing_time"] = stats["total_time"] / stats["total_processed"]
            stats["avg_spans_per_sequence"] = stats["total_spans"] / stats["successful"] if stats["successful"] > 0 else 0
            stats["avg_turns_per_sequence"] = 3.0  # Always 3 turns in unified approach
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_spans": 0,
            "total_turns": 0,
            "total_time": 0.0
        }
