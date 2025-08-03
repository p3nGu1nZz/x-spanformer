"""
Asynchronous span annotator agent for X-bar linguistic analysis.

Implements multi-turn conversation strategy for comprehensive syntactic
span annotation using X-bar theory with position-wise embedding alignment.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.ollama_client import chat
from x_spanformer.agents.prompts import (
    render_span_annotator_system_prompt,
    render_span_annotation_request,
    render_span_annotation_followup
)
from x_spanformer.xbar.position_mapper import (
    PositionMapper, 
    CharacterSpan, 
    PositionSpan
)
from x_spanformer.xbar.xbar_map import XBarClassifierMap, DomainType
from x_spanformer.schema.annotation_record import (
    AnnotationRecord, 
    SpanAnnotation, 
    AnnotationBatch
)
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.metadata import RecordMeta


logger = logging.getLogger(__name__)


class DialogueAgent:
    """
    Simple async wrapper around DialogueManager for LLM conversations.
    
    Provides async interface for multi-turn conversations with session management.
    """
    
    def __init__(self, model_name: str = "gpt-4o", max_turns: int = 16, temperature: float = 0.1):
        self.model_name = model_name
        self.max_turns = max_turns
        self.temperature = temperature
        self.sessions: Dict[str, DialogueManager] = {}
    
    async def start_session(self, session_id: str, system_prompt: str):
        """Start a new dialogue session."""
        self.sessions[session_id] = DialogueManager(system_prompt=system_prompt, max_turns=self.max_turns)
    
    async def send_message(self, session_id: str, message: str) -> str:
        """Send message and get response using Ollama client."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        dialogue = self.sessions[session_id]
        dialogue.add("user", message)
        
        # Use real LLM via ollama_client, with fallback for testing
        messages = dialogue.as_messages()
        try:
            response = await chat(
                model=self.model_name,
                conversation=[{"role": msg["role"], "content": msg["content"]} for msg in messages[1:]],  # Skip system
                system=messages[0]["content"],  # System prompt
                temperature=self.temperature
            )
        except Exception as e:
            # Fallback to mock for testing when Ollama not available
            if "test-model" in self.model_name or "ConnectionError" in str(e):
                response = self._generate_mock_response(message)
            else:
                raise e
        
        dialogue.add("assistant", response)
        return response
    
    async def end_session(self, session_id: str):
        """End dialogue session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _generate_mock_response(self, message: str) -> str:
        """Generate mock LLM response for testing."""
        return '''Based on X-bar theory analysis:

"The quick brown fox" (0-19) -> NP [confidence: 0.88]
"jumps" (20-25) -> V [confidence: 0.95]
"over the lazy dog" (26-43) -> PP [confidence: 0.87]

The sentence shows a simple clause structure with a complex noun phrase subject, intransitive verb, and prepositional phrase adjunct.'''


@dataclass
class AnnotationTask:
    """Single annotation task for async processing."""
    sequence_id: int
    text: str
    embedding_chunk_id: int
    pretrain_record: PretrainRecord
    priority: int = 0
    retry_count: int = 0


@dataclass
class AnnotationResult:
    """Result of span annotation processing."""
    sequence_id: int
    annotation_record: Optional[AnnotationRecord]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    turns_used: int = 0


class SpanAnnotatorSession:
    """
    Asynchronous span annotator session using X-bar theory.
    
    Implements multi-turn conversation priming strategy for comprehensive
    linguistic analysis with position-wise embedding alignment.
    
    ARCHITECTURE ALIGNMENT - Two-Stage Process:
    
    STAGE 1 (This Session): Boundary Detection Training Data
    The annotations produced by this session feed into boundary detection training
    (Section 3.3). Each SpanAnnotation creates training targets for boundary heads:
    
    y_start[span.start_pos] = 1.0  # Train start boundary detection
    y_end[span.end_pos] = 1.0      # Train end boundary detection
    
    The boundary heads learn from contextual position embeddings H[t]:
    - Start head: W_start @ H[start_pos] -> P(span starts here)
    - End head: W_end @ H[end_pos] -> P(span ends here)
    
    Key insight: Middle positions are IGNORED during boundary training.
    Only start/end positions matter for learning boundary detection.
    
    STAGE 2 (Section 4 Training): Full Span Representation
    Once boundary detection is trained, the full X-Spanformer model uses
    detected boundaries to create actual span embeddings through gated
    fusion, span interpolation, and hierarchical multi-label classification.
    
    This session provides the foundation for Stage 1 boundary detection.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_concurrent: int = 5,
        max_retries: int = 3,
        conversation_timeout: float = 30.0,
        max_turns: int = 16,
        temperature: float = 0.1,
        early_termination_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize span annotator agent.
        
        Args:
            model_name: LLM model for linguistic analysis
            max_concurrent: Maximum concurrent annotation requests
            max_retries: Maximum retry attempts per sequence
            conversation_timeout: Timeout for single conversation (seconds)
            max_turns: Maximum conversation turns per sequence
            temperature: Temperature parameter for model inference
            early_termination_config: Early termination settings
        """
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.conversation_timeout = conversation_timeout
        self.max_turns = max_turns
        self.temperature = temperature
        
        # Early termination settings
        self.early_termination = early_termination_config or {
            "enable": True,
            "no_improvement_threshold": 2,
            "failed_extraction_threshold": 2
        }
        
        # Initialize dialogue agent
        self.dialogue_agent = DialogueAgent(model_name=model_name, max_turns=max_turns, temperature=temperature)
        
        # Async processing controls
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue: asyncio.Queue[AnnotationTask] = asyncio.Queue()
        self.result_queue: asyncio.Queue[AnnotationResult] = asyncio.Queue()
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_spans": 0,
            "total_turns": 0,
            "total_time": 0.0
        }
    
    def _detect_response_repetition(self, current_response: str, recent_responses: List[str]) -> bool:
        """
        Detect if the agent is repeating the same response.
        
        Args:
            current_response: Current agent response
            recent_responses: List of recent responses to compare against
            
        Returns:
            True if repetition detected, False otherwise
        """
        if not recent_responses:
            return False
            
        # Simple repetition check - exact match
        if current_response in recent_responses:
            return True
            
        # More sophisticated check - similar JSON structure
        try:
            import json
            import re
            
            # Extract JSON from responses
            json_pattern = r'```json\s*(.*?)\s*```'
            
            current_json_match = re.search(json_pattern, current_response, re.DOTALL)
            if not current_json_match:
                return False
                
            current_json_str = current_json_match.group(1).strip()
            
            for recent_response in recent_responses[-2:]:  # Check last 2 responses
                recent_json_match = re.search(json_pattern, recent_response, re.DOTALL)
                if not recent_json_match:
                    continue
                    
                recent_json_str = recent_json_match.group(1).strip()
                
                # Compare JSON structures (ignoring whitespace)
                if current_json_str == recent_json_str:
                    return True
                    
                # Try parsing and comparing as actual JSON
                try:
                    current_data = json.loads(current_json_str)
                    recent_data = json.loads(recent_json_str)
                    
                    if current_data == recent_data:
                        return True
                        
                except json.JSONDecodeError:
                    continue
                    
        except Exception:
            pass
            
        return False
    
    def get_xbar_system_prompt(self, domain_type: str = "natural") -> str:
        """
        Get the system prompt for X-bar syntactic analysis.
        
        Args:
            domain_type: Content domain type ("natural", "code", "mixed")
        
        Returns:
            System prompt for linguistic span annotation
        """
        return render_span_annotator_system_prompt(domain_type=domain_type)

    def get_initial_annotation_request(self, text: str, domain_type: str = "natural") -> str:
        """
        Generate initial annotation request.
        
        Args:
            text: Text sequence to analyze
            domain_type: Content domain type ("natural", "code", "mixed")
            
        Returns:
            Initial annotation request
        """
        return render_span_annotation_request(
            text=text,
            domain_type=domain_type,
            turn_number=1,
            max_turns=self.max_turns
        )

    def generate_followup_request(self, text: str, domain_type: str, turn_number: int, previous_spans: List[str]) -> str:
        """
        Generate follow-up request to find missing spans.
        
        Args:
            text: Original text sequence
            domain_type: Content domain type
            turn_number: Current turn number
            previous_spans: List of span texts already found
            
        Returns:
            Follow-up request focusing on missing elements
        """
        # Analyze what types of spans might be missing
        words = text.split()
        found_span_texts = set(span.lower().strip() for span in previous_spans)
        
        # Check for missing individual words
        missing_words = []
        for word in words:
            clean_word = word.strip('.,!?;:').lower()
            if clean_word not in found_span_texts:
                missing_words.append(word)
        
        return render_span_annotation_followup(
            text=text,
            domain_type=domain_type,
            turn_number=turn_number,
            previous_spans=previous_spans,
            missing_words=missing_words
        )
    
    async def annotate_single_sequence(
        self, 
        task: AnnotationTask,
        progress_callback: Optional[Any] = None
    ) -> AnnotationResult:
        """
        Annotate a single text sequence with X-bar spans.
        
        Args:
            task: Annotation task to process
            
        Returns:
            Annotation result with spans or error information
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.semaphore:
                # Log which sequence is being processed (now inside semaphore)
                logger.info(f"[PROCESSING] Sequence {task.sequence_id} (length: {len(task.text)} chars)")
                # Initialize position mapper
                mapper = PositionMapper(task.text)
                
                # Determine domain type from pretrain record
                domain_type = task.pretrain_record.type or "natural"
                
                # Start dialogue session
                session_id = f"annotation-{task.sequence_id}-{datetime.now().isoformat()}"
                
                await self.dialogue_agent.start_session(
                    session_id=session_id,
                    system_prompt=self.get_xbar_system_prompt(domain_type)
                )
                
                # Iterative multi-turn conversation for comprehensive analysis
                all_responses = []
                all_spans = []
                turns_used = 0
                max_turns = self.max_turns
                
                # Initial annotation request
                initial_request = self.get_initial_annotation_request(task.text, domain_type)
                
                try:
                    response = await asyncio.wait_for(
                        self.dialogue_agent.send_message(
                            session_id=session_id,
                            message=initial_request
                        ),
                        timeout=self.conversation_timeout
                    )
                    all_responses.append({
                        "role": "user",
                        "content": initial_request
                    })
                    all_responses.append({
                        "role": "assistant",
                        "content": response
                    })
                    turns_used += 1
                    
                    # Call progress callback after initial turn
                    if progress_callback:
                        await progress_callback({
                            "sequence_id": task.sequence_id,
                            "turn": turns_used,
                            "total_spans": 0,  # Will be updated after extraction
                            "phase": "initial_request"
                        })
                    
                    # Extract spans from initial response
                    from x_spanformer.pipelines.shared.annotation_processor import AnnotationProcessor
                    processor = AnnotationProcessor()
                    domain_enum = {
                        "natural": DomainType.NATURAL,
                        "code": DomainType.CODE,
                        "mixed": DomainType.MIXED
                    }.get(domain_type, DomainType.NATURAL)
                    
                    initial_spans = processor.extract_spans_from_comprehensive_response(
                        response,
                        task.text,
                        XBarClassifierMap.get_classifier_names(domain_enum)
                    )
                    all_spans.extend(initial_spans)
                    
                    # Update progress after initial extraction
                    if progress_callback:
                        await progress_callback({
                            "sequence_id": task.sequence_id,
                            "turn": turns_used,
                            "total_spans": len(all_spans),
                            "new_spans": len(initial_spans),
                            "phase": "initial_extraction"
                        })
                    
                    # Continue with follow-up turns to find missing spans
                    previous_span_count = len(all_spans)
                    no_improvement_count = 0
                    consecutive_failed_extractions = 0
                    
                    no_improvement_threshold = self.early_termination.get("no_improvement_threshold", 2)
                    failed_extraction_threshold = self.early_termination.get("failed_extraction_threshold", 2)
                    early_termination_enabled = self.early_termination.get("enable", True)
                    
                    while (turns_used < max_turns and 
                           no_improvement_count < no_improvement_threshold and 
                           consecutive_failed_extractions < failed_extraction_threshold):
                        # Generate follow-up request
                        span_texts = [span.text for span in all_spans]
                        followup_request = self.generate_followup_request(
                            task.text, domain_type, turns_used + 1, span_texts
                        )
                        
                        try:
                            followup_response = await asyncio.wait_for(
                                self.dialogue_agent.send_message(
                                    session_id=session_id,
                                    message=followup_request
                                ),
                                timeout=self.conversation_timeout
                            )
                            
                            all_responses.append({
                                "role": "user",
                                "content": followup_request
                            })
                            all_responses.append({
                                "role": "assistant",
                                "content": followup_response
                            })
                            turns_used += 1
                            
                            # Progress callback for followup turn
                            if progress_callback:
                                await progress_callback({
                                    "sequence_id": task.sequence_id,
                                    "turn": turns_used,
                                    "total_spans": len(all_spans),  # Will be updated after extraction
                                    "phase": "followup_request"
                                })
                            
                            # Extract new spans
                            new_spans = processor.extract_spans_from_comprehensive_response(
                                followup_response,
                                task.text,
                                XBarClassifierMap.get_classifier_names(domain_enum)
                            )
                            
                            # Track extraction success
                            if not new_spans:
                                consecutive_failed_extractions += 1
                                logger.warning(f"No spans extracted from follow-up turn {turns_used} for sequence {task.sequence_id}")
                            else:
                                consecutive_failed_extractions = 0
                            
                            # Add only truly new spans (avoid duplicates)
                            initial_span_count = len(all_spans)
                            for new_span in new_spans:
                                span_key = (new_span.start_char, new_span.end_char, new_span.xbar_class)
                                existing_keys = {(s.start_char, s.end_char, s.xbar_class) for s in all_spans}
                                if span_key not in existing_keys:
                                    all_spans.append(new_span)
                            
                            # Check for improvement
                            if len(all_spans) == initial_span_count:
                                no_improvement_count += 1
                                logger.info(f"No new spans found in turn {turns_used} (no improvement count: {no_improvement_count})")
                            else:
                                no_improvement_count = 0
                                logger.info(f"Found {len(all_spans) - initial_span_count} new spans in turn {turns_used}")
                            
                            # Progress callback after extraction
                            if progress_callback:
                                await progress_callback({
                                    "sequence_id": task.sequence_id,
                                    "turn": turns_used,
                                    "total_spans": len(all_spans),
                                    "new_spans": len(all_spans) - initial_span_count,
                                    "phase": "followup_extraction",
                                    "no_improvement_count": no_improvement_count,
                                    "consecutive_failed": consecutive_failed_extractions
                                })
                            
                            # Early termination if LLM is clearly not improving
                            if early_termination_enabled and consecutive_failed_extractions >= failed_extraction_threshold:
                                logger.info(f"Terminating early due to consecutive failed extractions for sequence {task.sequence_id}")
                                break
                            
                            # Small delay between turns
                            await asyncio.sleep(0.1)
                            
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout on turn {turns_used + 1} for sequence {task.sequence_id}")
                            break
                        except Exception as e:
                            logger.error(f"Error in turn {turns_used + 1} for sequence {task.sequence_id}: {e}")
                            break
                    
                    logger.info(f"Completed annotation for sequence {task.sequence_id}: {len(all_spans)} spans in {turns_used} turns")
                    
                except Exception as e:
                    logger.error(f"Failed initial annotation for sequence {task.sequence_id}: {e}")
                    all_spans = []
                    # Ensure we still capture any conversation history that occurred before the failure
                    if turns_used == 0:
                        # Even if no turns completed, add the error info to responses
                        all_responses.append({
                            "role": "user",
                            "content": "Initial annotation request failed due to error"
                        })
                        all_responses.append({
                            "role": "assistant", 
                            "content": f"ERROR: {str(e)}"
                        })
                
                # Convert to position spans using collected spans
                char_spans = all_spans
                logger.info(f"Total character spans collected: {len(char_spans)}")
                
                # Convert to position spans
                position_spans = mapper.batch_char_to_position(char_spans)
                
                # Validate spans
                validated_spans = []
                for pos_span, issues in mapper.validate_position_spans(position_spans):
                    if not issues:
                        validated_spans.append(SpanAnnotation(
                            start_pos=pos_span.start_pos,
                            end_pos=pos_span.end_pos,
                            xbar_class=pos_span.xbar_class,
                            confidence=1.0,  # Always 1.0 since we trust the LLM output
                            linguistic_features={
                                "text": mapper.get_position_text(pos_span.start_pos, pos_span.end_pos),
                                "length": pos_span.end_pos - pos_span.start_pos
                            }
                        ))
                    else:
                        logger.warning(f"Invalid span for sequence {task.sequence_id}: {issues}")
                
                # Create annotation record
                # Validate conversation_turns format before creating AnnotationRecord
                validated_conversation_turns = []
                for turn in all_responses:
                    if not isinstance(turn, dict):
                        logger.warning(f"Invalid turn format (not dict): {turn}")
                        continue
                    if "role" not in turn or "content" not in turn:
                        logger.warning(f"Invalid turn format (missing role/content): {turn}")
                        continue
                    if not isinstance(turn["role"], str) or not isinstance(turn["content"], str):
                        logger.warning(f"Invalid turn format (non-string values): {turn}")
                        continue
                    validated_conversation_turns.append(turn)
                
                annotation_record = AnnotationRecord(
                    raw=task.text,
                    sequence_id=task.sequence_id,
                    embedding_chunk_id=task.embedding_chunk_id,
                    span_annotations=validated_spans,
                    total_positions=len(task.text),
                    conversation_turns=validated_conversation_turns,  # Use validated conversation
                    agent_metadata={
                        "model": self.model_name,
                        "processing_time": asyncio.get_event_loop().time() - start_time,
                        "turns_required": turns_used,
                        "annotation_strategy": "multi_turn_xbar",
                        "spans_extracted": len(validated_spans),
                        "validation_issues": sum(len(issues) for _, issues in mapper.validate_position_spans(position_spans)),
                        "domain_type": domain_type
                    },
                    meta=RecordMeta(
                        tags=["annotation", "xbar", domain_type],  # Use domain_type directly instead of task.pretrain_record.type
                        doc_language=task.pretrain_record.meta.doc_language or "unknown",
                        extracted_by="span_annotator_agent",
                        confidence=sum(span.confidence for span in validated_spans) / len(validated_spans) if validated_spans else 0.0,
                        source_file=task.pretrain_record.meta.source_file or "unknown",
                        notes=f"Annotated {len(validated_spans)} spans using {turns_used} conversation turns"
                    )
                )
                
                # Cleanup session
                await self.dialogue_agent.end_session(session_id)
                
                processing_time = asyncio.get_event_loop().time() - start_time
                
                # Update statistics
                self.stats["total_processed"] += 1
                self.stats["total_spans"] += len(validated_spans)
                self.stats["total_turns"] += turns_used
                self.stats["total_time"] += processing_time
                
                logger.info(f"Completed annotation for sequence {task.sequence_id}: {len(validated_spans)} spans in {turns_used} turns")
                
                # Final completion callback
                if progress_callback:
                    await progress_callback({
                        "sequence_id": task.sequence_id,
                        "turn": turns_used,
                        "total_spans": len(validated_spans),
                        "phase": "completed",
                        "annotation_record": annotation_record,
                        "processing_time": processing_time
                    })
                
                return AnnotationResult(
                    sequence_id=task.sequence_id,
                    annotation_record=annotation_record,
                    success=True,
                    processing_time=processing_time,
                    turns_used=turns_used
                )
                
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Failed to annotate sequence {task.sequence_id}: {str(e)}"
            logger.error(error_msg)
            
            # Update statistics
            self.stats["total_processed"] += 1
            self.stats["failed"] += 1
            self.stats["total_time"] += processing_time
            
            return AnnotationResult(
                sequence_id=task.sequence_id,
                annotation_record=None,
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    async def annotate_batch(
        self,
        pretrain_records: List[PretrainRecord],
        batch_id: Optional[str] = None,
        progress_callback: Optional[Any] = None
    ) -> AnnotationBatch:
        """
        Annotate a batch of pretrain records asynchronously.
        
        Args:
            pretrain_records: List of pretrain records to annotate
            batch_id: Optional batch identifier
            
        Returns:
            AnnotationBatch with results
        """
        if not batch_id:
            batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Starting batch annotation {batch_id} with {len(pretrain_records)} sequences")
        logger.info(f"Batch sequences: {[record.sequence_number for record in pretrain_records]}")
        
        # Create annotation tasks
        tasks = []
        for i, record in enumerate(pretrain_records):
            task = AnnotationTask(
                sequence_id=record.sequence_number or i,
                text=record.raw,
                embedding_chunk_id=record.embedding_chunk_id or 0,
                pretrain_record=record,
                priority=0
            )
            tasks.append(task)
        
        # Process tasks concurrently
        async def process_with_progress(task: AnnotationTask) -> AnnotationResult:
            """Wrapper to pass progress callback to individual sequence processing."""
            return await self.annotate_single_sequence(task, progress_callback)
        
        results = await asyncio.gather(
            *[process_with_progress(task) for task in tasks],
            return_exceptions=True
        )
        
        # Collect successful annotations
        annotation_records = []
        embedding_chunk_ids = set()
        
        for result in results:
            if isinstance(result, AnnotationResult) and result.success and result.annotation_record is not None:
                annotation_records.append(result.annotation_record)
                embedding_chunk_ids.add(result.annotation_record.embedding_chunk_id)
            elif isinstance(result, Exception):
                logger.error(f"Batch processing exception: {result}")
        
        # Create annotation batch
        batch = AnnotationBatch(
            records=annotation_records,
            batch_id=batch_id,
            embedding_chunk_ids=list(embedding_chunk_ids),
            batch_metadata={
                "total_sequences": len(pretrain_records),
                "successful_annotations": len(annotation_records),
                "failed_annotations": len(pretrain_records) - len(annotation_records),
                "processing_date": datetime.now().isoformat(),
                "agent_model": self.model_name,
                "avg_confidence": sum(
                    record.meta.confidence for record in annotation_records
                ) / len(annotation_records) if annotation_records else 0.0,
                "total_spans": sum(
                    len(record.span_annotations) for record in annotation_records
                ),
                "processing_stats": dict(self.stats)
            }
        )
        
        logger.info(f"Completed batch {batch_id}: {len(annotation_records)}/{len(pretrain_records)} successful")
        
        return batch
    
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
                self.annotate_single_sequence(
                    AnnotationTask(
                        sequence_id=record.sequence_number or i,
                        text=record.raw,
                        embedding_chunk_id=record.embedding_chunk_id or 0,
                        pretrain_record=record
                    )
                )
            )
            for i, record in enumerate(pretrain_records)
        ]
        
        # Yield results as they complete
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                yield result
            except Exception as e:
                logger.error(f"Stream annotation error: {e}")
                yield AnnotationResult(
                    sequence_id=-1,
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
            stats["avg_turns_per_sequence"] = stats["total_turns"] / stats["successful"] if stats["successful"] > 0 else 0
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
