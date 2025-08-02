from typing import List, Optional, Dict, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict
from x_spanformer.schema.identifier import RecordID
from x_spanformer.schema.metadata import RecordMeta


class SpanAnnotation(BaseModel):
    """
    Position-wise span annotation aligned with contextual embeddings.
    
    Uses position indices rather than character indices to align with
    the tokenizer-free embedding architecture.
    
    IMPORTANT: This annotation represents span BOUNDARIES for training
    the factorized pointer network. The training process converts these
    to binary targets:
    
    y_start[start_pos] = 1.0  # Start boundary target
    y_end[end_pos-1] = 1.0    # End boundary target (inclusive)
    
    The span is NOT represented by averaging embeddings H[start_pos:end_pos].
    Instead, boundary detection is learned from contextual position embeddings.
    """
    start_pos: int = Field(..., description="Start position index in sequence (0-based)")
    end_pos: int = Field(..., description="End position index in sequence (exclusive)")
    xbar_class: str = Field(..., description="X-bar classifier: X, X', XP, or specialized linguistic label")
    confidence: float = Field(default=1.0, description="Annotation confidence score [0.0, 1.0]")
    linguistic_features: Optional[Dict[str, Any]] = Field(default=None, description="Optional linguistic analysis from LLM agent")


class AnnotationRecord(BaseModel):
    """
    Training record for span-aware pre-training with position-wise alignment.
    
    Links raw text sequences to their contextual embeddings and linguistic
    span annotations for training the span predictor component.
    """
    # Core sequence data
    raw: str = Field(..., description="Original Unicode text sequence")
    sequence_id: int = Field(..., description="Sequential position in corpus for embedding lookup")
    embedding_chunk_id: int = Field(..., description="Chunk ID containing contextual embeddings")
    
    # Position-wise span annotations
    span_annotations: List[SpanAnnotation] = Field(default_factory=list, description="List of position-indexed span annotations")
    total_positions: int = Field(..., description="Total number of positions in sequence (for validation)")
    
    # Multi-turn conversation context (if applicable)
    conversation_turns: Optional[List[Dict[str, str]]] = Field(default=None, description="Multi-turn LLM conversation for complex analysis")
    agent_metadata: Optional[Dict[str, Any]] = Field(default=None, description="LLM agent processing metadata")
    
    # Standard metadata
    id: Optional[RecordID] = Field(default_factory=RecordID, description="Globally unique annotation record ID")
    meta: RecordMeta = Field(default_factory=lambda: RecordMeta(**{}), description="Annotation processing metadata")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "raw": "The quick brown fox jumps over the lazy dog.",
                "sequence_id": 1,
                "embedding_chunk_id": 1,
                "span_annotations": [
                    {
                        "start_pos": 0,
                        "end_pos": 3,
                        "xbar_class": "DP",
                        "confidence": 0.95,
                        "linguistic_features": {"determiner": "the", "definiteness": "definite"}
                    },
                    {
                        "start_pos": 4,
                        "end_pos": 19,
                        "xbar_class": "NP",
                        "confidence": 0.88,
                        "linguistic_features": {"head": "fox", "modifiers": ["quick", "brown"]}
                    },
                    {
                        "start_pos": 20,
                        "end_pos": 25,
                        "xbar_class": "VP",
                        "confidence": 0.92,
                        "linguistic_features": {"verb": "jumps", "tense": "present"}
                    }
                ],
                "total_positions": 44,
                "conversation_turns": [
                    {
                        "role": "user",
                        "content": "Analyze this sentence for syntactic spans using X-bar theory."
                    },
                    {
                        "role": "assistant", 
                        "content": "I'll identify the major syntactic constituents..."
                    }
                ],
                "agent_metadata": {
                    "model": "gpt-4o",
                    "processing_time": 2.3,
                    "turns_required": 2,
                    "annotation_strategy": "constituency_parse"
                },
                "id": {"id": "annotation-seq-00000001"},
                "meta": {
                    "tags": ["annotation", "xbar", "natural"],
                    "doc_language": "en",
                    "extracted_by": "span_annotator",
                    "confidence": 0.92,
                    "source_file": "corpus.jsonl",
                    "sequence_number": 1,
                    "status": "annotated"
                }
            }
        }
    )


class AnnotationBatch(BaseModel):
    """
    Batch of annotation records for efficient processing.
    
    Groups related sequences for batch annotation processing
    while maintaining individual record integrity.
    """
    records: List[AnnotationRecord] = Field(..., description="List of annotation records in batch")
    batch_id: str = Field(..., description="Unique identifier for this batch")
    embedding_chunk_ids: List[int] = Field(..., description="List of embedding chunk IDs covered by this batch")
    batch_metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch processing metadata")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "records": [],  # Would contain AnnotationRecord examples
                "batch_id": "batch-20250723-001",
                "embedding_chunk_ids": [1, 2, 3],
                "batch_metadata": {
                    "total_sequences": 300,
                    "total_annotations": 1247,
                    "processing_date": "2025-07-23",
                    "agent_model": "gpt-4o",
                    "avg_confidence": 0.89
                }
            }
        }
    )
