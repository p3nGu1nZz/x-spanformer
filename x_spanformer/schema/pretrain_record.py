from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from x_spanformer.schema.identifier import RecordID
from x_spanformer.schema.metadata import RecordMeta


class PretrainRecord(BaseModel):
    """
    Enhanced PretrainRecord schema for X-Spanformer pipelines.
    
    Supports the complete pipeline flow from corpus generation through embedding
    generation to span annotation. Compatible with all existing pipelines while
    adding support for position-wise embedding alignment.
    """
    raw: str = Field(..., description="The raw Unicode text sequence for processing")
    type: Optional[str] = Field(default=None, description="Content domain type: natural, code, or mixed")
    id: Optional[RecordID] = Field(default_factory=RecordID, description="Globally unique record identifier")
    meta: RecordMeta = Field(default_factory=lambda: RecordMeta(**{}), description="Processing metadata and source information")
    
    # New fields for embedding and annotation alignment
    sequence_number: Optional[int] = Field(default=None, description="Sequential position in corpus for embedding lookup")
    embedding_positions: Optional[int] = Field(default=None, description="Number of position-wise embeddings (sequence length)")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "raw": "The quick brown fox jumps over the lazy dog.",
                "type": "natural",
                "id": {"id": "corpus-seq-00000001"},
                "meta": {
                    "tags": ["natural", "example"],
                    "doc_language": "en",
                    "extracted_by": "jsonl2vocab",
                    "confidence": 0.95,
                    "source_file": "corpus.jsonl",
                    "sequence_number": 1,
                    "status": "keep"
                },
                "sequence_number": 1,
                "embedding_positions": 44
            }
        }
    )
