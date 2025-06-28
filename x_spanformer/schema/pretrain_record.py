from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from x_spanformer.schema.identifier import RecordID
from x_spanformer.schema.metadata import RecordMeta


class PretrainRecord(BaseModel):
    raw: str = Field(..., description="The raw text segment extracted from source material")
    id: Optional[RecordID] = Field(default_factory=RecordID, description="Globally unique record ID")
    meta: RecordMeta = Field(default_factory=lambda: RecordMeta(**{}), description="Metadata about the segment")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "raw": "The quick brown fox jumps over the lazy dog.",
                "id": {"id": "3d3e1e3e-8f6b-4a9a-9fc6-efedc5f805a8"},
                "meta": {
                    "tags": ["example", "english", "simple"],
                    "doc_language": "en",
                    "extracted_by": "pdf2seg v0.3.1",
                    "confidence": 0.95,
                    "source_file": "fox-example-2025.pdf"
                }
            }
        }
    )
