from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

class RecordMeta(BaseModel):
    tags: List[str] = Field(default_factory=list, description="Arbitrary tags (e.g. domain, quality, source)")
    doc_language: Optional[str] = Field(None, description="ISO language code (e.g. 'en', 'ja')")
    extracted_by: Optional[str] = Field(None, description="Name of the tool or pipeline that produced this segment")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional confidence or fluency estimate")
    source_file: Optional[str] = Field(None, description="Original filename or source document ID (e.g. from adjacent metadata file)")
    notes: Optional[str] = Field(None, description="Optional textual explanation or scoring justification (e.g. from selfcrit)")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "tags": ["math", "multilingual", "noisy"],
                "doc_language": "en",
                "extracted_by": "pdf2seg v0.3.1",
                "confidence": 0.87,
                "source_file": "mycelial-patterns-2025.pdf",
                "notes": "short segment; missing verb — marked 'revise'"
            }
        }
    )