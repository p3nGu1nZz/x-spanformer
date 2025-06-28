from typing import Optional, List
from pydantic import BaseModel, Field


class RecordMeta(BaseModel):
    page_number: Optional[int] = Field(None, description="Page number from the source document")
    tags: Optional[List[str]] = Field(default_factory=list, description="Arbitrary tags (e.g. domain, quality, source)")
    doc_language: Optional[str] = Field(None, description="ISO language code (e.g. 'en', 'ja')")
    extracted_by: Optional[str] = Field(None, description="Name of the tool or pipeline that produced this segment")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional confidence or fluency estimate")

    class Config:
        schema_extra = {
            "example": {
                "page_number": 12,
                "tags": ["math", "multilingual", "noisy"],
                "doc_language": "en",
                "extracted_by": "pdf2seg v0.3.1",
                "confidence": 0.87
            }
        }