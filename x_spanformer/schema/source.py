from typing import Optional
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    document_id: Optional[str] = Field(None, description="Canonical ID or hash of the source document")
    filename: Optional[str] = Field(None, description="Original source filename (e.g., my_paper.pdf)")
    filetype: Optional[str] = Field(None, description="File extension or type (e.g., 'pdf', 'md')")
    page_number: Optional[int] = Field(None, description="Page number the segment came from, if applicable")
    line_number: Optional[int] = Field(None, description="Line number or block index from source document")
    section: Optional[str] = Field(None, description="Optional section heading or document zone")
    source_url: Optional[str] = Field(None, description="URL or repo link to the document if public")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "d41d8cd98f00b204e9800998ecf8427e",
                "filename": "xbar_transformer_2024.pdf",
                "filetype": "pdf",
                "page_number": 15,
                "line_number": 42,
                "section": "3.2. Span Induction",
                "source_url": "https://github.com/p3nGu1nZz/x-spanformer/"
            }
        }