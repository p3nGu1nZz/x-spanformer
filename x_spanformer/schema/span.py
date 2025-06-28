from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict


class SpanLabel(BaseModel):
    span: Tuple[int, int] = Field(..., description="Inclusive start and end token indices")
    label: str = Field(..., description="Syntactic or semantic category (e.g., 'noun', 'keyword')")
    role: Optional[str] = Field(None, description="Functional role within context (e.g., 'subject', 'assignment')")
    text: Optional[str] = Field(None, description="Span text (redundant with input[span[0]:span[1]+1], but useful for validation/debug)")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "span": [4, 4],
                "label": "noun",
                "role": "subject",
                "text": "fox"
            }
        }
    )
