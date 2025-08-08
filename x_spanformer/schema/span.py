from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict


class SpanLabel(BaseModel):
    span: Tuple[int, int] = Field(..., description="Inclusive start and end token indices")
    xbar_label: str = Field(..., description="X-bar label (e.g., 'noun', 'verb_phrase', 'keyword')")
    text: Optional[str] = Field(None, description="Span text (redundant with input[span[0]:span[1]+1], but useful for validation/debug)")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "span": [4, 4],
                "xbar_label": "noun",
                "text": "fox"
            }
        }
    )
