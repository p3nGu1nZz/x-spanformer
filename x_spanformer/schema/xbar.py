from typing import Literal, Optional, Tuple
from pydantic import BaseModel, Field


class XPSpan(BaseModel):
    span: Tuple[int, int] = Field(..., description="Start and end token indices for the X-bar phrase")
    category: Literal["X⁰", "X′", "XP"] = Field(..., description="Constituent category (head, intermediate, phrase)")
    role: Optional[
        Literal[
            "specifier",
            "complement",
            "adjunct",
            "head"
        ]
    ] = Field(None, description="Constituent role relative to the head")
    label: Optional[str] = Field(None, description="Optional syntactic label (e.g., 'NP', 'VP', 'PP')")
    text: Optional[str] = Field(None, description="Human-readable span content (optional, for debugging)")

    class Config:
        json_schema_extra = {
            "example": {
                "span": [2, 5],
                "category": "XP",
                "role": "specifier",
                "label": "NP",
                "text": "the quick fox"
            }
        }