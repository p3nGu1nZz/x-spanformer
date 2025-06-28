from typing import Optional
from pydantic import BaseModel, Field


class EntropyProfile(BaseModel):
    token_entropy: Optional[float] = Field(None, ge=0.0, description="Estimated per-token entropy from a language model or span scorer")
    span_overlap: Optional[float] = Field(None, ge=0.0, description="Average number of overlapping spans per token (structure density)")
    structure_variance: Optional[float] = Field(None, ge=0.0, description="Entropy or deviation of span lengths or roles")
    fluency_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional fluency/confidence score from a validator or model")

    class Config:
        schema_extra = {
            "example": {
                "token_entropy": 2.34,
                "span_overlap": 1.8,
                "structure_variance": 0.42,
                "fluency_score": 0.91
            }
        }