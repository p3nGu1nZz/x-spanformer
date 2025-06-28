from typing import List, Optional
from pydantic import BaseModel, Field


class TokenizedInput(BaseModel):
    input: List[str] = Field(..., description="Tokenized version of the raw string, including spaces and punctuation")
    tokenizer: Optional[str] = Field(None, description="Name or version of the tokenizer used (e.g., 'whitespace', 'byte-level')")
    preserve_whitespace: Optional[bool] = Field(default=True, description="Whether spacing and formatting were preserved during tokenization")

    class Config:
        schema_extra = {
            "example": {
                "input": ["The", " ", "quick", " ", "brown", " ", "fox", "."],
                "tokenizer": "whitespace",
                "preserve_whitespace": True
            }
        }