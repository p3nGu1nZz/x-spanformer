from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TokenizedInput(BaseModel):
    input: List[str] = Field(..., description="Tokenized version of the raw string, including spaces, punctuation, special characters")
    tokenizer: Optional[str] = Field(None, description="Name or version of the tokenizer used (e.g., 'oxbar', 'sentencepiece')")
    preserve_whitespace: Optional[bool] = Field(default=True, description="Whether spacing and formatting were preserved during tokenization")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "input": ["The", " ", "quick", " ", "brown", " ", "fox", "."],
                "tokenizer": "oxbar",
                "preserve_whitespace": True
            }
        }
    )