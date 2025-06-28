from typing import Literal, Optional
from pydantic import BaseModel, Field


class RecordType(BaseModel):
    type: Optional[Literal["code", "natural_language", "mixed"]] = Field(
        None,
        description="Domain type of the record: code, natural language, or mixed"
    )

    class Config:
        schema_extra = {
            "example": {
                "type": "mixed"
            }
        }