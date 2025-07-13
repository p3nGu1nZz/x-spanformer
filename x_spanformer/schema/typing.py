from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class RecordType(BaseModel):
    type: Optional[Literal["code", "natural", "mixed"]] = Field(
        None,
        description="Domain type of the record: code, natural language, or mixed"
    )

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "type": "mixed"
            }
        }
    )
