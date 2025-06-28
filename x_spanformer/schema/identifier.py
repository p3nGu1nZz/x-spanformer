from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import uuid


class RecordID(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique identifier for the record"
    )

    model_config = ConfigDict(
        frozen=True,  # Makes RecordID hashable and immutable
        json_schema_extra={
            "example": {
                "id": "3f9e6c50-8b8f-4be0-a3c0-90d9a1c3e691"
            }
        }
    )