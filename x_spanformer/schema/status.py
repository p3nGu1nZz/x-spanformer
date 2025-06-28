from typing import List, Literal
from pydantic import BaseModel, Field


class RecordStatus(BaseModel):
    stages: List[
        Literal[
            "csv_ingested",
            "tokenized",
            "typed",
            "scored",
            "xbar_labeled",
            "span_labels_attached",
            "validated",
            "ready_for_phase1",
            "phase1_complete"
        ]
    ] = Field(default_factory=list, description="Ordered log of completed preprocessing/enrichment stages")

    class Config:
        json_schema_extra = {
            "example": {
                "stages": [
                    "csv_ingested",
                    "tokenized",
                    "scored",
                    "span_labels_attached"
                ]
            }
        }