from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict

from x_spanformer.schema.span import SpanLabel


class TrainingRecord(BaseModel):
    """
    Training record format for X-Spanformer span induction training.
    
    This schema matches the format described in pretraining_schema.md and represents
    the raw tokenized input with span annotations for training the model to
    identify and predict meaningful spans.
    """
    input: List[str] = Field(..., description="Character-conscious tokenized sequence including spaces, punctuation, special characters")
    type: Literal["natural", "code", "mixed"] = Field(..., description="Domain type of the sequence")
    span_labels: List[SpanLabel] = Field(..., description="Span annotations with labels, roles, and text")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "input": ["The", " ", "quick", " ", "brown", " ", "fox", " ", "jumps", "."],
                "type": "natural",
                "span_labels": [
                    {
                        "span": [0, 0],
                        "label": "determiner",
                        "role": "noun specifier", 
                        "text": "The"
                    },
                    {
                        "span": [2, 2],
                        "label": "adjective",
                        "role": "modifier",
                        "text": "quick"
                    },
                    {
                        "span": [6, 6],
                        "label": "noun",
                        "role": "subject",
                        "text": "fox"
                    },
                    {
                        "span": [8, 8],
                        "label": "verb",
                        "role": "predicate",
                        "text": "jumps"
                    }
                ]
            }
        }
    )
