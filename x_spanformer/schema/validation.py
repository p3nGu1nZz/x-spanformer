from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ValidationIssue(BaseModel):
    field: str = Field(..., description="Field where the issue occurred (e.g., 'span_labels', 'input')")
    message: str = Field(..., description="Human-readable description of the validation issue")
    severity: str = Field(..., description="Severity level: 'error', 'warning', or 'info'")


class ValidationResult(BaseModel):
    is_valid: bool = Field(..., description="True if the record passed all validation checks")
    issues: List[ValidationIssue] = Field(default_factory=list, description="List of validation issues found")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "is_valid": False,
                "issues": [
                    {
                        "field": "span_labels[2]",
                        "message": "Span index [14, 16] is out of bounds for input of length 15",
                        "severity": "error"
                    },
                    {
                        "field": "text",
                        "message": "Span text does not match input slice",
                        "severity": "warning"
                    }
                ]
            }
        }
    )