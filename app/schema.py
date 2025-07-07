"""Pydantic schemas for the ML API."""
from pydantic import BaseModel, Field


class IrisInput(BaseModel):
    """Input schema for iris flower classification."""
    sepal_length: float = Field(..., gt=0, lt=10)
    sepal_width: float = Field(..., gt=0, lt=10)
    petal_length: float = Field(..., gt=0, lt=10)
    petal_width: float = Field(..., gt=0, lt=10)
