# backend/app/schemas/prediction_schema.py

from pydantic import BaseModel, Field, validator
from typing import List


class PredictionRequest(BaseModel):
    category: str = Field(..., min_length=1, max_length=100, example="indian")
    ingredients: List[str] = Field(
        ...,
        min_items=1,
        max_items=30,
        example=["paneer", "tomato", "cream", "butter"],
    )

    @validator("category")
    def category_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("category must not be blank")
        return v.strip().lower()

    @validator("ingredients", each_item=True)
    def clean_ingredient(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("ingredient must not be blank")
        return v.lower()


class RecommendationItem(BaseModel):
    dish: str
    confidence: float = Field(..., ge=0, le=100, description="Confidence percentage 0–100")


class PredictionResponse(BaseModel):
    best_match: str
    recommendations: List[RecommendationItem]
    explanation: str

    class Config:
        schema_extra = {
            "example": {
                "best_match": "paneer butter masala",
                "recommendations": [
                    {"dish": "paneer butter masala", "confidence": 91.0},
                    {"dish": "shahi paneer", "confidence": 83.0},
                    {"dish": "kadai paneer", "confidence": 78.0},
                ],
                "explanation": "Based on the indian category and key ingredients like paneer, tomato, cream, the model strongly predicts 'paneer butter masala' as the best match with a confidence of 91.0%.",
            }
        }


class ErrorResponse(BaseModel):
    detail: str