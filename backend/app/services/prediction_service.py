# backend/app/services/prediction_service.py

from typing import List
from app.ml.predict import predict_dish
from app.core.schemas.prediction_schema import PredictionResponse, RecommendationItem, RecipeResponse
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    def get_dish_prediction(self, category: str, ingredients: List[str]) -> PredictionResponse:
        try:
            raw = predict_dish(category=category, ingredients=ingredients)
        except RuntimeError as e:
            logger.error(f"ML model not loaded: {e}")
            raise
        except ValueError as e:
            logger.warning(f"Prediction returned no results: {e}")
            raise

        recommendations = [
            RecommendationItem(dish=r["dish"], confidence=r["confidence"])
            for r in raw.get("recommendations", [])
        ]

        recipe_raw = raw.get("recipe", {"ingredients": [], "steps": []})
        recipe = RecipeResponse(
            ingredients=recipe_raw.get("ingredients", []),
            steps=recipe_raw.get("steps", []),
        )

        return PredictionResponse(
            best_match=raw["best_match"],
            recommendations=recommendations,
            recipe=recipe,
            explanation=raw["explanation"],
        )


prediction_service = PredictionService()