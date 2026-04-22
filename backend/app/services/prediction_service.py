# backend/app/services/prediction_service.py

from typing import List
from backend.app.ml.predict import predict_dish
from backend.app.core.schemas.prediction_schema import PredictionResponse, RecommendationItem
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    def get_dish_prediction(
        self, category: str, ingredients: List[str]
    ) -> PredictionResponse:
        """
        Calls the ML prediction logic, validates results,
        and returns a structured PredictionResponse.
        """
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

        return PredictionResponse(
            best_match=raw["best_match"],
            recommendations=recommendations,
            explanation=raw["explanation"],
        )


# Singleton instance for dependency injection
prediction_service = PredictionService()