# backend/app/api/v1/routes/prediction.py

from fastapi import APIRouter, HTTPException, status, Depends
from backend.app.core.schemas.prediction_schema import PredictionRequest, PredictionResponse, ErrorResponse
from backend.app.services.prediction_service import PredictionService, prediction_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Dish Prediction"])


def get_prediction_service() -> PredictionService:
    return prediction_service


@router.post(
    "/predict-dish",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Predict a dish from category and ingredients",
    description=(
        "Accepts a food category and a list of ingredients, "
        "then returns the best-matched dish, top 3 recommendations "
        "with confidence scores, and a human-readable explanation."
    ),
)
async def predict_dish_endpoint(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        result = service.get_dish_prediction(
            category=request.category,
            ingredients=request.ingredients,
        )
        return result
    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not available. Please train the model first.",
        )
    except ValueError as e:
        logger.warning(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        )