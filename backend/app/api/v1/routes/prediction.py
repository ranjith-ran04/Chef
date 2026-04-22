# backend/app/api/v1/routes/prediction.py

from fastapi import APIRouter, HTTPException, status, Depends # type: ignore
from app.core.schemas.prediction_schema import PredictionRequest, PredictionResponse, ErrorResponse
from app.services.prediction_service import PredictionService, prediction_service
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
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict dish + recipe from category and ingredients",
)
async def predict_dish_endpoint(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        return service.get_dish_prediction(
            category=request.category,
            ingredients=request.ingredients,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="ML model unavailable. Please train first.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Unexpected error. Please try again.")