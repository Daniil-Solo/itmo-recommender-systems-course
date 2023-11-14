from fastapi import APIRouter, Depends, Header, Path

from src.api.auth import check_authorization_token
from src.api.dependencies import get_user_id
from src.api.schemas import (ErrorMessage, HealthCheckResponse,
                             UserRecommendationResponse)
from src.recommenders.recommender_register import RecommenderRegister

router = APIRouter()


@router.get("/health", tags=["Health"], response_model=HealthCheckResponse)
async def health_check():
    """
    Проверяет, работает ли сервис
    """
    return HealthCheckResponse()


@router.get(
    "/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=UserRecommendationResponse,
    responses={401: {"model": ErrorMessage}, 404: {"model": ErrorMessage}},
)
async def get_recommendations(
    model_name: str = Path(),
    user_id: int = Depends(get_user_id),
    authorization: str = Header(None),
):
    """
    Формирует рекомендации модели model_name для пользователя с user_id
    """
    check_authorization_token(authorization)
    recommender = RecommenderRegister.get_recommender_by_model_name(model_name)
    items = recommender.predict(user_id)
    return UserRecommendationResponse(user_id=user_id, items=items)
