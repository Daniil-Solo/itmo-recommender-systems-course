from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    """
    Схема ответа для проверки работы сервиса
    """

    status: str = "ok"


class UserRecommendationResponse(BaseModel):
    """
    Схема ответа рекомендаций для пользователя
    """

    user_id: int
    items: list[int]


class ErrorMessage(BaseModel):
    """
    Схема ответа при ошибке
    """

    message: str
