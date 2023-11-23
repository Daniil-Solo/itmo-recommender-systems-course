from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    """
    Response scheme for checking the health of the service
    """

    status: str = "ok"


class UserRecommendationResponse(BaseModel):
    """
    The scheme of the recommendations response for the user
    """

    user_id: int
    items: list[int]


class ErrorMessage(BaseModel):
    """
    Error response scheme
    """

    message: str
