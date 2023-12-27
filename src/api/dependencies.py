from fastapi import Path

from src.exceptions import UserNotFoundError
from src.recommenders.base_recommender import BaseRecommender
from src.recommenders.two_stage_recommender import TwoStageRecommender


def get_user_id(user_id: int = Path()) -> int:
    """
    Extracts user ID from the path
    Checks for the correctness of the value
    """
    if user_id > 10**9:
        raise UserNotFoundError()
    return user_id


def get_actual_recommender() -> BaseRecommender:
    """
    Gets actual recommender (for bot ddos)
    """
    return TwoStageRecommender()
