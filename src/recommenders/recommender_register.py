from src.exceptions import ModelNotFoundError
from src.recommenders.base_recommender import BaseRecommender
from src.recommenders.random_recommender import RandomRecommender
from src.recommenders.user_knn_recommender import UserKnnTFIDFRecommender

RECOMMENDERS: list[type[BaseRecommender]] = [RandomRecommender, UserKnnTFIDFRecommender]


class RecommenderRegister:  # pylint: disable=too-few-public-methods
    """
    Registers of recommendation systems
    """

    @staticmethod
    def get_recommender_by_model_name(model_name: str) -> BaseRecommender:
        """
        Returns recommender by model_name
        """
        for recommender in RECOMMENDERS:
            if model_name == recommender.MODEL_NAME:
                return recommender()
        raise ModelNotFoundError()
