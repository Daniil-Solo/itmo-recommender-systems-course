from src.exceptions import ModelNotFoundError
from src.recommenders.base_recommender import BaseRecommender
from src.recommenders.random_recommender import RandomRecommender


class RecommenderRegister:  # pylint: disable=too-few-public-methods
    """
    Реестр рекомендательных систем
    """

    @staticmethod
    def get_recommender_by_model_name(model_name: str) -> BaseRecommender:
        """

        :param model_name:
        :return:
        """
        if model_name == "random":
            return RandomRecommender()
        raise ModelNotFoundError()
