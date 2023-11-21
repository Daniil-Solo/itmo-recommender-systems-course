from random import sample

from src.config import settings
from src.recommenders.base_recommender import BaseRecommender


# pylint: disable=too-few-public-methods
class RandomRecommender(BaseRecommender):
    """
    Recommendation model with random prediction of items
    """

    N_MAX_ITEMS = 100
    MODEL_NAME = "random"

    def predict(self, user_id: int) -> list[int]:
        """
        Returns a random list of item IDs recommended
        to the user with id 'user_id'
        """
        all_items = list(range(RandomRecommender.N_MAX_ITEMS))
        items = sample(all_items, settings.n_returned_items)
        return sorted(items)
