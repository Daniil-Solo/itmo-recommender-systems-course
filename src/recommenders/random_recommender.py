from random import sample

from src.config import settings
from src.recommenders.base_recommender import BaseRecommender


# pylint: disable=too-few-public-methods
class RandomRecommender(BaseRecommender):
    """
    Рекомендательная система со случайным предсказанием объектов
    """

    N_MAX_ITEMS = 100

    def predict(self, user_id: int) -> list[int]:
        """
        Возвращает список идентификатор объектов, рекомендуемых
        пользователю с идентификатором user_id
        """
        all_items = list(range(RandomRecommender.N_MAX_ITEMS))
        items = sample(all_items, settings.n_returned_items)
        return sorted(items)
