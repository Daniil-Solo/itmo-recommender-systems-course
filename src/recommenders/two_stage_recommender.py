import json
from src.recommenders.base_recommender import BaseRecommender


DIR_PATH = "src/recommenders/artefacts"
OFFLINE_RECS_PATH = f"{DIR_PATH}/predicts.json"


# pylint: disable=too-few-public-methods
class TwoStageRecommender(BaseRecommender):
    """
    Recommendation model with random prediction of items
    """

    N_MAX_ITEMS = 100
    MODEL_NAME = "two_stages"

    def __init__(self):
        super().__init__()
        with open(OFFLINE_RECS_PATH, "r", encoding="utf-8") as f:
            self.predicts = json.load(f)

    def predict(self, user_id: int) -> list[int]:
        """
        Returns a random list of item IDs recommended
        to the user with id 'user_id'
        """
        try:
            items = self.predicts[str(user_id)]
        except KeyError:
            items = self.predicts[str(50000)]
        return items
