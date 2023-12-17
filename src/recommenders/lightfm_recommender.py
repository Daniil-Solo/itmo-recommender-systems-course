import pickle
import numpy as np
from src.recommenders.base_recommender import FilterViewedAndPopularRecommender


DIR_PATH = "src/recommenders/artefacts"
MODEL_PATH = f"{DIR_PATH}/lightfm.pickle"
ITEM_INV_MAPPING_PATH = f"{DIR_PATH}/lightfm_item_inv_mappings.json"
USER_MAPPING_PATH = f"{DIR_PATH}/lightfm_user_mappings.json"
USER_ITEM_MATRIX_PATH = f"{DIR_PATH}/user_item_matrix.npz"
TOP_ITEMS_PATH = f"{DIR_PATH}/top_items.json"


class LightFMRecommender(FilterViewedAndPopularRecommender):
    """
    The base class for the recommendation system
    """

    MODEL_NAME = "lightfm"

    # pylint: disable=duplicate-code
    def __init__(self):
        super().__init__()
        self.load_artefacts(
            USER_ITEM_MATRIX_PATH,
            TOP_ITEMS_PATH,
            USER_MAPPING_PATH,
            ITEM_INV_MAPPING_PATH,
        )
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

    def get_recommendations(self, user_id: int, k: int) -> list[int]:
        """
        Gets recommendations using LightFM
        """
        all_items = np.arange(len(self.item_inv_mappings))
        scores = self.model.predict(user_id, all_items)
        indexes = np.argpartition(scores, -k)[-k:][::-1]
        item_id_list = [self.item_inv_mappings[str(ind)] for ind in indexes]
        return item_id_list
