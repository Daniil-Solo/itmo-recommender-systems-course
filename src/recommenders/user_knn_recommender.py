import json
import numpy as np
from scipy.sparse import load_npz
from implicit.nearest_neighbours import TFIDFRecommender
from src.config import settings
from src.recommenders.base_recommender import BaseRecommender


MODEL_PATH = "src/recommenders/artefacts/user_knn.npz"
WEIGHTS_PATH = "src/recommenders/artefacts/matrix_for_user_knn.npz"
ITEM_INV_MAPPING_PATH = "src/recommenders/artefacts/items_inv_mapping.json"
USER_MAPPING_PATH = "src/recommenders/artefacts/users_mapping.json"
TOP_ITEMS_PATH = "src/recommenders/artefacts/top_items.json"
ITEM_IDF_PATH = "src/recommenders/artefacts/item_idfs.json"


class UserKnnTFIDFRecommender(BaseRecommender):
    """
    Recommendation model with random prediction of items
    """

    MODEL_NAME = "user_knn_tf_idf"

    def __init__(self):
        self.model = TFIDFRecommender.load(MODEL_PATH)
        self.weights = load_npz(WEIGHTS_PATH)
        with open(ITEM_INV_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.item_inv_mapping = json.load(f)
        with open(USER_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.user_mapping = json.load(f)
        with open(TOP_ITEMS_PATH, "r", encoding="utf-8") as f:
            self.top_items = set(json.load(f)["items"])
        with open(ITEM_IDF_PATH, "r", encoding="utf-8") as f:
            self.item_idfs = np.array(json.load(f)["items"])

    def predict(self, user_id: int) -> list[int]:
        """
        Returns list of item IDs recommended
        to the user with id 'user_id' using user knn
        """
        recommended_items = []
        try:
            internal_user_id = self.user_mapping[str(user_id)]
            recommended_items = self.get_user_knn_recommendations(
                internal_user_id, settings.n_returned_items
            )
        except (IndexError, TypeError, KeyError):
            pass
        finally:
            if len(recommended_items) < settings.n_returned_items:
                recommended_items = self.add_popular_items(recommended_items)
        return recommended_items

    def add_popular_items(self, recommended_items: np.array) -> list[int]:
        """
        Add new popular items for recommendations
        """
        recommended_set = set(recommended_items)
        recommended_set = recommended_set.union(self.top_items)
        return list(recommended_set)[: settings.n_returned_items]

    def get_user_knn_recommendations(self, user_id: int, k_recs: int):
        """
        Gets k_recs recommendations for user with user_id
        """
        # gets similar users
        user_ids, scores = self.model.similar_items(user_id, N=self.model.K)
        user_similarity = dict(zip(user_ids, scores))
        # matrix[user_id, item_id] = user_similarity
        all_item_ids = np.arange(self.item_idfs.shape[0])
        for current_user_id in user_ids:
            current_item_ids = all_item_ids[
                self.weights[:, current_user_id].toarray().reshape(-1) > 0
            ]
            self.weights[current_item_ids, current_user_id] = user_similarity[current_user_id]
        # max similarity on similar users by item_id
        all_item_maximums = np.max(self.weights[:, user_ids], axis=1).toarray().reshape(-1)
        # filtering items
        scores = all_item_maximums * self.item_idfs
        items = [
            (idx, maximum)
            for (idx, maximum) in zip(np.arange(scores.shape[0]), scores)
            if maximum > 0 and maximum != 1
        ]
        # sorting items
        items = sorted(items, key=lambda x: x[1], reverse=True)[:k_recs]
        return [
            self.item_inv_mapping[str(item[0])]
            for item in items
            if str(item[0]) in self.item_inv_mapping
        ]
