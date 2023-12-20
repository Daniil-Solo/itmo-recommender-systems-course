import json

import faiss
import numpy as np
from src.recommenders.base_recommender import FilterViewedAndPopularRecommender

# pylint: disable=duplicate-code
DIR_PATH = "src/recommenders/artefacts"
ITEM_INV_MAPPING_PATH = f"{DIR_PATH}/lightfm_item_inv_mappings.json"
USER_MAPPING_PATH = f"{DIR_PATH}/lightfm_user_mappings.json"
USER_ITEM_MATRIX_PATH = f"{DIR_PATH}/user_item_matrix.npz"
TOP_ITEMS_PATH = f"{DIR_PATH}/top_items.json"
# pylint: disable=duplicate-code
ITEM_EMBEDDING_PATH = f"{DIR_PATH}/dssm_item_embeddings.npy"
USER_EMBEDDING_PATH = f"{DIR_PATH}/dssm_user_embeddings.npy"
MODEL_USER_MAPPING_PATH = f"{DIR_PATH}/dssm_user_mappings.json"
MODEL_ITEM_INV_MAPPING_PATH = f"{DIR_PATH}/dssm_item_inv_mappings.json"

N_FACTORS = 256


class DSSMRecommender(FilterViewedAndPopularRecommender):
    """
    The base class for the recommendation system
    """

    MODEL_NAME = "hnsw_lightfm"

    # pylint: disable=duplicate-code, no-value-for-parameter
    def __init__(self):
        super().__init__()
        self.load_artefacts(
            USER_ITEM_MATRIX_PATH,
            TOP_ITEMS_PATH,
            USER_MAPPING_PATH,
            ITEM_INV_MAPPING_PATH,
        )
        self.inv_user_mappings = {str(v): int(k) for (k, v) in self.user_mappings.items()}

        with open(MODEL_ITEM_INV_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.model_item_inv_mappings = json.load(f)
        with open(MODEL_USER_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.model_user_mappings = json.load(f)
        with open(USER_EMBEDDING_PATH, "rb") as f:
            self.user_embeddings = np.load(f)
        with open(ITEM_EMBEDDING_PATH, "rb") as f:
            item_embeddings = np.load(f)
        self.index = faiss.IndexFlatL2(N_FACTORS)
        self.index.add(item_embeddings)

    def get_recommendations(self, user_id: int, k: int) -> list[int]:
        """
        Gets recommendations using Flat-index on DSSM-embeddings
        """
        external_user_id = self.inv_user_mappings[str(user_id)]
        internal_user_id = self.model_user_mappings[str(external_user_id)]

        user_vector = self.user_embeddings[internal_user_id, :].reshape(1, -1)
        _, indexes = self.index.search(user_vector, k)
        item_id_list = [self.model_item_inv_mappings[str(ind)] for ind in indexes.reshape(-1)]
        return item_id_list
