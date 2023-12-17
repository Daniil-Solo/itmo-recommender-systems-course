import faiss
import numpy as np
from src.recommenders.base_recommender import FilterViewedAndPopularRecommender


DIR_PATH = "src/recommenders/artefacts"
ITEM_INV_MAPPING_PATH = f"{DIR_PATH}/lightfm_item_inv_mappings.json"
USER_MAPPING_PATH = f"{DIR_PATH}/lightfm_user_mappings.json"
ITEM_EMBEDDING_PATH = f"{DIR_PATH}/hnsw_item_embeddings.npy"
USER_EMBEDDING_PATH = f"{DIR_PATH}/hnsw_user_embeddings.npy"
USER_ITEM_MATRIX_PATH = f"{DIR_PATH}/user_item_matrix.npz"
TOP_ITEMS_PATH = f"{DIR_PATH}/top_items.json"

D = 258 + 1
M = 48
EF_SEARCH = 16
EF_CONSTRUCTION = 64


class HNSWLightFMRecommender(FilterViewedAndPopularRecommender):
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
        with open(USER_EMBEDDING_PATH, "rb") as f:
            self.user_embeddings = np.load(f)
        with open(ITEM_EMBEDDING_PATH, "rb") as f:
            item_embeddings = np.load(f)
        self.index = faiss.IndexHNSWFlat(D, M)
        self.index.hnsw.efConstruction = EF_CONSTRUCTION
        self.index.hnsw.efSearch = EF_SEARCH
        self.index.add(item_embeddings)

    def get_recommendations(self, user_id: int, k: int) -> list[int]:
        """
        Gets recommendations using HNSW-index on LightFM-embeddings
        """
        user_vector = self.user_embeddings[self.user_mappings[str(user_id)]].reshape(1, -1)
        _, indexes = self.index.search(user_vector, k)
        item_id_list = [self.item_inv_mappings[str(ind)] for ind in indexes.reshape(-1)]
        return item_id_list
