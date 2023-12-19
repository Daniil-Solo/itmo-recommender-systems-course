import json

import torch
from torch import nn
import numpy as np
from scipy import sparse

from src.recommenders.base_recommender import FilterViewedAndPopularRecommender


DIR_PATH = "src/recommenders/artefacts"
ITEM_INV_MAPPING_PATH = f"{DIR_PATH}/lightfm_item_inv_mappings.json"
USER_MAPPING_PATH = f"{DIR_PATH}/lightfm_user_mappings.json"
USER_ITEM_MATRIX_PATH = f"{DIR_PATH}/user_item_matrix.npz"
TOP_ITEMS_PATH = f"{DIR_PATH}/top_items.json"

MODEL_PATH = f"{DIR_PATH}/autoencoder.pth"
MODEL_USER_MAPPING_PATH = f"{DIR_PATH}/autoencoder_user_mappings.json"
MODEL_ITEM_INV_MAPPING_PATH = f"{DIR_PATH}/autoencoder_item_inv_mappings.json"
MODEL_USER_ITEM_MATRIX_PATH = f"{DIR_PATH}/autoencoder_user_item_matrix.npz"


class Model(nn.Module):
    def __init__(self, in_and_out_features):
        super().__init__()
        self.in_and_out_features = in_and_out_features
        self.hidden_size = 500

        self.sequential = nn.Sequential(
            nn.Linear(in_and_out_features, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, in_and_out_features, bias=True),
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


class AutoEncoderRecommender(FilterViewedAndPopularRecommender):
    """
    The base class for the recommendation system
    """

    MODEL_NAME = "auto_encoder"

    # pylint: disable=duplicate-code
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
        self.model_user_item_matrix = sparse.load_npz(MODEL_USER_ITEM_MATRIX_PATH)
        self.model = Model(len(self.model_item_inv_mappings))
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

    def get_recommendations(self, user_id: int, k: int) -> list[int]:
        """
        Gets recommendations using AutoEncoder
        user_id - internal_user_id in full matrix
        for autoencoder we need get internal_user_id in other matrix
        """
        external_user_id = self.inv_user_mappings[str(user_id)]
        internal_user_id = self.model_user_mappings[str(external_user_id)]

        user_interactions = self.model_user_item_matrix[internal_user_id, :].toarray()
        with torch.no_grad():
            predicted_scores = self.model(torch.Tensor(user_interactions))
        indexes = np.argpartition(predicted_scores.numpy().reshape(-1), -k)[-k:][::-1]
        item_id_list = [self.model_item_inv_mappings[str(ind)] for ind in indexes]
        return item_id_list
