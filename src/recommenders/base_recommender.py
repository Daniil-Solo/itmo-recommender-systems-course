import json
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse

from src.config import settings


class BaseRecommender(ABC):  # pylint: disable=too-few-public-methods
    """
    The base class for the recommendation system
    """

    MODEL_NAME = ""

    @abstractmethod
    def predict(self, user_id: int) -> list[int]:
        """
        Returns a list of item IDs recommended to the user
        with id 'user_id'
        """
        raise NotImplementedError("This method should be implemented in a derived class")


class FilterViewedAndPopularRecommender(BaseRecommender):
    """
    Base class with filtering and adding items
    Filtering by viewed items
    Adding popular items
    """

    def __init__(self):
        self.top_items = None
        self.user_item_matrix = None
        self.item_inv_mappings = None
        self.user_mappings = None

    def load_artefacts(
        self,
        user_item_matrix_path: str,
        top_items_path: str,
        user_mappings_path: str,
        item_inv_mappings_path: str,
    ) -> None:
        """
        Loads user-item matrix, top items, user and item mappings
        """
        with open(top_items_path, "r", encoding="utf-8") as f:
            self.top_items = json.load(f)["items"]
        self.user_item_matrix = sparse.load_npz(user_item_matrix_path)
        with open(item_inv_mappings_path, "r", encoding="utf-8") as f:
            self.item_inv_mappings = json.load(f)
        with open(user_mappings_path, "r", encoding="utf-8") as f:
            self.user_mappings = json.load(f)

    def predict(self, user_id: int) -> list[int]:
        """
        Gets recommendations with filtering interacted items and
        adding popular items
        """
        recommended_items = []
        try:
            internal_user_id = self.user_mappings[str(user_id)]
            viewed_items = self.get_viewed_items(internal_user_id)
            recommended_items = self.get_recommendations(
                internal_user_id, settings.n_returned_items + len(viewed_items)
            )
            recommended_items = [
                item_id for item_id in recommended_items if item_id not in viewed_items
            ]
        except KeyError:
            pass
        if len(recommended_items) < settings.n_returned_items:
            recommended_items = self.add_popular_items(recommended_items)
        return recommended_items[: settings.n_returned_items]

    def get_viewed_items(self, internal_user_id: int) -> set[int]:
        """
        Gets viewed items
        """
        viewed_items = np.arange(len(self.item_inv_mappings))[
            self.user_item_matrix[internal_user_id, :].toarray().reshape(-1) > 0
        ]
        return {self.item_inv_mappings[str(item_id)] for item_id in viewed_items}

    def add_popular_items(self, recommended_items: list[int]) -> list[int]:
        """
        Add new popular items for recommendations
        """
        recommended_items.extend(
            [item_id for item_id in self.top_items if item_id not in recommended_items]
        )
        return recommended_items[: settings.n_returned_items]

    @abstractmethod
    def get_recommendations(self, user_id: int, k: int) -> list[int]:
        """
        Gets recommendations
        """
        raise NotImplementedError("This method should be implemented in a derived class")
