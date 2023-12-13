from abc import ABC, abstractmethod


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
