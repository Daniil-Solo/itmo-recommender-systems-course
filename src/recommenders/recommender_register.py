from src.exceptions import ModelNotFoundError
from src.recommenders.base_recommender import BaseRecommender
from src.recommenders.random_recommender import RandomRecommender
from src.recommenders.user_knn_recommender import UserKnnTFIDFRecommender
from src.recommenders.lightfm_recommender import LightFMRecommender
from src.recommenders.hnsw_lightfm_recommender import HNSWLightFMRecommender
from src.recommenders.autoencoder_recommender import AutoEncoderRecommender
from src.recommenders.dssm_recommender import DSSMRecommender
from src.recommenders.two_stage_recommender import TwoStageRecommender


RECOMMENDERS: list[type[BaseRecommender]] = [
    RandomRecommender,
    UserKnnTFIDFRecommender,
    LightFMRecommender,
    HNSWLightFMRecommender,
    AutoEncoderRecommender,
    DSSMRecommender,
    TwoStageRecommender,
]


class RecommenderRegister:  # pylint: disable=too-few-public-methods
    """
    Registers of recommendation systems
    """

    @staticmethod
    def get_recommender_by_model_name(model_name: str) -> BaseRecommender:
        """
        Returns recommender by model_name
        """
        for recommender in RECOMMENDERS:
            if model_name == recommender.MODEL_NAME:
                return recommender()
        raise ModelNotFoundError()
