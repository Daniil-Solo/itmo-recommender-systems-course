from collections import Counter

import pandas as pd
import numpy as np
from rectools.dataset import Dataset
from rectools.models import PopularModel
from rectools.columns import Columns
from scipy.sparse import coo_matrix, spmatrix
from implicit.nearest_neighbours import ItemItemRecommender


class UserKnn:
    """
    Class for fit-predict UserKNN model
    based on ItemItemRecommender model from implicit.nearest_neighbours
    features:
    - fixed count of recommendations
    - the most interacted items for cold users
    """

    SIMILAR_USER_COLUMN = "similar_user_id"
    SIMILARITY_COLUMN = "similarity"
    IDF_COLUMN = "idf"

    def __init__(self, model: ItemItemRecommender, n_similar_users: int):
        self.model = model
        self.n_similar_users = n_similar_users

        self.users_inv_mapping = None
        self.users_mapping = None
        self.items_inv_mapping = None
        self.items_mapping = None

        self.interacted_items_dataframe = None
        self.item_idf = None
        self.top_items = None

    def _set_mappings(self, dataframe: pd.DataFrame) -> None:
        """
        Set mappings for users and items on train dataframe
        """
        self.users_inv_mapping = dict(enumerate(dataframe[Columns.User].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}
        self.items_inv_mapping = dict(enumerate(dataframe[Columns.Item].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def _get_item_user_matrix(self, dataframe: pd.DataFrame) -> spmatrix:
        """
        Gets sparse Item-User-matrix in CSR format
        """
        item_user_matrix = coo_matrix(
            (
                dataframe[Columns.Weight].astype(np.float32),
                (
                    dataframe[Columns.Item].map(self.items_mapping.get),
                    dataframe[Columns.User].map(self.users_mapping.get),
                ),
            )
        )
        return item_user_matrix.tocsr()

    def _set_interacted_items_dataframe(self, dataframe: pd.DataFrame) -> None:
        """
        Groups interactions by user to get item_id list for each user
        """
        self.interacted_items_dataframe = (
            dataframe.groupby(Columns.User, as_index=False)
            .agg({Columns.Item: list})
            .rename(columns={Columns.User: self.SIMILAR_USER_COLUMN})
        )

    @staticmethod
    def idf(n: int, x: float):
        """
        Calculates IDF for one item
        """
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, dataframe: pd.DataFrame):
        """
        Calculates IDF for all items and save into self.item_idf
        """
        item_counter = Counter(dataframe["item_id"].values)
        self.item_idf = pd.DataFrame.from_dict(
            item_counter, orient="index", columns=["doc_freq"]
        ).reset_index()
        self.item_idf[self.IDF_COLUMN] = self.item_idf["doc_freq"].apply(
            lambda x: self.idf(len(dataframe), x)
        )

    def _prepare_for_model(self, train_dataframe: pd.DataFrame) -> None:
        """
        Sets mappings, grouped interactions, calculates idf
        """
        self._set_mappings(train_dataframe)
        self._set_interacted_items_dataframe(train_dataframe)
        self._count_item_idf(train_dataframe)

    def _prepare_cold_recommendations(self, train_dataframe: pd.DataFrame, user_id: int) -> None:
        """
        Trains PopularModel and saves recommendations for cold users into self.top_items
        """
        model = PopularModel(popularity="n_interactions")
        dataset = Dataset.construct(train_dataframe)
        model.fit(dataset)
        reco_dataframe = model.recommend([user_id], dataset, k=100, filter_viewed=False)
        self.top_items = reco_dataframe[Columns.Item].values

    def fit(self, train_dataframe: pd.DataFrame) -> None:
        """
        Prepares and fits model
        """
        self._prepare_for_model(train_dataframe)
        self._prepare_cold_recommendations(train_dataframe, train_dataframe[Columns.User].iloc[0])

        item_user_matrix = self._get_item_user_matrix(train_dataframe)
        self.model.fit(item_user_matrix)

    def _get_similar_users(self, external_user_id: int) -> tuple[list[int], list[float]]:
        """
        Gets external user_id list of similar users
        Negative similarity is used to make the score
        lower than cold score
        """
        if external_user_id not in self.users_mapping:
            return [-1], [-1]
        internal_user_id = self.users_mapping[external_user_id]
        users, similarities = self.model.similar_items(internal_user_id, N=self.n_similar_users)
        return [self.users_inv_mapping[user] for user in users], similarities

    def _get_cold_user_recommendations(self, users: np.array, n_recs: int) -> pd.DataFrame:
        """
        Adds recommendation items for each user
        Cold recommendation score equal 0 because
        it should be lower than the knn score
        """
        data = []
        for user_id in users:
            for item_id in self.top_items[:n_recs]:
                data.append({Columns.User: user_id, Columns.Item: item_id, Columns.Score: 0})
        return pd.DataFrame(data)

    @staticmethod
    def finalize_recommendations(recs: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Adds Rank and cut k recommendations
        """
        recs = recs.sort_values([Columns.User, Columns.Score], ascending=False)
        recs[Columns.Rank] = recs.groupby(Columns.User).cumcount() + 1
        recs = recs[recs[Columns.Rank] <= k][
            [Columns.User, Columns.Item, Columns.Score, Columns.Rank]
        ]
        return recs

    def recommend(self, users: np.array, k: int) -> pd.DataFrame:
        """
        Gets recommendations dataframe
        """

        recs = pd.DataFrame({Columns.User: users})
        recs[self.SIMILAR_USER_COLUMN], recs[self.SIMILARITY_COLUMN] = zip(
            *recs[Columns.User].map(lambda user_id: self._get_similar_users(user_id))
        )
        recs = recs.set_index(Columns.User).apply(pd.Series.explode).reset_index()

        recs = (
            recs[~(recs[Columns.User] == recs[self.SIMILAR_USER_COLUMN])]
            .merge(
                self.interacted_items_dataframe,
                on=[self.SIMILAR_USER_COLUMN],
                how="left",
            )
            .explode(Columns.Item)
            .sort_values([Columns.User, self.SIMILARITY_COLUMN], ascending=False)
            .drop_duplicates([Columns.User, Columns.Item], keep="first")
            .merge(self.item_idf, left_on=Columns.Item, right_on="index", how="left")
        )

        recs[Columns.Score] = recs[self.SIMILARITY_COLUMN] * recs[self.IDF_COLUMN]
        recs = recs[[Columns.User, Columns.Item, Columns.Score]]
        recs = pd.concat([recs, self._get_cold_user_recommendations(users, k)])
        recs = self.finalize_recommendations(recs, k=k)
        return recs
