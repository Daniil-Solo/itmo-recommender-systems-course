import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from rectools import Columns
from rectools.metrics import calc_metrics
from rectools.models.base import ModelBase
from rectools.metrics.base import MetricAtK
from rectools.model_selection.splitter import Splitter
from rectools.dataset import Interactions, Dataset


class MetricCalculator:
    """
    Class for training and testing recommenders
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        models: Dict[str, ModelBase],
        metrics: Dict[str, MetricAtK],
        splitter: Splitter,
        k: int,
        interactions: Interactions,
    ) -> None:
        self._models = models
        self._metrics = metrics
        self._splitter = splitter
        self._k_recommendations = k
        self._interactions = interactions

    @staticmethod
    def prepare_dataframe(data: List[Dict]) -> pd.DataFrame:
        """
        Transforms list of dictionaries with fold results to DataFrame
        """
        result_df = (
            pd.DataFrame(data)
            .drop(columns="fold")
            .groupby(["model"], sort=False)
            .agg(["mean"])
        )
        result_df.columns = [v[0] for v in result_df.columns]
        return result_df

    def generate_report(  # pylint: disable=too-many-locals
        self, show_logs: bool = False
    ) -> DataFrame:
        """
        Runs training, testing initialized models
        and calculating metrics on k
        Returns pandas.DataFrame with results aggregated by folds
        """
        data = []
        fold_iterator = self._splitter.split(
            self._interactions, collect_fold_stats=True
        )

        for train_ids, test_ids, fold_info in fold_iterator:
            train_df = self._interactions.df.iloc[train_ids]
            dataset = Dataset.construct(train_df)
            test_df = self._interactions.df.iloc[test_ids][Columns.UserItem]
            test_users = np.unique(test_df[Columns.User])
            catalog = train_df[Columns.Item].unique()

            for model_name, model in self._models.items():
                fit_start_time = datetime.datetime.now()
                model.fit(dataset)
                fit_end_time = datetime.datetime.now()
                recommendations = model.recommend(
                    users=test_users,
                    dataset=dataset,
                    k=self._k_recommendations,
                    filter_viewed=True,
                )
                reco_end_time = datetime.datetime.now()
                metric_values = calc_metrics(
                    self._metrics,
                    reco=recommendations,
                    interactions=test_df,
                    prev_interactions=train_df,
                    catalog=catalog,
                )
                item = {"fold": fold_info["i_split"], "model": model_name}
                item.update(metric_values)
                data.append(item)
                if not show_logs:
                    continue
                print(
                    f'Test fold: {fold_info["i_split"] + 1}, '
                    f"model: {model_name}, fitted in: "
                    f"{(fit_end_time - fit_start_time).microseconds // 1000} "
                    f"milliseconds, predicted in: "
                    f"{(reco_end_time - fit_end_time).microseconds // 1000} "
                    f"milliseconds"
                )

        return self.prepare_dataframe(data)


class VisualAnalyzer:
    """
    Class for analysis of history and recommendations items
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: ModelBase,
        dataset: Dataset,
        user_id_list: List[int],
        item_data: List[str],
        k: int,
        items_df: pd.DataFrame,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._user_id_list = user_id_list
        if "item_id" not in item_data:
            item_data.insert(0, "item_id")
        self._item_data = item_data
        self._k_recommendations = k
        self._items_df = items_df

    def get_recommendations(self):
        """
        Gets recommendations
        """
        recommendations = self._model.recommend(
            users=self._user_id_list,
            dataset=self._dataset,
            k=self._k_recommendations,
            filter_viewed=True,
        )
        return recommendations

    def get_history_and_recommendation_dataframes(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gets dataframes with interactions and items for
        history data and recommendations
        """
        recommendations = self.get_recommendations()
        history_df = (
            self._dataset.interactions.df[
                self._dataset.interactions.df.user_id.isin(self._user_id_list)
            ]
            .merge(self._items_df[self._item_data], on="item_id")
            .sort_values("user_id")
        )
        reco_df = recommendations.merge(
            self._items_df[self._item_data], on="item_id"
        ).sort_values(["user_id", "rank"])
        return history_df, reco_df
