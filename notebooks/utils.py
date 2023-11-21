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

    def __init__(
        self,
        models: Dict[str, ModelBase],
        metrics: Dict[str, MetricAtK],
        splitter: Splitter,
        k: int,
    ) -> None:
        self._models = models
        self._metrics = metrics
        self._splitter = splitter
        self._k_recommendations = k

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

    def generate_report(
        self, interactions: Interactions, show_logs: bool = False
    ) -> DataFrame:
        """
        Runs training, testing initialized models
        and calculating metrics on k
        Returns pandas.DataFrame with results aggregated by folds
        """
        data = []
        fold_iter = self._splitter.split(interactions, collect_fold_stats=True)

        for train_ids, test_ids, fold_info in fold_iter:
            train_df = interactions.df.iloc[train_ids]
            dataset = Dataset.construct(train_df)
            test_df = interactions.df.iloc[test_ids][Columns.UserItem]
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
