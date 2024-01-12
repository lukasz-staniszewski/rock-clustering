from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Set

import numpy as np
import pandas as pd
from datasets import ClusteringDataset
from sklearn.model_selection import train_test_split


@dataclass
class ClusteringPoint:
    original_idx: int  # identifies point in original dataset
    idx: int  # identifies point in a new dataset
    transaction: Set[str]  # binary vector of features
    y: Optional[Any] = None  # target value
    output_cluster_idx: Optional[int] = None  # output cluster index


@dataclass
class RockInput:
    data_train: List[ClusteringPoint]
    data_test: List[ClusteringPoint] | None = None


def get_rock_input(dataset: ClusteringDataset, split_train: float = 1.0) -> RockInput:
    """Splits the dataset into train and test sets and converts it to the RockInput format.
    Categories are converted to dummy variables.
    """
    dataset.data['transactions'] = dataset.data.apply(lambda row: set([f"{k}.{v}" for (k,v) in row.items() if pd.notna(v)]), axis=1)

    if split_train == 1.0:
        return RockInput(
            data_train=[
                ClusteringPoint(original_idx=idx, idx=idx, transaction=transaction, y=y)
                for idx, transaction, y in zip(
                    dataset.data.index.to_numpy(),
                    dataset.data.transactions.values,
                    dataset.target.values,
                )
            ]
        )

    idx_train, idx_test, y_train, y_test = train_test_split(
        dataset.data.index, dataset.target.values, train_size=split_train, random_state=42
    )

    return RockInput(
        data_train=[
            ClusteringPoint(original_idx=original_idx, idx=idx, transaction=transaction, y=y)
            for original_idx, idx, transaction, y in zip(
                idx_train.to_numpy(),
                idx_train.to_frame().reset_index().index.values,
                dataset.data.iloc[idx_train.to_list(), :].transactions.values,
                y_train,
            )
        ],
        data_test=[
            ClusteringPoint(original_idx=original_idx, idx=idx, transaction=transaction, y=y)
            for original_idx, idx, transaction, y in zip(
                idx_test.to_numpy(),
                idx_test.to_frame().reset_index().index.values,
                dataset.data.iloc[idx_test.to_list(), :].transactions.values,
                y_test,
            )
        ],
    )


def get_rock_output(dataset: ClusteringDataset, rock_output: List[ClusteringPoint]) -> pd.DataFrame:
    """Converts the output of the Rock algorithm to a DataFrame with the original dataset and the cluster index.
    Assigns None for the outliers.
    """
    rock_out = sorted(rock_output, key=lambda x: x.original_idx)
    rock_out = pd.Series([x.output_cluster_idx for x in rock_out], index=dataset.data.index)
    rock_out = pd.Series(pd.factorize(rock_out, use_na_sentinel=True)[0]).replace(-1, np.nan)
    df = pd.concat([dataset.data, dataset.target.rename("target"), rock_out.rename("cluster_idx")], axis=1)
    return df
