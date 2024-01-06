from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from rock.data import ClusteringDataset


@dataclass
class ClusteringPoint:
    idx: int  # identifies point in dataset
    x: np.ndarray[bool]  # binary vector of features
    y: Optional[Any] = None  # target value


@dataclass
class RockInput:
    data: List[ClusteringPoint]


def get_rock_input(dataset: ClusteringDataset) -> RockInput:
    dataset_data_dummy = pd.get_dummies(dataset.data)
    return RockInput(
        data=[
            ClusteringPoint(idx=idx, x=x, y=y)
            for idx, x, y in zip(
                dataset.data.index.to_numpy(),
                dataset_data_dummy.values,
                dataset.target.values,
            )
        ]
    )
