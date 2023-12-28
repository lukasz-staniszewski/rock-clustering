from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd

from .data import ClusteringDataset
from .entities import ClusteringPoint, RockInput


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
