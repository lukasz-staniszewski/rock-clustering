from dataclasses import dataclass
from heapq import heapify, heappush
from typing import Any, List, Optional

import numpy as np
import pandas as pd

Heap = List[Any]


@dataclass
class ClusteringPoint:
    idx: int  # identifies point in dataset
    x: np.ndarray[bool]  # binary vector of features
    y: Optional[Any] = None  # target value


@dataclass
class RockInput:
    data: List[ClusteringPoint]


@dataclass
class ClusteringDataset:
    data: pd.DataFrame
    target: pd.Series
    metadata: Optional[pd.DataFrame] = None
    variables: Optional[pd.DataFrame] = None


@dataclass
class Cluster:
    points: List[ClusteringPoint]
