from dataclasses import dataclass
from typing import Optional

import pandas as pd
from ucimlrepo import fetch_ucirepo

from .data import ClusteringDataset


def get_mushroom_dataset() -> ClusteringDataset:
    mushroom = fetch_ucirepo(id=73)
    return ClusteringDataset(
        data=mushroom.data.features,
        target=mushroom.data.targets.squeeze(),
        metadata=mushroom.metadata,
        variables=mushroom.variables,
    )


def get_congressional_dataset() -> ClusteringDataset:
    congressional_voting_records = fetch_ucirepo(id=105)
    return ClusteringDataset(
        data=congressional_voting_records.data.features,
        target=congressional_voting_records.data.targets.squeeze(),
        metadata=congressional_voting_records.metadata,
        variables=congressional_voting_records.variables,
    )
