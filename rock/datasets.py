from dataclasses import dataclass
from typing import Optional

import pandas as pd
from ucimlrepo import fetch_ucirepo

MUSHROOM_DATASET = "mushroom"
CONGRESSIONAL_DATASET = "congressional"
CSV_DATASET = "csv"


@dataclass
class ClusteringDataset:
    data: pd.DataFrame
    target: pd.Series
    metadata: Optional[pd.DataFrame] = None
    variables: Optional[pd.DataFrame] = None


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


def get_csv_dataset(path: str) -> ClusteringDataset:
    df = pd.read_csv(path)
    target = None
    if "target" in df.columns:
        target = df["target"].squeeze()
        df = df.drop(columns=["target"])
    return ClusteringDataset(data=df, target=target)
