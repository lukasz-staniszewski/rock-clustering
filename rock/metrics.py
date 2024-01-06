import numpy as np


def jaccard(a: np.ndarray[bool], b: np.ndarray[bool]) -> float:
    assert a.shape == b.shape
    return np.sum(a == b) / a.shape[0]


def is_jaccard_similar(a: np.ndarray[bool], b: np.ndarray[bool], theta_threshold: float = 0.5) -> bool:
    return jaccard(a, b) >= theta_threshold
