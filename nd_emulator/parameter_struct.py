from typing import NamedTuple
import numpy as np


class Parameters(NamedTuple):
    max_depth: int
    spacing: np.ndarray
    dims: np.ndarray
    error_threshold: float
    model_classes: np.ndarray
    max_test_points: int
    relative_error: bool
    domain: np.ndarray
