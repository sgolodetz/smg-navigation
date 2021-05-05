import numpy as np

from scipy.interpolate import CubicSpline
from typing import List


class PathUtil:
    """Utility functions related to paths."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def interpolate(path: np.ndarray, *, smoothed_length: int = 100) -> np.ndarray:
        x: List[int] = np.arange(len(path))
        cs: CubicSpline = CubicSpline(x, path, bc_type='clamped')
        return cs(np.linspace(0, len(path) - 1, smoothed_length))

    @staticmethod
    def pull_strings(path: np.ndarray) -> np.ndarray:
        pulled_path: List[np.ndarray] = []

        i: int = 0
        while i < len(path):
            pulled_path.append(path[i, :])

            j: int = i + 2
            while j < len(path) and PathUtil.__is_traversible(path, i, j):
                j += 1

            i = j - 1

        return np.vstack(pulled_path)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __is_traversible(path: np.ndarray, i: int, j: int) -> bool:
        # TODO
        return True
