import numpy as np


class RadialDescriptor:
    @staticmethod
    def compute(distances: np.ndarray, target_distance: float, delta: float) -> float:
        return np.sum(
            np.exp(-np.square(distances - target_distance) / (2 * delta * delta))
        )
