import numpy as np


class DistanceCalculator:
    @staticmethod
    def compute_radial_distances(reference_coordinates: np.ndarray, particles_coordinates: np.ndarray,
                                 box: np.ndarray) -> np.ndarray:
        distances = DistanceCalculator.compute_vectorial_distances(reference_coordinates, particles_coordinates, box)
        distances = np.sqrt(np.sum(np.square(distances), axis=1))
        return distances[distances > 0]

    @staticmethod
    def compute_vectorial_distances(reference_coordinates: np.ndarray, particles_coordinates: np.ndarray,
                                    box: np.ndarray) -> np.ndarray:
        distances = reference_coordinates - particles_coordinates
        distances -= box * np.rint(distances / box)
        return distances
