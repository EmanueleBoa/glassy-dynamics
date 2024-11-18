import numpy as np


class Particle:
    def __init__(self, species: str, coordinates: np.ndarray, radius: float):
        self.species = species
        self.coordinates = coordinates
        self.radius = radius

    @classmethod
    def init_from_line(cls, file_line: str):
        attributes = file_line.split()
        species = str(attributes[0])
        coordinates = np.array(attributes[1:4], dtype=float)
        radius = float(attributes[-1])
        return cls(species, coordinates, radius)
