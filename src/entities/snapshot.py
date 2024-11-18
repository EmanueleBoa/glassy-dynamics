from typing import List, Optional

import numpy as np

from src.entities.particle import Particle


class Snapshot:
    def __init__(self, box: np.ndarray, particles: List[Particle]):
        self.box: np.ndarray = box
        self.particles: List[Particle] = particles
        self.available_species: List[str] = self.get_available_species()

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r') as file:
            _ = file.readline()
            box = np.array(file.readline().split(), dtype=float)
            particles = [Particle.init_from_line(line) for line in file.readlines()]
        return cls(box, particles)

    def get_particles_coordinates(self, target_species: Optional[str] = None) -> np.ndarray:
        if target_species is None:
            return np.array([particle.coordinates for particle in self.particles])

        return np.array([particle.coordinates for particle in self.particles if particle.species == target_species])

    def get_available_species(self) -> List[str]:
        return list(set([particle.species for particle in self.particles]))

    def get_other_species(self, species: str) -> str:
        if species == self.available_species[0]:
            return self.available_species[1]
        return self.available_species[0]
