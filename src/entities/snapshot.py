from typing import List, Optional

import numpy as np

from src.entities.particle import Particle

SNAPSHOT_FILE_NAME = 'snap.sph'
PROPENSITIES_FILE_NAME = 'meandistdata.dat'
TIMES_FILE_NAME = 'writetimes.dat'


class Snapshot:
    def __init__(self, box: np.ndarray, particles: List[Particle], propensities: np.ndarray, times: np.ndarray):
        self.box: np.ndarray = box
        self.particles: List[Particle] = particles
        self.available_species: List[str] = self._get_available_species()
        self.propensities = propensities
        self.times = times

    @classmethod
    def load(cls, path: str):
        snapshot_path = f'{path}{SNAPSHOT_FILE_NAME}'
        with open(snapshot_path, 'r') as file:
            _ = file.readline()
            box = np.array(file.readline().split(), dtype=float)
            particles = [Particle.init_from_line(line) for line in file.readlines()]

        propensities_path = f'{path}{PROPENSITIES_FILE_NAME}'
        with open(propensities_path, 'r') as file:
            propensities = np.array([line.split() for line in file.readlines()], dtype=float).T
            propensities = propensities[:, 1:]

        times_path = f'{path}{TIMES_FILE_NAME}'
        with open(times_path, 'r') as file:
            times = np.array(file.readlines(), dtype=float)
            times = times[1:]

        return cls(box, particles, propensities, times)

    def get_particles_coordinates(self, target_species: Optional[str] = None) -> np.ndarray:
        if target_species is None:
            return np.array([particle.coordinates for particle in self.particles])

        return np.array([particle.coordinates for particle in self.particles if particle.species == target_species])

    def get_particles_propensities(self, target_species: Optional[str] = None) -> np.ndarray:
        if target_species is None:
            return self.propensities

        return self.propensities[self._get_species_indices(target_species)]

    def get_other_species(self, species: str) -> str:
        if species == self.available_species[0]:
            return self.available_species[1]
        return self.available_species[0]

    def _get_available_species(self) -> List[str]:
        return list(set([particle.species for particle in self.particles]))

    def _get_species_indices(self, target_species: str) -> np.ndarray:
        return np.array([i for i, particle in enumerate(self.particles) if particle.species == target_species])
