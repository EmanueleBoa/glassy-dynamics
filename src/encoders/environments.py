import numpy as np

from src.descriptors.distance_calculator import DistanceCalculator
from src.descriptors.radial_descriptor import RadialDescriptor
from src.entities.snapshot import Snapshot

MIN_DISTANCE = 0.85
MAX_DISTANCE1 = 2.0
DELTA1 = 0.025
MAX_DISTANCE2 = 3.0
DELTA2 = 0.05
MAX_DISTANCE3 = 5.0
DELTA3 = 0.1


class EnvironmentsEncoder:
    def encode_particles_environments(self, snapshot: Snapshot, target_species: str) -> np.ndarray:
        target_species_coordinates = snapshot.get_particles_coordinates(target_species=target_species)
        other_species = snapshot.get_other_species(target_species)
        other_species_coordinates = snapshot.get_particles_coordinates(target_species=other_species)
        encoded_environments = []
        for particle_coordinates in target_species_coordinates:
            same_species_descriptors = self._encode_particle_environment(
                particle_coordinates,
                target_species_coordinates,
                snapshot.box
            )
            other_species_descriptors = self._encode_particle_environment(
                particle_coordinates,
                other_species_coordinates,
                snapshot.box
            )
            encoded_environments.append(
                np.concatenate((same_species_descriptors, other_species_descriptors), axis=0)
            )
        return np.array(encoded_environments)

    @staticmethod
    def _encode_particle_environment(particle_coordinates: np.ndarray, particles_coordinates: np.ndarray,
                                     box: np.ndarray) -> np.ndarray:
        distances = DistanceCalculator.compute_radial_distances(particle_coordinates, particles_coordinates, box)
        descriptors = []
        descriptors.extend(
            RadialDescriptor.compute(distances, target_distance, DELTA1) for target_distance in
            np.arange(MIN_DISTANCE, MAX_DISTANCE1 + DELTA1, DELTA1)
        )
        descriptors.extend(
            RadialDescriptor.compute(distances, target_distance, DELTA1) for target_distance in
            np.arange(MAX_DISTANCE1 + DELTA2, MAX_DISTANCE2 + DELTA2, DELTA2)
        )
        descriptors.extend(
            RadialDescriptor.compute(distances, target_distance, DELTA1) for target_distance in
            np.arange(MAX_DISTANCE2 + DELTA3, MAX_DISTANCE3 + DELTA3, DELTA3)
        )
        return np.array(descriptors)
