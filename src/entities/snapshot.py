from typing import List

import numpy as np

from src.entities.particle import Particle


class Snapshot:
    def __init__(self, box: np.ndarray, particles: List[Particle]):
        self.box = box
        self.particles = particles

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r') as file:
            _ = file.readline()
            box = np.array(file.readline().split(), dtype=float)
            particles = [Particle.init_from_line(line) for line in file.readlines()]
        return cls(box, particles)
