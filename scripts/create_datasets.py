import logging
import os

import numpy as np
import pandas as pd

from src.encoders.environments import EnvironmentsEncoder
from src.entities.snapshot import Snapshot

SOURCE_PATH = '../data/raw/'
OUTPUT_PATH = '../data/processed/'

TARGET_SPECIES = 'a'

logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')


def create_dataset(inputs_encoder: EnvironmentsEncoder, output_directory: str):
    source_path = f'{SOURCE_PATH}{output_directory}/'
    output_path = f'{OUTPUT_PATH}{output_directory}/'
    paths = sorted(
        [f'{source_path}{directory}/' for directory in os.listdir(source_path) if not directory.startswith('.')]
    )
    inputs = []
    targets = []
    for path in paths:
        logging.info(f'Loading snapshot from {path}')
        snapshot = Snapshot.load(path)

        logging.info('Encoding particles environments')
        inputs.extend(inputs_encoder.encode_particles_environments(snapshot, TARGET_SPECIES))

        logging.info('Extracting particles propensities')
        targets.extend(snapshot.get_particles_propensities(TARGET_SPECIES))

    logging.info(f'Saving dataset in {output_path}')
    df_inputs = pd.DataFrame(np.array(inputs))
    df_inputs.to_csv(f'{output_path}inputs_{TARGET_SPECIES}.csv', index=False)

    df_targets = pd.DataFrame(np.array(targets), columns=snapshot.times)
    df_targets.to_csv(f'{output_path}targets_{TARGET_SPECIES}.csv', index=False)


if __name__ == "__main__":
    encoder = EnvironmentsEncoder()

    logging.info('Creating train dataset')
    create_dataset(encoder, 'train')

    logging.info('Creating test dataset')
    create_dataset(encoder, 'test')
