import os
import sys
import multiprocessing
import subprocess
import itertools

from inceptionkeynet.data_mining import AudioMiners
from inceptionkeynet.machine_learning import *



base_params = [
    Hyperparameters.USE_FULL_TRACKS_FOR_VALIDATION(True),
    Hyperparameters.USE_FULL_TRACKS_FOR_TESTING(True),

    # Preprocessing configuration
    Hyperparameters.SPECTROGRAM_MAX_LENGTH_FRAMES(100),

    # Model configuration
    Hyperparameters.MODEL_INPUT_TYPE(ModelInputType.SPECTROGRAM),
    Hyperparameters.MODEL_CLASSIFICATION_TYPE(ModelClassificationType.ROOT_MODE_COMBINED),

    
    # Training configuration
    Hyperparameters.EPOCHS(1000),
    Hyperparameters.CROSS_VALIDATION_FOLD_COUNT(5),
    Hyperparameters.EARLY_STOPPING(50),
    Hyperparameters.BATCH_SIZE(32),
    Hyperparameters.OPTIMIZER('adam'),
]

datasets = [
    # Dataset configuration
    Hyperparameters.TRAIN_DATASETS([Datasets.GIANTSTEPS_MTG_KEY.get_name(), Datasets.MCGILL_BILLBOARD.get_name()]),
    Hyperparameters.TEST_DATASETS([Datasets.GIANTSTEPS_KEY.get_name(), Datasets.KEYFINDER_V2.get_name(), Datasets.ROCKCORPUS.get_name(), Datasets.ISOPHONICS_BEATLES.get_name(), Datasets.ISOPHONICS_QUEEN.get_name(), Datasets.ISOPHONICS_ZWEIECK.get_name(), Datasets.ISOPHONICS_CAROLE_KING.get_name()]),
    Hyperparameters.AUDIO_SOURCE_WHITELIST([AudioMiners.GIANTSTEPS.name]),
    Hyperparameters.AUDIO_SOURCE_WHITELIST_STRICT(False), # Only apply whitelist if possible, do not exclude samples just because there is no sample from the right source available
]

inception_keynet = [
    Hyperparameters.MODEL_NAME('inceptionkeynet'),
    Hyperparameters.DROPOUT_RATE(.5),
]

cq_preprocessing = [
    Hyperparameters.TRANSFORMATION_TYPE(TimeFrequencyTransformationType.CONSTANT_Q),
    Hyperparameters.SPECTROGRAM_TIME_DOWNSCALING_FACTOR(17.236666666666), # We want a 5/s frequency, while our data is of frequency 86,183333333, so a "compression" of a factor of 17.23666... balances that out
]

pitch_shift_only = [
    Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE(6),
]
augmentation_scheme = [
    Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE(12),
    Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX(0),
    Hyperparameters.RANDOM_EQ_AUGMENTATION_SIGMA_MAX(72.5),
    Hyperparameters.RANDOM_EQ_AUGMENTATION_S_MAX(30),
    Hyperparameters.LOUDNESS_AUGMENTATION_FACTOR_RANGE(1),
    Hyperparameters.TIME_WARPING_DISTANCE_RANGE(13),
    Hyperparameters.TIME_MASKING_LENGTH_RANGE(6),
    Hyperparameters.FREQUENCY_MASKING_LENGTH_RANGE(2),
]

configurations = list(itertools.product([base_params], [datasets], [inception_keynet], [cq_preprocessing], [augmentation_scheme, pitch_shift_only]))



if __name__ == '__main__':
    n_trained_models = 0
    logging.getLogger(__name__).info(f'{len([hyperparameter_combination for configuration in configurations for hyperparameter_combination in Hyperparameters.combinations(itertools.chain(*configuration))])} hyperparameter combinations queued for training.')
    for configuration in configurations:
        configuration = itertools.chain(*configuration)
        for hyperparameter_combination in Hyperparameters.combinations(configuration):
            if not does_trained_model_exist(hyperparameter_combination):
                try:
                    ModelTrainer(hyperparameter_combination).train()

                    n_trained_models += 1
                except Exception:
                    logging.getLogger(__name__).exception('Unhandled exception while training model.')
            else:
                logging.getLogger(__name__).warning(f'Model for the following hyperparameters has already been trained, skipping:\n{repr(hyperparameter_combination)}')