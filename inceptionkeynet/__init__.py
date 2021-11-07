import os
import sys
import logging, coloredlogs
import progressbar

import inceptionkeynet.config as config



__GLOBAL_CONFIG_PATH = 'global'
__GLOBAL_CONFIG = config.get_config(__GLOBAL_CONFIG_PATH, expected_values=['terminal_log_level', 'default_format', 'dataset_format', 'data_format', 'datasets_path', 'processed_data_path', 'audio_sample_rate', 'models_path'], throw_on_missing_file=True)

TERMINAL_LOG_LEVEL = __GLOBAL_CONFIG['terminal_log_level']
DEFAULT_FORMAT = __GLOBAL_CONFIG['default_format']
DATASET_FORMAT = __GLOBAL_CONFIG['dataset_format']
DATA_FORMAT = __GLOBAL_CONFIG['data_format']

AUDIO_SAMPLE_RATE = __GLOBAL_CONFIG['audio_sample_rate']
DATASETS_PATH = os.path.join(*os.path.normpath(__GLOBAL_CONFIG['datasets_path']).split('\\'))
PROCESSED_DATA_PATH = os.path.join(*os.path.normpath(__GLOBAL_CONFIG['processed_data_path']).split('\\'))
MODELS_PATH = os.path.join(*os.path.normpath(__GLOBAL_CONFIG['models_path']).split('\\'))
BAYESIAN_OPTIMIZATION_PATH = os.path.join(*os.path.normpath(__GLOBAL_CONFIG['bayesian_path']).split('\\'))

MIREX_IGNORE_FIFTH_MODE = False
MIREX_ALLOW_DESCENDING_FIFTHS = True



progressbar.streams.wrap_stderr()

coloredlogs.install(level=TERMINAL_LOG_LEVEL)



sys.path.append(os.path.join(sys.path[0], 'submodules'))