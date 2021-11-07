import inspect
from typing import List
import logging
import time
import os

import inceptionkeynet.data
import inceptionkeynet.utils
import inceptionkeynet.config

import inceptionkeynet.data_mining.giantsteps as giantsteps

from inceptionkeynet.data_mining.common import AudioMiner



__DATA_MINING_CONFIG_PATH = 'data_mining/data_mining'
__DATA_MINING_CONFIG = inceptionkeynet.config.get_config(__DATA_MINING_CONFIG_PATH, expected_values=['download_delay', 'retry_inconclusive_results_from_previous_runs'], throw_on_missing_file=True)

DOWNLOAD_DELAY = __DATA_MINING_CONFIG['download_delay']
RETRY_INCONCLUSIVE_RESULTS_FROM_PREVIOUS_RUNS = __DATA_MINING_CONFIG['retry_inconclusive_results_from_previous_runs']



class AudioMiners:
    GIANTSTEPS = giantsteps.GiantStepsAudioMiner()
    
    @classmethod
    def list_audio_miners(cls) -> List[AudioMiner]:
        return [member[1] for member in inspect.getmembers(AudioMiners) if not member[0].startswith('_') and isinstance(member[1], AudioMiner)]



class DatasetAudioMiner:
    def __init__(self, dataset: inceptionkeynet.data.Dataset):
        self.dataset = dataset
        if not os.path.isfile(self.dataset.path):
            raise ValueError('The dataset to mine audio for must have a set path.')

    def run(self):
        for i, entry in enumerate(self.dataset):
            logging.getLogger(__name__).info(f'Mining audio for "{entry.name}" by "{entry.artist}" ({i + 1}/{len(self.dataset)})...')
            called_apis = False
            start_time = time.time()
            for miner in AudioMiners.list_audio_miners():
                if not miner.name in entry.files_relative or (entry.files_relative[miner.name] is None and RETRY_INCONCLUSIVE_RESULTS_FROM_PREVIOUS_RUNS):
                    called_apis = True
                    try:
                        
                        url = miner.try_get_download_url(entry)
                        entry.mining_sources[miner.name] = url
                        data = miner.try_get_audio_data(url)
                        if not data is None:
                            path = os.path.join(os.path.normpath(f'./data/sources/{miner.name}'), miner.get_filename_from_url(url))
                            with inceptionkeynet.utils.open_mkdirs(path, 'wb') as f:
                                f.write(data)
                            entry.files_relative[miner.name] = path
                        else:
                            entry.files_relative[miner.name] = None
                        
                        logging.getLogger(__name__).info(f' - {miner.name}: {entry.files_relative[miner.name]}')
                    except Exception:
                        logging.getLogger(__name__).exception(f'Unexpected error while trying to mine audio for "" by "" via "{miner.name}".')
                        logging.getLogger(__name__).info(f'Saving...')
                        self.dataset.save()
                        logging.getLogger(__name__).info(f'Done.')
                    except BaseException as e:
                        logging.getLogger(__name__).error(f'Audio mining process interrupted by user.')
                        logging.getLogger(__name__).info(f'Saving...')
                        self.dataset.save()
                        logging.getLogger(__name__).info(f'Done.')
                        raise e from None
                else:
                    logging.getLogger(__name__).info(f'Audio for "{entry.name}" by "{entry.artist}" already mined from {miner.name}, skipping.')
            if called_apis:
                if i % 100 == 0:
                    logging.getLogger(__name__).debug(f'Saving...')
                    self.dataset.save()
                logging.getLogger(__name__).debug(f'Sleeping to prevent rate limiting...')
                time.sleep(max(0, DOWNLOAD_DELAY - (time.time() - start_time)))
        logging.getLogger(__name__).debug(f'Saving...')
        self.dataset.save()



if __name__ == '__main__':
    print(AudioMiners.list_audio_miners())